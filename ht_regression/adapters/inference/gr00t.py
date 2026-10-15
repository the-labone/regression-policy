"""Keep GR00T's observation processing and action decoding around ht_regression inference."""

from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

from ...objectives.common import validate_prediction
from ...objectives.generation.flow import FlowObjective


class GR00TInference(nn.Module):
    """Reuse the shared policy forward without claiming a training objective.

    Native HT needs an elementwise-scale likelihood that the scalar HTObjective
    does not implement. This object deliberately has no compute_loss interface.
    Flow reuses the shared Euler sampler; HT/MSE use one zero-input prediction.
    """

    def __init__(self, policy, *, method, num_inference_steps):
        super().__init__()
        if method not in ("flow", "hetero_t", "mse"):
            raise ValueError(f"Unsupported GR00T inference method: {method}.")
        self.policy, self.method = policy, method
        self.sampler = (
            FlowObjective(
                num_inference_steps=num_inference_steps,
                sample_mode="stochastic",
                inference_time_dtype="float64",
            )
            if method == "flow"
            else None
        )
        self.eval()

    def train(self, mode=True):
        if mode:
            raise RuntimeError(
                "GR00TInference is inference-only; construct a training policy and compatible objective explicitly."
            )
        return super().train(False)

    @torch.no_grad()
    def predict_action(self, observation):
        if self.policy.training:
            raise RuntimeError("Call policy.eval() before inference.")
        condition = self.policy.encode_condition(observation)
        spec = self.policy.sample_spec(condition)
        if self.sampler is not None:
            prediction = self.sampler.sample(self.policy, condition, sample_spec=spec)
        else:
            sample = torch.zeros(spec.shape, device=spec.device, dtype=spec.dtype)
            time = torch.zeros(spec.shape[0], device=spec.device, dtype=torch.float32)
            prediction = self.policy(sample, time, condition)
            validate_prediction(prediction, sample)
            prediction = prediction.to(spec.dtype)
        return self.policy.decode_prediction(prediction)


@contextmanager
def gr00t_policy_inference(bundle):
    """Temporarily replace only this native model's normalized action prediction.

    Native Gr00tPolicy and Gr00tSimPolicyWrapper retain their message conversion,
    saved normalization, relative-action decoding and simulator key conventions.
    RTC options are deliberately rejected: this evaluator uses synchronous chunks.
    """
    model, inference = bundle.native_policy.model, bundle.inference
    if getattr(model, "_ht_regression_gr00t_bound", False):
        raise RuntimeError("This GR00T model already has an active adapter.")
    if inference.training:
        raise ValueError("Call inference.eval() before binding inference.")

    @torch.no_grad()
    def get_action(inputs, options=None):
        if options:
            raise ValueError(
                "GR00T ht_regression evaluation does not support RTC options."
            )
        inputs = dict(inputs)
        if "vlm_content" in inputs:
            content = inputs.pop("vlm_content")
            if not isinstance(content, list):
                content = [content]
            inputs.update(
                model.collator([{"vlm_content": x} for x in content])["inputs"]
            )
        # The historical collator emits floating embodiment IDs; native action
        # inference also casts these to long before category-specific encoders.
        inputs["embodiment_id"] = inputs["embodiment_id"].long()
        return {"action_pred": inference.predict_action(inputs)["action_pred"]}

    missing = object()
    original = model.__dict__.get("get_action", missing)
    model._ht_regression_gr00t_bound = True
    try:
        model.get_action = get_action
        yield bundle.native_policy
    finally:
        if original is missing:
            del model.get_action
        else:
            model.get_action = original
        del model._ht_regression_gr00t_bound


def _embodiment_tag(value):
    from gr00t.data.embodiment_tags import EmbodimentTag

    return EmbodimentTag(value)


def _message(images, states, text, embodiment):
    # Optional GR00T dependency: importing the evaluator does not load a VLM.
    from gr00t.data.types import MessageType, VLAStepData

    step = VLAStepData(
        images=images,
        states=states,
        actions={},
        text=text,
        embodiment=_embodiment_tag(embodiment),
    )
    return [{"type": MessageType.EPISODE_STEP.value, "content": step}]


class GR00TAdapter:
    """Encode batched histories and decode against those SAME raw state histories."""

    def __init__(
        self,
        processor,
        *,
        embodiment_tag="robocasa_gr1_tabletop",
        language_key="annotation.human.coarse_action",
    ):
        self.processor = processor
        self.embodiment_tag = getattr(embodiment_tag, "value", embodiment_tag)
        self.language_key = language_key
        self.modalities = processor.get_modality_configs()[self.embodiment_tag]
        self.history_spec = {}
        for modality in ("video", "state"):
            cfg = self.modalities[modality]
            indices = tuple(cfg.delta_indices)
            if (
                not indices
                or indices[-1] != 0
                or any(type(i) is not int or i > 0 for i in indices)
                or any(a >= b for a, b in zip(indices, indices[1:]))
            ):
                raise ValueError(
                    f"{modality} history must be increasing nonpositive indices ending at 0."
                )
            for key in cfg.modality_keys:
                self.history_spec[f"{modality}.{key}"] = indices
        language = self.modalities["language"]
        if not language.modality_keys or list(language.delta_indices) != [0]:
            raise ValueError("Evaluation requires a current language instruction.")
        action_indices = list(self.modalities["action"].delta_indices)
        if not action_indices or action_indices != list(range(len(action_indices))):
            raise ValueError("Actions must be consecutive timesteps starting at zero.")
        self.action_horizon = len(action_indices)

    def validate_policy(self, policy):
        if policy.action_head.state_history_length != len(
            self.modalities["state"].delta_indices
        ):
            raise ValueError("Processor state history does not match the policy.")
        if policy.action_start_index + policy.n_action_steps > self.action_horizon:
            raise ValueError(
                "Execution chunk exceeds the processor's valid action horizon."
            )
        if self.action_horizon > policy.horizon:
            raise ValueError("Processor action horizon exceeds the policy horizon.")

    def prepare(self, observation):
        """Return collated model inputs and separate unnormalized decoding context."""
        batch_size = len(observation[self.language_key])
        for key, indices in self.history_spec.items():
            value = observation[key]
            is_video = key.startswith("video.")
            expected_rank = 5 if is_video else 3
            if value.ndim != expected_rank or value.shape[:2] != (
                batch_size,
                len(indices),
            ):
                raise ValueError(f"Invalid batched history shape for {key}.")
            if is_video:
                if value.dtype != np.uint8 or value.shape[-1] != 3:
                    raise ValueError(f"{key} must be uint8 RGB in (B,T,H,W,C) order.")
            elif (
                not np.issubdtype(value.dtype, np.floating)
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"{key} must contain finite floating-point states.")
        states = {
            key: observation[f"state.{key}"].astype(np.float32, copy=True)
            for key in self.modalities["state"].modality_keys
        }
        processed = []
        for i in range(batch_size):
            text = observation[self.language_key][i]
            if not isinstance(text, str):
                raise ValueError("Language instructions must be strings.")
            images = {
                key: observation[f"video.{key}"][i].copy()
                for key in self.modalities["video"].modality_keys
            }
            state = {key: value[i].copy() for key, value in states.items()}
            processed.append(
                self.processor(_message(images, state, text, self.embodiment_tag))
            )
        collated = self.processor.collator(processed)
        inputs = dict(collated["inputs"])
        inputs["embodiment_id"] = inputs["embodiment_id"].to(dtype=torch.long)
        return inputs, states

    def decode(self, prediction, states, *, start, count):
        normalized = prediction.detach().float().cpu().numpy()
        decoded = self.processor.decode_action(
            normalized, _embodiment_tag(self.embodiment_tag), states
        )
        keys = self.modalities["action"].modality_keys
        if set(decoded) != set(keys):
            raise ValueError(
                "Decoded action groups do not match the processor configuration."
            )
        actions = {}
        for key in keys:
            value = np.asarray(decoded[key], dtype=np.float32)
            if (
                value.ndim != 3
                or value.shape[0] != normalized.shape[0]
                or value.shape[1] < start + count
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"Invalid decoded action chunk for {key}.")
            actions[f"action.{key}"] = value[:, start : start + count].copy()
        return actions

    @torch.inference_mode()
    def predict(self, pipeline, observation):
        inputs, states = self.prepare(observation)
        # Decode the complete prediction first: processor action horizons can be
        # shorter than the padded model horizon. Slice exactly once afterwards.
        prediction = pipeline.predict_action(inputs)["action_pred"]
        return self.decode(
            prediction,
            states,
            start=pipeline.policy.action_start_index,
            count=pipeline.policy.n_action_steps,
        )
