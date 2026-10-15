# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
"""PI0.5 network adapter with canonical noise-to-data time and predictions.

Adapted from LeRobot PI0.5; see LICENSE.pi05 for the upstream license.
"""

from collections.abc import Mapping

import torch
from torch import Tensor

from ..base import BasePolicy, PolicyTarget, SampleSpec, Timestep
from .processing import PI05Condition, additive_mask, attention_mask


class PI05Policy(BasePolicy):
    """Reuse a constructed LeRobot PI05Pytorch model without its loss or solver.

    Inputs contain images already resized/scaled for SigLIP, image_masks, tokens
    and token_mask. Tokenization and normalization stay in the external processor.
    This adapter uses canonical time s=1-t and returns -v_native: existing MSE/HT
    at s=0 therefore predict -v_native(0,1); flow predicts clean minus noise.

    Training uses joint prefix/suffix attention to preserve backbone gradients.
    Evaluation caches the prefix once per chunk; each suffix call clones that
    cache using the upstream helper. The model remains padded to max_action_dim;
    loss masks exclude only padded action dimensions, as in the source recipe.
    """

    def __init__(self, model, *, action_dim: int, n_action_steps: int | None = None):
        super().__init__()
        self.model = model
        cfg = model.config
        if getattr(cfg, "use_visual_memory", False) or getattr(
            cfg, "use_proprioceptive_memory", False
        ):
            raise ValueError(
                "This PI0.5 adapter currently supports single-observation checkpoints."
            )
        if getattr(cfg, "rtc_config", None) is not None:
            raise ValueError("RTC guidance is not supported by the shared objectives.")
        if getattr(cfg, "ht_gripper_bce", False):
            raise ValueError(
                "Bernoulli gripper readouts need a separate output adapter."
            )
        self.horizon = cfg.chunk_size
        self.action_dim = cfg.max_action_dim
        self.output_action_dim = action_dim
        self.n_action_steps = (
            cfg.n_action_steps if n_action_steps is None else n_action_steps
        )
        self.action_start_index = 0
        if type(action_dim) is not int or not 0 < action_dim <= self.action_dim:
            raise ValueError("action_dim must fit the model's padded action dimension.")
        if (
            type(self.n_action_steps) is not int
            or not 0 < self.n_action_steps <= self.horizon
        ):
            raise ValueError("n_action_steps must fit the model's chunk size.")
        self.train(model.training)

    @property
    def dtype(self):
        # LeRobot keeps action projections, generated trajectories and losses fp32,
        # even when the attention/vision modules use bf16.
        return self.model.action_in_proj.weight.dtype

    def encode_condition(self, observation: Mapping[str, Tensor]) -> PI05Condition:
        embeddings, padding, blocks = self.model.embed_prefix(
            observation["images"],
            observation["image_masks"],
            observation["tokens"],
            observation["token_mask"],
        )
        cache = None
        if not self.training:
            language = self.model.paligemma_with_expert.paligemma.model.language_model
            language.config._attn_implementation = "eager"
            _, cache = self.model.paligemma_with_expert.forward(
                attention_mask=additive_mask(attention_mask(padding, blocks)),
                position_ids=torch.cumsum(padding, dim=1) - 1,
                past_key_values=None,
                inputs_embeds=[embeddings, None],
                use_cache=True,
            )
        return PI05Condition(embeddings, padding, blocks, cache)

    def encode_target(self, batch) -> PolicyTarget:
        action = batch["action"]
        if action.ndim != 3 or action.shape[1:] != (
            self.horizon,
            self.output_action_dim,
        ):
            raise ValueError("action must have shape (B, chunk_size, real_action_dim).")
        if not action.is_floating_point():
            raise ValueError("action must be a normalized floating-point tensor.")
        action = action.to(device=self.device, dtype=self.dtype)
        target = torch.nn.functional.pad(
            action, (0, self.action_dim - self.output_action_dim)
        )
        weights = torch.zeros_like(target)
        weights[..., : self.output_action_dim] = 1
        # Native PI0.5 does not exclude padded temporal targets. Dataset padding
        # must therefore follow the source recipe; action_is_pad is not applied.
        return PolicyTarget(target, weights)

    def sample_spec(self, condition: PI05Condition) -> SampleSpec:
        return SampleSpec(
            (condition.padding_mask.shape[0], self.horizon, self.action_dim),
            self.device,
            self.dtype,
        )

    def _native_time(self, time: Timestep, sample: Tensor) -> Tensor:
        time = torch.as_tensor(time, device=sample.device)
        if time.ndim == 0 or time.shape == (1,):
            time = time.expand(sample.shape[0])
        if time.shape != (sample.shape[0],):
            raise ValueError("time must be scalar or shape (B,).")
        return (1 - time).float()

    def forward_features(self, sample, time, condition):
        if sample.shape != self.sample_spec(condition).shape:
            raise ValueError("Sample shape does not match the PI0.5 chunk.")
        suffix, suffix_pad, suffix_blocks, adarms = self.model.embed_suffix(
            sample, self._native_time(time, sample)
        )
        core = self.model.paligemma_with_expert
        if condition.past_key_values is not None:
            # Imported lazily: importing ht_regression never imports optional LeRobot.
            from lerobot.policies.common.vla_utils import clone_past_key_values

            prefix = condition.padding_mask[:, None, :].expand(-1, suffix.shape[1], -1)
            mask = torch.cat((prefix, attention_mask(suffix_pad, suffix_blocks)), dim=2)
            positions = (
                condition.padding_mask.sum(-1)[:, None] + suffix_pad.cumsum(1) - 1
            )
            core.gemma_expert.model.config._attn_implementation = "eager"
            outputs, _ = core.forward(
                attention_mask=additive_mask(mask),
                position_ids=positions,
                past_key_values=clone_past_key_values(condition.past_key_values),
                inputs_embeds=[None, suffix],
                use_cache=False,
                adarms_cond=[None, adarms],
            )
            features = outputs[1]
        else:
            prefix = condition.embeddings
            if (
                core.paligemma.model.language_model.layers[
                    0
                ].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                prefix, suffix = prefix.bfloat16(), suffix.bfloat16()
            padding = torch.cat((condition.padding_mask, suffix_pad), dim=1)
            blocks = torch.cat((condition.attention_mask, suffix_blocks), dim=1)
            mask = additive_mask(attention_mask(padding, blocks))
            positions = padding.cumsum(1) - 1

            def forward(prefix, suffix, mask, positions, adarms):
                outputs, _ = core.forward(
                    attention_mask=mask,
                    position_ids=positions,
                    past_key_values=None,
                    inputs_embeds=[prefix, suffix],
                    use_cache=False,
                    adarms_cond=[None, adarms],
                )
                return outputs[1]

            features = self.model._apply_checkpoint(
                forward, prefix, suffix, mask, positions, adarms
            )
        return features[:, -self.horizon :].float()

    def forward(self, sample, time, condition):
        features = self.forward_features(sample, time, condition)
        return -self.model._apply_checkpoint(self.model.action_out_proj, features)

    def decode_prediction(self, prediction):
        if prediction.ndim != 3 or prediction.shape[1:] != (
            self.horizon,
            self.action_dim,
        ):
            raise ValueError("Prediction must match the padded model chunk.")
        action = prediction[..., : self.output_action_dim]
        return {"action_pred": action, "action": action[:, : self.n_action_steps]}

    def get_extra_state(self):
        return {
            "version": 1,
            "time_convention": "noise_to_data",
            "horizon": self.horizon,
            "action_dim": self.action_dim,
            "output_action_dim": self.output_action_dim,
            "n_action_steps": self.n_action_steps,
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("PI0.5 policy settings differ from the saved checkpoint.")
