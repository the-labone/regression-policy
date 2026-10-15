"""Import original DP state and hybrid-image checkpoints without a workspace.

Only observation-conditioned Transformer and global-conditioned U-Net policies
are supported. The loader instantiates known classes and loads model weights
strictly; it does not instantiate a training workspace or load optimizer state.
"""

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from ht_regression.objectives.generation.diffusion import DiffusionObjective
from ht_regression.objectives.generation.diffusion.objective import (
    _scheduler_from_config,
)
from ht_regression.pipelines import PolicyPipeline
from ht_regression.policies.from_scratch.action_chunk_policy import (
    ActionChunkPolicy,
    AffineNormalizer,
)

_POLICY_PREFIX = "diffusion_policy.policy."
_MODEL_PREFIX = "diffusion_policy.model.diffusion."


@dataclass
class LoadedDiffusionCheckpoint:
    policy: ActionChunkPolicy
    objective: DiffusionObjective
    config: dict
    state_key: str
    obs_keys: tuple[str, ...] | None
    action_space: Literal["delta", "abs"] | None

    def make_pipeline(self) -> PolicyPipeline:
        """Compose the loaded model and recorded objective for evaluation."""
        return PolicyPipeline(self.policy, self.objective).eval()


def _plain_config(config) -> dict:
    from omegaconf import OmegaConf

    if OmegaConf.is_config(config):
        # Released DP checkpoints contain resolved configs. Avoid running custom
        # resolvers (especially the upstream `eval` resolver) during import.
        config = OmegaConf.to_container(config, resolve=False)
    if not isinstance(config, Mapping):
        raise ValueError("Checkpoint cfg must be a mapping.")
    return copy.deepcopy(dict(config))


def _require_resolved(value):
    if isinstance(value, Mapping):
        for child in value.values():
            _require_resolved(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _require_resolved(child)
    elif isinstance(value, str) and "${" in value:
        raise ValueError(
            "Unresolved checkpoint config; pass a resolved mapping via config=."
        )


def _check_fields(config, allowed, label):
    unknown = {key for key in config if not key.startswith("_")} - set(allowed)
    if unknown:
        raise ValueError(f"Unsupported {label} configuration fields: {sorted(unknown)}")


def _build_model(policy):
    # Original DP checkpoints select these architectures; the objective itself
    # accepts any backbone implementing the common tensor contract.
    from ht_regression.policies.from_scratch.backbones import (
        TransformerBackbone,
        UNetBackbone,
    )

    dimensions = {
        key: policy[key] for key in ("action_dim", "obs_dim", "horizon", "n_obs_steps")
    }
    model = policy["model"]
    target = policy.get("_target_")
    if policy.get("pred_action_steps_only", False):
        raise ValueError(
            "pred_action_steps_only checkpoints are not supported; use full-horizon policies."
        )
    if model.get("input_dim") != dimensions["action_dim"]:
        raise ValueError("Backbone input_dim must match policy action_dim.")
    if (
        target
        == _POLICY_PREFIX
        + "diffusion_transformer_lowdim_policy.DiffusionTransformerLowdimPolicy"
    ):
        if (
            model.get("_target_")
            != _MODEL_PREFIX + "transformer_for_diffusion.TransformerForDiffusion"
        ):
            raise ValueError("Unsupported Transformer model target.")
        if (
            not policy.get("obs_as_cond", False)
            or not model.get("obs_as_cond", False)
            or not model.get("time_as_cond", True)
        ):
            raise ValueError(
                "Only observation-conditioned Transformer decoders with a time token are supported."
            )
        expected = {
            "output_dim": dimensions["action_dim"],
            "cond_dim": dimensions["obs_dim"],
            "horizon": dimensions["horizon"],
            "n_obs_steps": dimensions["n_obs_steps"],
        }
        if any(model.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "Transformer dimensions do not match the policy configuration."
            )
        # Defaults here are UPSTREAM defaults, not the ht_regression backbone defaults.
        defaults = dict(
            n_layer=12,
            n_head=12,
            n_emb=768,
            p_drop_emb=0.1,
            p_drop_attn=0.1,
            causal_attn=False,
            n_cond_layers=0,
        )
        _check_fields(
            model,
            {*defaults, *expected, "input_dim", "obs_as_cond", "time_as_cond"},
            "Transformer",
        )
        backbone = TransformerBackbone(
            **dimensions,
            **{key: model.get(key, value) for key, value in defaults.items()},
        )
        start = dimensions["n_obs_steps"] - 1
    elif (
        target
        == _POLICY_PREFIX + "diffusion_unet_lowdim_policy.DiffusionUnetLowdimPolicy"
    ):
        if (
            model.get("_target_")
            != _MODEL_PREFIX + "conditional_unet1d.ConditionalUnet1D"
        ):
            raise ValueError("Unsupported U-Net model target.")
        if not policy.get("obs_as_global_cond", False) or policy.get(
            "obs_as_local_cond", False
        ):
            raise ValueError(
                "Only global-observation-conditioned U-Net checkpoints are supported."
            )
        if (
            model.get("local_cond_dim") is not None
            or model.get("global_cond_dim")
            != dimensions["obs_dim"] * dimensions["n_obs_steps"]
        ):
            raise ValueError(
                "U-Net conditioning dimensions do not match the policy configuration."
            )
        if model.get("n_groups", 8) != 8:
            raise ValueError(
                "DP U-Net checkpoints require n_groups=8 for final GroupNorm parity."
            )
        defaults = dict(
            diffusion_step_embed_dim=256,
            down_dims=(256, 512, 1024),
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False,
        )
        _check_fields(
            model,
            {*defaults, "input_dim", "local_cond_dim", "global_cond_dim"},
            "U-Net",
        )
        backbone = UNetBackbone(
            **dimensions,
            **{key: model.get(key, value) for key, value in defaults.items()},
        )
        start = dimensions["n_obs_steps"] - int(policy.get("oa_step_convention", False))
    else:
        raise ValueError(f"Unsupported DP policy target: {target!r}.")
    return backbone, start


def _normalizer(state, obs_dim, action_dim):
    parameters = {}
    for field, dimension in (("obs", obs_dim), ("action", action_dim)):
        for name in ("scale", "offset"):
            key = f"normalizer.params_dict.{field}.{name}"
            if key not in state:
                raise ValueError(
                    f"Checkpoint is missing normalization parameter {key}."
                )
            value = state[key]
            if (
                not isinstance(value, Tensor)
                or value.ndim != 1
                or value.numel() not in (1, dimension)
            ):
                raise ValueError(f"Checkpoint normalization shape mismatch for {key}.")
            parameters[f"{field}_{name}"] = value.expand(dimension)
    return AffineNormalizer(**parameters)


def _metadata(config, action_dim):
    task = config.get("task", {})
    dataset = task.get("dataset", {})
    runner = task.get("env_runner", {})
    _require_resolved([task.get("obs_keys"), task.get("abs_action"), dataset, runner])
    keys = [
        tuple(section["obs_keys"])
        for section in (task, dataset, runner)
        if "obs_keys" in section
    ]
    if keys and (
        any(value != keys[0] for value in keys) or len(set(keys[0])) != len(keys[0])
    ):
        raise ValueError("Checkpoint observation key ordering is inconsistent.")
    flags = [
        section["abs_action"]
        for section in (task, dataset, runner)
        if "abs_action" in section
    ]
    if any(type(value) is not bool for value in flags) or (
        flags and any(value != flags[0] for value in flags)
    ):
        raise ValueError(
            "Checkpoint dataset/runner abs_action settings are inconsistent."
        )
    action_space = ("abs" if flags[0] else "delta") if flags else None
    is_robomimic = any(
        "robomimic" in section.get("_target_", "") for section in (dataset, runner)
    )
    if is_robomimic and action_space is not None:
        if action_dim not in ((10, 20) if action_space == "abs" else (7, 14)):
            raise ValueError(
                "RoboMimic action dimensions do not match checkpoint action_space."
            )
        if (
            action_space == "abs"
            and dataset.get("rotation_rep", "rotation_6d") != "rotation_6d"
        ):
            raise ValueError("Only rotation_6d ABS checkpoints are supported.")
    if config.get("n_latency_steps", 0) or runner.get("n_latency_steps", 0):
        raise ValueError("Checkpoints with action latency are not supported.")
    if config.get("past_action_visible", False) or runner.get("past_action", False):
        raise ValueError("Checkpoints conditioned on past actions are not supported.")
    return keys[0] if keys else None, action_space


def _select_weights(payload: Mapping, config: Mapping, weights: str):
    """Select the requested online/EMA state without silently falling back."""
    if weights not in ("auto", "model", "ema"):
        raise ValueError("weights must be 'auto', 'model', or 'ema'.")
    use_ema = config.get("training", {}).get("use_ema", False)
    if type(use_ema) is not bool:
        raise ValueError("training.use_ema must be a resolved boolean.")
    state_key = (
        "ema_model" if weights == "ema" or (weights == "auto" and use_ema) else "model"
    )
    if state_key not in payload["state_dicts"]:
        raise ValueError(f"Checkpoint is missing requested {state_key!r} weights.")
    state = payload["state_dicts"][state_key]
    return state_key, state


def _model_state(state: Mapping) -> dict:
    """Strip known workspace fields and reject everything outside the DP schema."""
    model_state = {
        key.removeprefix("model."): value
        for key, value in state.items()
        if key.startswith("model.") and key != "model._dummy_variable"
    }
    dummy_keys = {
        "_dummy_variable",
        "model._dummy_variable",
        "mask_generator._dummy_variable",
    }
    statistics = {
        f"normalizer.params_dict.{field}.{suffix}"
        for field in ("obs", "action")
        for suffix in (
            "scale",
            "offset",
            "input_stats.min",
            "input_stats.max",
            "input_stats.mean",
            "input_stats.std",
        )
    }
    unexpected = (
        set(state) - {f"model.{key}" for key in model_state} - dummy_keys - statistics
    )
    if unexpected:
        raise ValueError(f"Unexpected DP policy state keys: {sorted(unexpected)}")
    if any(state[key].numel() for key in dummy_keys & state.keys()):
        raise ValueError("Expected empty DP device-marker tensors.")
    return model_state


def load_dp_checkpoint(
    checkpoint: str | Path | Mapping,
    *,
    device: str | torch.device = "cpu",
    weights: Literal["auto", "model", "ema"] = "auto",
    config: Mapping | None = None,
) -> LoadedDiffusionCheckpoint:
    """Load a trusted original DP workspace checkpoint, ready for inference.

    ``auto`` follows cfg.training.use_ema and fails if that state is absent.
    A path is loaded with pickle/dill, as required by original DP's OmegaConf
    payload: only use trusted files. An already-loaded payload is also accepted.
    ``config`` can supply an explicitly resolved config for custom checkpoints.
    Normalization is always recovered from the selected policy state, never fit.
    """
    if isinstance(checkpoint, Mapping):
        payload = checkpoint
    else:
        import dill

        payload = torch.load(
            checkpoint, map_location="cpu", pickle_module=dill, weights_only=False
        )
    if not isinstance(payload, Mapping) or "state_dicts" not in payload:
        raise ValueError(
            "Expected an original DP workspace payload with cfg and state_dicts."
        )
    config = _plain_config(payload.get("cfg") if config is None else config)
    policy_config = config["policy"]
    _require_resolved(policy_config)
    from .diffusion_policy_image import TARGETS, build_image_policy

    visual = policy_config.get("_target_") in TARGETS
    if not visual:
        _check_fields(
            policy_config,
            {
                "model",
                "noise_scheduler",
                "horizon",
                "obs_dim",
                "action_dim",
                "n_action_steps",
                "n_obs_steps",
                "num_inference_steps",
                "obs_as_cond",
                "obs_as_global_cond",
                "obs_as_local_cond",
                "pred_action_steps_only",
                "oa_step_convention",
                "eta",
                "use_clipped_model_output",
            },
            "policy",
        )
    state_key, state = _select_weights(payload, config, weights)
    action_dim = (
        policy_config["shape_meta"]["action"]["shape"][0]
        if visual
        else policy_config["action_dim"]
    )
    obs_keys, action_space = _metadata(config, action_dim)
    if visual:
        model, start, normalizer, encoder, obs_keys = build_image_policy(config, state)
    else:
        model, start = _build_model(policy_config)
        model.load_state_dict(_model_state(state), strict=True)
        normalizer = _normalizer(state, model.obs_dim, model.action_dim)
        encoder = None
    scheduler_config = policy_config["noise_scheduler"]
    scheduler_target = scheduler_config.get("_target_")
    scheduler_names = {
        "diffusers.schedulers.scheduling_ddpm.DDPMScheduler": "DDPMScheduler",
        "diffusers.schedulers.scheduling_ddim.DDIMScheduler": "DDIMScheduler",
    }
    if scheduler_target not in scheduler_names:
        raise ValueError(f"Unsupported scheduler target: {scheduler_target!r}.")
    scheduler = _scheduler_from_config(
        scheduler_names[scheduler_target], scheduler_config
    )
    policy = (
        ActionChunkPolicy(
            model,
            normalizer,
            n_action_steps=policy_config["n_action_steps"],
            action_start_index=start,
            obs_encoder=encoder,
        )
        .to(device)
        .eval()
    )
    objective = DiffusionObjective(
        noise_scheduler=scheduler,
        num_inference_steps=policy_config.get("num_inference_steps"),
        scheduler_step_kwargs={
            key: policy_config[key]
            for key in ("eta", "use_clipped_model_output")
            if key in policy_config
        },
    ).eval()
    return LoadedDiffusionCheckpoint(
        policy, objective, config, state_key, obs_keys, action_space
    )
