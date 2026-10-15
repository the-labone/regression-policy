"""Strict conversion of DP's RoboMimic hybrid-image policy weights."""

import torch

from ht_regression.policies.from_scratch.action_chunk_policy import AffineNormalizer
from ht_regression.policies.from_scratch.encoders import MultiImageEncoder

TRANSFORMER = "diffusion_policy.policy.diffusion_transformer_hybrid_image_policy.DiffusionTransformerHybridImagePolicy"
UNET = "diffusion_policy.policy.diffusion_unet_hybrid_image_policy.DiffusionUnetHybridImagePolicy"
TARGETS = {TRANSFORMER, UNET}


def build_image_policy(config, state):
    # Reuse the state adapter's strict backbone conversion. Only the observation
    # encoder and the hybrid policy's flattened configuration differ.
    from .diffusion_policy import _build_model, _check_fields, _model_state

    policy = config["policy"]
    transformer = policy["_target_"] == TRANSFORMER
    architecture = (
        dict(
            n_layer=8,
            n_cond_layers=0,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            causal_attn=True,
        )
        if transformer
        else dict(
            diffusion_step_embed_dim=256,
            down_dims=(256, 512, 1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
        )
    )
    _check_fields(
        policy,
        {
            *architecture,
            "shape_meta",
            "noise_scheduler",
            "horizon",
            "n_action_steps",
            "n_obs_steps",
            "num_inference_steps",
            "crop_shape",
            "obs_encoder_group_norm",
            "eval_fixed_crop",
            "time_as_cond",
            "obs_as_cond",
            "obs_as_global_cond",
            "pred_action_steps_only",
            "eta",
            "use_clipped_model_output",
        },
        "hybrid image policy",
    )
    if not policy.get("obs_encoder_group_norm", False):
        raise ValueError("Image import requires DP's GroupNorm observation encoder.")
    crop = policy.get("crop_shape", (76, 76))
    if crop is not None and not policy.get("eval_fixed_crop", False):
        raise ValueError("Image import requires deterministic evaluation crops.")
    shapes = policy["shape_meta"]
    for section in (
        config.get("task", {}),
        config.get("task", {}).get("env_runner", {}),
    ):
        if "shape_meta" in section and section["shape_meta"] != shapes:
            raise ValueError("Checkpoint image shape metadata is inconsistent.")
    obs = shapes["obs"]
    cameras = {
        k: tuple(v["shape"])
        for k, v in obs.items()
        if v.get("type", "low_dim") == "rgb"
    }
    proprio = {
        k: tuple(v["shape"])
        for k, v in obs.items()
        if v.get("type", "low_dim") == "low_dim"
    }
    if list(obs) != [*cameras, *proprio] or any(len(s) != 1 for s in proprio.values()):
        raise ValueError(
            "Image import requires cameras followed by vector proprioception in checkpoint order."
        )
    if len(shapes["action"]["shape"]) != 1:
        raise ValueError("Image actions must be vectors.")
    encoder = MultiImageEncoder(
        cameras,
        sum(s[0] for s in proprio.values()),
        crop_shape=crop,
        proprio_keys=tuple(proprio),
    )
    consumed = set()

    def tensor(key):
        consumed.add(key)
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Expected a tensor at {key}.")
        return value

    for camera, network in encoder.cameras.items():
        prefix = f"obs_encoder.obs_nets.{camera}."
        converted = {}
        for key, expected in network.state_dict().items():
            if key == "pool.coordinates":
                x = tensor(prefix + "pool.pos_x")
                y = tensor(prefix + "pool.pos_y")
                value = torch.stack((x.flatten(), y.flatten()), dim=-1)
                if not torch.allclose(value, expected, rtol=0, atol=1e-7):
                    raise ValueError("Unsupported spatial-softmax coordinate grid.")
                converted[key] = value
            else:
                source = (
                    "backbone.nets." + key.removeprefix("trunk.")
                    if key.startswith("trunk.")
                    else "pool.nets." + key.removeprefix("pool.projection.")
                    if key.startswith("pool.projection.")
                    else "nets.3." + key.removeprefix("output.")
                )
                converted[key] = tensor(prefix + source)
        if not torch.equal(tensor(prefix + "pool.temperature"), torch.ones(1)):
            raise ValueError("Unsupported spatial-softmax temperature.")
        # RoboMimic registers the same backbone/pool twice. Verify aliases rather
        # than silently discarding possibly inconsistent checkpoint tensors.
        aliases = [
            (k, k.replace(prefix + "backbone.", prefix + "nets.0.", 1))
            for k in tuple(consumed)
            if k.startswith(prefix + "backbone.")
        ]
        aliases += [
            (k, k.replace(prefix + "pool.", prefix + "nets.1.", 1))
            for k in tuple(consumed)
            if k.startswith(prefix + "pool.")
        ]
        for original, alias in aliases:
            if not torch.equal(tensor(alias), state[original]):
                raise ValueError(f"Inconsistent duplicated visual weight: {alias}.")
        network.load_state_dict(converted, strict=True)

    def affine(field, name, dim):
        value = tensor(f"normalizer.params_dict.{field}.{name}")
        if value.ndim != 1 or value.numel() not in (1, dim):
            raise ValueError(f"Invalid {field} normalization shape.")
        return value.expand(dim)

    parameters = {}
    for name in ("scale", "offset"):
        parameters[f"obs_{name}"] = torch.cat(
            [affine(k, name, s[0]) for k, s in proprio.items()]
        )
        parameters[f"action_{name}"] = affine(
            "action", name, shapes["action"]["shape"][0]
        )
        for camera in cameras:
            if not torch.equal(
                affine(camera, name, 1),
                torch.tensor([2.0 if name == "scale" else -1.0]),
            ):
                raise ValueError(
                    "Image import requires DP RGB normalization to [-1, 1]."
                )
    for field in (*obs, "action"):
        for statistic in ("min", "max", "mean", "std"):
            consumed.add(f"normalizer.params_dict.{field}.input_stats.{statistic}")
    backbone_state = _model_state({k: v for k, v in state.items() if k not in consumed})
    dimensions = dict(
        action_dim=shapes["action"]["shape"][0],
        obs_dim=encoder.output_dim,
        horizon=policy["horizon"],
        n_obs_steps=policy["n_obs_steps"],
    )
    model = {k: policy.get(k, v) for k, v in architecture.items()}
    model["input_dim"] = dimensions["action_dim"]
    converted_policy = {**dimensions, "model": model}
    if transformer:
        converted_policy.update(
            _target_="diffusion_policy.policy.diffusion_transformer_lowdim_policy.DiffusionTransformerLowdimPolicy",
            obs_as_cond=policy.get("obs_as_cond", True),
            pred_action_steps_only=policy.get("pred_action_steps_only", False),
        )
        model.update(
            _target_="diffusion_policy.model.diffusion.transformer_for_diffusion.TransformerForDiffusion",
            output_dim=dimensions["action_dim"],
            cond_dim=encoder.output_dim,
            horizon=dimensions["horizon"],
            n_obs_steps=dimensions["n_obs_steps"],
            obs_as_cond=policy.get("obs_as_cond", True),
            time_as_cond=policy.get("time_as_cond", True),
        )
    else:
        converted_policy.update(
            _target_="diffusion_policy.policy.diffusion_unet_lowdim_policy.DiffusionUnetLowdimPolicy",
            obs_as_global_cond=policy.get("obs_as_global_cond", True),
            oa_step_convention=True,
        )
        model.update(
            _target_="diffusion_policy.model.diffusion.conditional_unet1d.ConditionalUnet1D",
            local_cond_dim=None,
            global_cond_dim=encoder.output_dim * dimensions["n_obs_steps"],
        )
    backbone, start = _build_model(converted_policy)
    backbone.load_state_dict(backbone_state, strict=True)
    return backbone, start, AffineNormalizer(**parameters), encoder, tuple(proprio)
