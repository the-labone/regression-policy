"""Restore a native RoboMimic policy for inference without its training dataset."""

import copy
from dataclasses import dataclass
from pathlib import Path

from ...pipelines import PolicyPipeline


@dataclass
class NativeCheckpoint:
    pipeline: PolicyPipeline
    config: dict
    metadata: dict
    training_config: dict
    state_key: str
    step: int


def load_native_checkpoint(
    path: str | Path, *, device="cpu", weights="auto"
) -> NativeCheckpoint:
    """Load saved weights and normalization; never refit or recalibrate on eval data."""
    from hydra.utils import instantiate

    from ...data.robomimic.normalizer import RobomimicNormalizer
    from ...training.checkpoint import load_checkpoint

    if weights not in {"auto", "model", "ema"}:
        raise ValueError("weights must be auto, model, or ema.")
    state = load_checkpoint(path)
    metadata = state["metadata"]
    if "experiment" not in metadata or "obs_keys" not in metadata:
        raise ValueError("Checkpoint lacks the RoboMimic experiment metadata.")
    config = copy.deepcopy(metadata["experiment"])
    state_key = (
        "ema"
        if weights == "ema" or (weights == "auto" and state["ema"] is not None)
        else "pipeline"
    )
    parameters = state[state_key]
    if parameters is None:
        raise ValueError("This checkpoint does not contain EMA weights.")
    policy_config = copy.deepcopy(config["policy"])
    normalizer = RobomimicNormalizer(
        metadata["obs_dim"],
        metadata["action_dim"],
        config["dataset"]["action_space"],
        normalization=config["dataset"].get("normalization", "dp"),
    )
    model = instantiate(policy_config.pop("model"), _convert_="all")
    encoder_config = policy_config.pop("obs_encoder", None)
    encoder = instantiate(encoder_config, _convert_="all") if encoder_config else None
    policy = instantiate(
        policy_config,
        model=model,
        obs_encoder=encoder,
        normalizer=normalizer,
        _recursive_=False,
        _convert_="all",
    )
    pipeline = PolicyPipeline(policy, instantiate(config["objective"], _convert_="all"))
    pipeline.load_state_dict(parameters, strict=True)
    pipeline.to(device).eval().requires_grad_(False)
    return NativeCheckpoint(
        pipeline=pipeline,
        config=config,
        metadata=copy.deepcopy(metadata),
        training_config=copy.deepcopy(state["config"]),
        state_key=state_key,
        step=state["progress"]["global_step"],
    )
