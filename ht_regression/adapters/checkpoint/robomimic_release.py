"""Load the compact RoboMimic HT checkpoints published on Hugging Face."""

from dataclasses import dataclass
from pathlib import Path

import torch

from ...data.robomimic import RobomimicNormalizer
from ...objectives.regression.HT import HTObjective
from ...pipelines import PolicyPipeline
from ...policies.from_scratch.action_chunk_policy import ActionChunkPolicyWithScale
from ...policies.from_scratch.backbones import TransformerBackbone, UNetBackbone

FORMAT = "praxis.robomimic.ht.state.v1"
VISION_FORMAT = "praxis.robomimic.ht.vision.v1"


@dataclass
class ReleasedCheckpoint:
    pipeline: PolicyPipeline
    config: dict
    state_key: str = "state_dict"


def build_released_pipeline(config):
    """Build only the declared inference modules; never import saved targets."""
    backbones = {"ht-t": TransformerBackbone, "ht-c": UNetBackbone}
    if config["backbone"] not in backbones:
        raise ValueError(f"Unsupported release backbone: {config['backbone']}")
    model = backbones[config["backbone"]](**config["model"])
    normalizer = RobomimicNormalizer(**config["normalizer"])
    policy_config = dict(config["policy"])
    encoder_config = policy_config.pop("obs_encoder", None)
    visual = config["format"] == VISION_FORMAT
    if visual != (encoder_config is not None):
        raise ValueError("Release modality and observation encoder disagree.")
    encoder = None
    if visual:
        from ...policies.from_scratch.encoders import MultiImageEncoder

        encoder_config = dict(encoder_config)
        target = encoder_config.pop("_target_", None)
        if target is not None and target.rsplit(".", 1)[-1] != "MultiImageEncoder":
            raise ValueError("Unsupported release observation encoder.")
        encoder = MultiImageEncoder(**encoder_config)
    policy = ActionChunkPolicyWithScale(
        model, normalizer, obs_encoder=encoder, **policy_config
    )
    return PolicyPipeline(policy, HTObjective(**config["objective"]))


def load_released_checkpoint(path, *, device="cpu", weights="auto"):
    """Restore state or vision weights and normalization without a dataset."""
    if weights not in ("auto", "model", "ema"):
        raise ValueError("weights must be auto, model, or ema.")
    saved = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(saved, dict) or saved.get("format") not in (
        FORMAT,
        VISION_FORMAT,
    ):
        raise ValueError("Expected a released RoboMimic HT checkpoint.")
    config = saved["config"]
    if config.get("format") != saved["format"]:
        raise ValueError("Release configuration and checkpoint formats disagree.")
    available = saved["metrics"]["weights"]
    if available not in ("model", "ema") or weights not in ("auto", available):
        raise ValueError(f"This release contains only {available!r} weights.")
    pipeline = build_released_pipeline(config)
    pipeline.load_state_dict(saved["state_dict"], strict=True)
    pipeline.to(device).eval().requires_grad_(False)
    return ReleasedCheckpoint(pipeline, config)
