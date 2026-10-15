"""Restore local N1.7 checkpoints for native-processor, shared-policy inference."""

import inspect
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from ...policies.gr00t import GR00TActionHead, GR00TPolicy, GR00TPolicyWithScale
from ..inference.gr00t import GR00TInference
from .common import file_hash


@dataclass
class GR00TCheckpoint:
    native_policy: object
    inference: GR00TInference
    provenance: dict


@contextmanager
def _cached_processor(enabled):
    """Resolve cached VLM assets before native tokenizer metadata lookups."""
    if not enabled:
        yield
        return
    import gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 as processing
    from huggingface_hub import snapshot_download

    original = processing.build_processor

    def build(name, kwargs):
        local = (
            name
            if Path(name).is_dir()
            else snapshot_download(name, local_files_only=True)
        )
        return original(local, kwargs)

    processing.build_processor = build
    try:
        yield
    finally:
        processing.build_processor = original


def load_gr00t_checkpoint(path, *, embodiment, device="cuda", local_files_only=False):
    """Use the checkpoint's saved processors, normalization and native weights.

    HT/MSE inference is one mean prediction at zero time and zero input.
    The returned inference object has no training loss interface. The native
    sigma head remains on the policy for an explicit future training composition.
    """
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    path = Path(path).expanduser().resolve(strict=True)
    config = json.loads((path / "config.json").read_text())
    method = config.get("loss_type", "flow")
    if method not in ("flow", "hetero_t", "mse"):
        raise ValueError(f"Unsupported GR00T objective: {method}.")
    processor_dir = (
        path if (path / "processor_config.json").is_file() else path / "processor"
    )
    if not (processor_dir / "processor_config.json").is_file():
        raise FileNotFoundError("The saved GR00T processor is required.")
    with _cached_processor(local_files_only):
        native = Gr00tPolicy(
            embodiment_tag=embodiment, model_path=str(path), device=device, strict=True
        )
    head = GR00TActionHead(native.model.action_head)
    kwargs = dict(backbone=native.model.backbone)
    policy = (
        GR00TPolicyWithScale(
            head, sigma_decoder=native.model.action_head.sigma_decoder, **kwargs
        )
        if method == "hetero_t"
        else GR00TPolicy(head, **kwargs)
    )
    inference = GR00TInference(
        policy,
        method=method,
        num_inference_steps=native.model.action_head.num_inference_timesteps,
    )
    files = set(path.glob("*.json")) | set(processor_dir.rglob("*.json"))
    source = Path(inspect.getfile(type(native.model)))
    provenance = dict(
        checkpoint=str(path),
        embodiment=embodiment,
        loss_type=method,
        inference="flow_euler" if method == "flow" else "direct_mean",
        num_inference_steps=inference.sampler.num_inference_steps
        if method == "flow"
        else 1,
        action_horizon=policy.horizon,
        metadata_sha256={str(p.relative_to(path)): file_hash(p) for p in sorted(files)},
        native_model_source=str(source),
        native_model_sha256=file_hash(source),
    )
    return GR00TCheckpoint(native, inference, provenance)
