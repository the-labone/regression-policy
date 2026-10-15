"""Restore a local Cosmos3 action checkpoint with its native inference loader."""

import inspect
import os
from dataclasses import dataclass
from pathlib import Path

from ...policies.cosmos import CosmosPolicy
from .common import file_hash


@dataclass
class CosmosCheckpoint:
    """Native model service plus explicit inference semantics and provenance.

    This is an evaluation adapter, not a shared-objective training pipeline:
    Cosmos Flow evolves video and actions jointly with the native UniPC solver.
    """

    service: object
    method: str
    provenance: dict
    policy: CosmosPolicy


def load_cosmos_checkpoint(
    path,
    *,
    method,
    service_factory,
    action_stats_path,
    vae_path,
    output_dir,
    seed=0,
    deterministic=False,
):
    """Load the historical Flow or video-supervised direct HT/MSE recipe.

    The benchmark supplies service_factory with its native inference settings.
    Requires the frozen much-ado Cosmos framework on PYTHONPATH. The caller
    explicitly identifies the training recipe; it is not inferred from a folder
    name. Normalization statistics must be supplied from that same recipe.
    """
    if method not in ("flow", "ht", "mse"):
        raise ValueError("Cosmos method must be flow, ht, or mse.")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must fit uint32.")
    checkpoint = Path(path).expanduser().resolve(strict=True)
    stats = Path(action_stats_path).expanduser().resolve(strict=True)
    vae = Path(vae_path).expanduser().resolve(strict=True)
    if not checkpoint.is_dir() or not stats.is_file() or not vae.is_file():
        raise ValueError(
            "Supply a checkpoint directory, action statistics, and VAE file."
        )
    if not any(checkpoint.rglob("*.safetensors")) and not any(
        checkpoint.rglob(".metadata")
    ):
        raise ValueError("Expected a native DCP or safetensors checkpoint.")
    # Historical evaluation excludes the training-only sigma head. Do not let
    # an inherited training environment silently create or calibrate a new head.
    if os.environ.get("HT_ACTION", "0") != "0":
        raise ValueError("Run Cosmos evaluation with HT_ACTION=0.")

    if deterministic:
        import torch

        if torch.cuda.is_initialized():
            raise RuntimeError(
                "Enable deterministic Cosmos inference before initializing CUDA."
            )
        # cuBLAS reads this when it first creates its CUDA workspaces.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    service = service_factory(
        checkpoint=checkpoint,
        method=method,
        action_stats_path=stats,
        vae_path=vae,
        output_dir=output_dir,
        seed=seed,
    )
    service.model.eval()
    if deterministic:
        # Native model initialization enables cuDNN benchmarking. Configure this
        # after loading so that the native loader cannot undo the setting.
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if not service.model.config.action_gen or not service.model.config.vision_gen:
        raise ValueError("This adapter requires the joint video/action Cosmos model.")
    sources = {}
    for obj in (type(service), type(service.model)):
        source = Path(inspect.getfile(obj)).resolve()
        sources[obj.__module__] = {"path": str(source), "sha256": file_hash(source)}
    metadata = {}
    for pattern in (".metadata", "*.json", "*.yaml"):
        for file in sorted(checkpoint.rglob(pattern)):
            metadata[str(file.relative_to(checkpoint))] = file_hash(file)
    provenance = dict(
        checkpoint=str(checkpoint),
        method=method,
        checkpoint_metadata_sha256=metadata,
        action_stats={"path": str(stats), "sha256": file_hash(stats)},
        vae_path=str(vae),
        native_source=sources,
        native_server_info=service.get_info(),
        video_supervision=True,
        action_input="native_noise" if method == "flow" else "zeros",
        action_output="integrated_velocity" if method == "flow" else "clean_action",
        num_inference_steps=service.cfg.num_steps,
        use_torch_compile=service.setup_args.use_torch_compile,
        inference_seed=seed,
        runtime=cosmos_runtime(),
    )
    policy = CosmosPolicy(
        service.model.net,
        prediction_type="velocity" if method == "flow" else "clean_action",
        time_scale=service.model.rectified_flow_video.noise_scheduler.config.num_train_timesteps,
    ).eval()
    return CosmosCheckpoint(service, method, provenance, policy)


def cosmos_runtime():
    """Record numerical settings rather than infer reproducibility from a seed."""
    import importlib.metadata

    import torch

    packages = {}
    for name in ("torch", "transformer-engine", "transformers", "numpy", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "packages": packages,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
