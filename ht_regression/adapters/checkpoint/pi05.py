"""Load a trusted local LeRobot PI0.5 checkpoint and its saved processors."""

import inspect
from dataclasses import dataclass
from pathlib import Path

from ...objectives.generation.flow import FlowObjective
from ...objectives.regression.HT import HTObjective
from ...objectives.regression.MSE import MSEObjective
from ...pipelines import PolicyPipeline
from ...policies.pi05 import PI05Policy, PI05PolicyWithScale
from .common import file_hash


@dataclass
class PI05Checkpoint:
    pipeline: PolicyPipeline
    native_policy: object
    preprocessor: object
    postprocessor: object
    provenance: dict


def load_pi05_checkpoint(
    path, *, device="cuda", n_action_steps=None, num_inference_steps=None
):
    """Load the matching normalization/tokenizer configuration without fitting it.

    Requires the much-ado LeRobot fork for HT/MSE checkpoints. No fallback to
    unrelated starting weights or freshly constructed processor statistics.
    """
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy as NativePolicy

    path = Path(path).expanduser().resolve(strict=True)
    if not (path / "model.safetensors").is_file():
        raise FileNotFoundError(
            "Expected a local PI0.5 pretrained_model/model.safetensors checkpoint."
        )
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        if not (path / name).is_file():
            raise FileNotFoundError(f"Missing checkpoint processor: {path / name}")
    config = PI05Config.from_pretrained(path)
    config.device = str(device)
    config.compile_model = False
    config.pretrained_path = path
    if n_action_steps is not None:
        config.n_action_steps = n_action_steps
    if num_inference_steps is not None:
        config.num_inference_steps = num_inference_steps
    method = getattr(config, "loss_type", "flow")
    if method not in ("flow", "mse", "hetero_t"):
        raise ValueError(f"Unsupported PI0.5 checkpoint objective: {method}.")
    from safetensors.torch import load_file

    # The historical from_pretrained catches loading failures and relaxes HT
    # strictness for fresh training heads. Evaluation must never do either.
    native = NativePolicy(config)
    weights = native._fix_pytorch_state_dict_keys(
        load_file(str(path / "model.safetensors")), config
    )
    weights = {
        key if key.startswith("model.") else f"model.{key}": value
        for key, value in weights.items()
    }
    weights = native._prepare_pretrained_state_dict(weights)
    native.load_state_dict(weights, strict=True)
    del weights
    native.to(device).eval()
    dim = config.output_features["action"].shape[0]
    policy_class = PI05PolicyWithScale if method == "hetero_t" else PI05Policy
    policy = policy_class(native.model, action_dim=dim)
    if method == "flow":
        objective = FlowObjective(
            num_inference_steps=config.num_inference_steps,
            sample_mode="stochastic",
            loss_scale=1 / config.max_action_dim,
            time_beta=(config.time_sampling_beta_alpha, config.time_sampling_beta_beta),
            time_scale=config.time_sampling_scale,
            time_offset=config.time_sampling_offset,
            time_flip=True,
            inference_time_dtype="float64",
        )
    elif method == "hetero_t":
        # Historical ht_df=2, ht_mvt=False is exactly joint nu=2*(50*7)=700.
        nu = (
            config.ht_df
            if getattr(config, "ht_mvt", False)
            else config.ht_df * config.chunk_size * dim
        )
        objective = HTObjective(nu=nu, scale_bias=config.ht_sbias, min_scale=1e-3)
    else:
        objective = MSEObjective()
    pre, post = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    files = [
        p
        for p in path.iterdir()
        if p.is_file() and p.suffix in (".json", ".safetensors")
    ]
    files += [p for p in (path / "tokenizer").rglob("*") if p.is_file()]
    hashes = {}
    for file in sorted(files):
        hashes[str(file.relative_to(path))] = file_hash(file)
    source_files = {
        obj.__module__: Path(inspect.getfile(obj)).resolve()
        for obj in (NativePolicy, PI05Config, make_pre_post_processors)
    }
    provenance = dict(
        checkpoint=str(path),
        loss_type=method,
        file_sha256=hashes,
        native_source={
            name: {"path": str(file), "sha256": file_hash(file)}
            for name, file in source_files.items()
        },
        n_action_steps=config.n_action_steps,
        chunk_size=config.chunk_size,
        num_inference_steps=config.num_inference_steps if method == "flow" else 1,
    )
    return PI05Checkpoint(
        PolicyPipeline(policy, objective).eval(), native, pre, post, provenance
    )
