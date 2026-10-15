"""RoboMimic data, action-chunk models and rollout assembly for ILTrainer."""

import random
from dataclasses import asdict
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import call, instantiate
from torch.utils.data import DataLoader

from ...pipelines import PolicyPipeline
from .config import ExperimentConfig, LoaderConfig
from .trainer import ILTrainer


def build_trainer(cfg: ExperimentConfig, *, log_callback=None) -> ILTrainer:
    """Assemble a config already validated by ``training.robomimic.train.build_trainer``.

    Prepare data/normalization before the model, and optimizer after placement.
    """
    output = Path(cfg.training.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output_dir, including for resumed runs.")

    # Seed before data preparation and model initialization; order matters on resume.
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # The training split defines normalization, dimensions, and action alignment.
    dataset = instantiate(cfg.dataset, _convert_="all")
    validation = dataset.get_validation_dataset()
    policy_config = _resolve_policy_config(cfg.policy, dataset)
    pipeline = _build_pipeline(policy_config, cfg.objective, dataset)

    # Give each loader an independent generator so validation cannot reshuffle training.
    train_loader = _make_loader(
        dataset,
        cfg.dataloader.train,
        seed=cfg.seed,
        shuffle=True,
        device=cfg.training.device,
    )
    validation_loader = _make_loader(
        validation,
        cfg.dataloader.validation,
        seed=cfg.seed + 1,
        shuffle=False,
        device=cfg.training.device,
    )

    evaluation_factory = _build_evaluation_factory(cfg, dataset, output)
    resolved, metadata = _build_run_metadata(cfg, policy_config, dataset, validation)

    # ILTrainer owns device placement, optimizer/EMA creation, and state restoration.
    return ILTrainer(
        pipeline,
        train_loader,
        cfg.training,
        validation_loader=validation_loader,
        evaluation_factory=evaluation_factory,
        log_callback=log_callback,
        metadata=metadata,
        experiment_config=resolved,
        initializer=(
            partial(_initialize_pipeline, cfg.initialization, dataset)
            if cfg.initialization
            else None
        ),
    )


def _initialize_pipeline(config, dataset, pipeline):
    """Invoke a configured initialization tool on training data only."""
    return call(
        config, pipeline=pipeline, dataset=dataset, _recursive_=False, _convert_="all"
    )


def _resolve_policy_config(policy_config: dict, dataset) -> dict:
    """Infer model dimensions and check action-window alignment against the data."""
    model_config = dict(policy_config["model"])
    policy_config = dict(policy_config)
    visual = hasattr(dataset, "camera_shapes")
    encoder = policy_config.get("obs_encoder")
    if visual != (encoder is not None):
        raise ValueError(
            "Visual datasets and observation encoders must be selected together."
        )
    condition_dim = dataset.obs_dim
    if visual:
        encoder = dict(encoder)
        if "camera_shapes" in encoder:
            encoder["camera_shapes"] = {
                key: tuple(shape) for key, shape in encoder["camera_shapes"].items()
            }
        if "proprio_keys" in encoder:
            encoder["proprio_keys"] = tuple(encoder["proprio_keys"])
        for key, expected in (
            ("camera_shapes", dataset.camera_shapes),
            ("proprio_dim", dataset.obs_dim),
            ("proprio_keys", tuple(dataset.obs_keys)),
        ):
            if key in encoder and encoder[key] != expected:
                raise ValueError(f"Encoder {key} does not match dataset.")
            encoder[key] = expected
        condition_dim += len(dataset.camera_shapes) * encoder.get("feature_dim", 64)
        policy_config["obs_encoder"] = encoder
    for key, expected in (
        ("obs_dim", condition_dim),
        ("action_dim", dataset.action_dim),
        ("horizon", dataset.horizon),
        ("n_obs_steps", dataset.n_obs_steps),
    ):
        if key in model_config and model_config[key] != expected:
            raise ValueError(f"Policy {key} does not match the dataset ({expected}).")
        model_config[key] = expected
    if policy_config["n_action_steps"] != dataset.n_action_steps:
        raise ValueError("Policy execution chunk does not match dataset padding.")
    start = policy_config.get("action_start_index")
    if start is not None and start != dataset.n_obs_steps - 1:
        raise ValueError(
            "This dataset's training windows require action_start_index=n_obs_steps-1."
        )

    return {**policy_config, "model": model_config}


def _build_pipeline(
    policy_config: dict, objective_config: dict, dataset
) -> PolicyPipeline:
    """Attach training-set normalization to the policy, then compose its objective."""
    model = instantiate(policy_config["model"], _convert_="all")
    encoder = (
        instantiate(policy_config["obs_encoder"], _convert_="all")
        if policy_config.get("obs_encoder")
        else None
    )
    policy = instantiate(
        {
            key: value
            for key, value in policy_config.items()
            if key not in ("model", "obs_encoder")
        },
        model=model,
        obs_encoder=encoder,
        normalizer=dataset.get_normalizer(),
        _recursive_=False,
        _convert_="all",
    )
    objective = instantiate(objective_config, _convert_="all")
    return PolicyPipeline(policy, objective)


def _seed_worker(_worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def _make_loader(dataset, config: LoaderConfig, *, seed, shuffle, device) -> DataLoader:
    kwargs = {}
    if config.num_workers:
        kwargs.update(
            prefetch_factor=config.prefetch_factor,
            persistent_workers=config.persistent_workers,
            multiprocessing_context="spawn",
        )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory and torch.device(device).type == "cuda",
        drop_last=config.drop_last,
        generator=torch.Generator().manual_seed(seed),
        worker_init_fn=_seed_worker,
        **kwargs,
    )


def _build_evaluation_factory(cfg: ExperimentConfig, dataset, output: Path):
    """Bind rollout settings to the training data and validate them before fitting."""
    if not cfg.training.trainer.eval_every:
        return None

    visual = hasattr(dataset, "camera_shapes")
    from ...evaluation.robomimic import RobomimicImageRunner

    runner_type = hydra.utils.get_class(cfg.evaluation["_target_"])
    if visual != issubclass(runner_type, RobomimicImageRunner):
        raise ValueError("Evaluation runner modality must match the dataset.")
    extra = {"camera_shapes": dataset.camera_shapes} if visual else {}
    evaluation_factory = partial(
        _make_runner,
        cfg.evaluation,
        dataset_path=str(dataset.path),
        obs_keys=list(dataset.obs_keys),
        action_space=dataset.action_space,
        n_obs_steps=dataset.n_obs_steps,
        n_action_steps=dataset.n_action_steps,
        seed=cfg.training.evaluation_seed,
        **extra,
    )
    # Constructor validates metadata and rollout settings without stepping an
    # environment or creating files. Catch provenance errors before training.
    evaluation_factory(output / "evaluation" / "preflight")
    return evaluation_factory


def _build_run_metadata(
    cfg: ExperimentConfig, policy_config: dict, dataset, validation
):
    """Record resolved settings and data identity for reproducible checkpoint resume."""
    resolved = asdict(cfg)
    # Preserve resume provenance for the existing default RoboMimic recipe.
    if cfg.recipe == "ht_regression.training.robomimic.setup.build_trainer":
        resolved.pop("recipe")
    resolved["dataset"]["path"] = str(dataset.path)
    resolved["policy"] = policy_config
    provenance = {key: value for key, value in resolved.items() if key != "training"}
    if cfg.initialization is None:
        # Keep provenance compatible with pre-initialization DP/flow/MIP runs.
        provenance.pop("initialization")
    stat = dataset.path.stat()
    metadata = {
        "experiment": provenance,
        "dataset_file": {
            "path": str(dataset.path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "train_demos": list(dataset.demo_names),
        "validation_demos": list(validation.demo_names),
        "obs_keys": list(dataset.obs_keys),
        "obs_dim": dataset.obs_dim,
        "action_dim": dataset.action_dim,
    }
    return resolved, metadata


def _make_runner(config, output_dir, **kwargs):
    return instantiate(config, output_dir=str(output_dir), **kwargs, _convert_="all")
