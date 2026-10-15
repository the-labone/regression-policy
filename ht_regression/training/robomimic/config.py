"""Typed experiment and trainer settings, Hydra registration, and preflight validation.

Experiment values belong in YAML. These schemas cover experiment assembly and training infrastructure;
policy, objective and dataset schemas belong to their respective components.
"""

import math
from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf


@dataclass
class OptimizerConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-6
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    warmup_steps: int = 0
    # Preserve the LR schedule when stopping training before its full duration.
    total_steps: int | None = None  # None uses trainer.max_steps.


@dataclass
class EMAConfig:
    enabled: bool = True
    schedule: str = "constant"
    decay: float = 0.995
    # DP power-law warmup; decay is the maximum when schedule="power".
    update_after_step: int = 0
    inv_gamma: float = 1.0
    power: float = 0.75
    min_decay: float = 0.0


@dataclass
class MetricsConfig:
    monitor_key: str | None = None
    mode: str = "max"
    protocols: list[str] = field(default_factory=list)


@dataclass
class TrainerConfig:
    max_steps: int = MISSING  # Successful optimizer updates, not microbatches.
    gradient_accumulation_steps: int = 1
    precision: str = "fp32"
    max_grad_norm: float | None = None
    log_every: int = 10
    validate_every: int = 0
    eval_every: int = 0
    checkpoint_every: int = 1000
    ema: EMAConfig = field(default_factory=EMAConfig)


@dataclass
class LoggingConfig:
    tensorboard: bool = False
    flush_secs: int = 30
    max_queue: int = 100


@dataclass
class CompileConfig:
    # Compile only the online loss computation, after EMA construction / restore.
    enabled: bool = False
    mode: str = "reduce-overhead"


@dataclass
class TrainingConfig:
    evaluation_seed: int = 0
    device: str = "cpu"
    allow_tf32: bool | None = None  # None preserves the caller/runtime settings.
    output_dir: str = MISSING
    resume: str | None = None
    # Multiworker epoch-boundary recovery restores training state, not worker RNG.
    resume_mode: str = "exact"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)


@dataclass
class LoaderConfig:
    batch_size: int = 256
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last: bool = False


@dataclass
class LoadersConfig:
    train: LoaderConfig = field(default_factory=LoaderConfig)
    validation: LoaderConfig = field(default_factory=LoaderConfig)


@dataclass
class ExperimentConfig:
    recipe: str = "ht_regression.training.robomimic.setup.build_trainer"
    name: str = "diffusion_policy"
    seed: int = 42
    dataset: dict[str, Any] = MISSING
    policy: dict[str, Any] = MISSING
    objective: dict[str, Any] = MISSING
    initialization: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    dataloader: LoadersConfig = field(default_factory=LoadersConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def register_configs() -> None:
    """Explicitly register a schema that YAML can extend in its defaults list."""
    ConfigStore.instance().store(group="training", name="schema", node=TrainingConfig)


def validate_trainer(config: TrainerConfig) -> None:
    for name in ("max_steps", "gradient_accumulation_steps", "log_every"):
        value = getattr(config, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"trainer.{name} must be a positive integer.")
    for name in ("validate_every", "eval_every", "checkpoint_every"):
        value = getattr(config, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"trainer.{name} must be a nonnegative integer.")
    if config.precision not in ("fp32", "bf16", "fp16"):
        raise ValueError("trainer.precision must be fp32, bf16, or fp16.")
    if config.max_grad_norm is not None and (
        not math.isfinite(config.max_grad_norm) or config.max_grad_norm <= 0
    ):
        raise ValueError("trainer.max_grad_norm must be finite and positive.")
    if not math.isfinite(config.ema.decay) or not 0 <= config.ema.decay < 1:
        raise ValueError("trainer.ema.decay must lie in [0, 1).")
    ema = config.ema
    if ema.schedule not in ("constant", "power"):
        raise ValueError("trainer.ema.schedule must be constant or power.")
    if ema.update_after_step < 0 or not 0 <= ema.min_decay <= ema.decay:
        raise ValueError(
            "EMA requires update_after_step >= 0 and 0 <= min_decay <= decay."
        )
    if any(
        not math.isfinite(value) or value <= 0 for value in (ema.inv_gamma, ema.power)
    ):
        raise ValueError("EMA inv_gamma and power must be finite and positive.")


def validate_config(config: DictConfig | TrainingConfig) -> TrainingConfig:
    """Validate the final composition, including CLI overrides/interpolations.

    Explicit invocation is intentional: Hydra normally produces a DictConfig,
    so dataclass __post_init__ is not a reliable validation entry point.
    """
    merged = OmegaConf.merge(OmegaConf.structured(TrainingConfig), config)
    OmegaConf.resolve(merged)
    missing = OmegaConf.missing_keys(merged)
    if missing:
        raise ValueError(f"Missing required configuration: {sorted(missing)}")
    cfg = OmegaConf.to_object(merged)
    validate_trainer(cfg.trainer)
    if cfg.resume_mode not in ("exact", "epoch_boundary"):
        raise ValueError("resume_mode must be exact or epoch_boundary.")
    if cfg.compile.mode not in ("default", "reduce-overhead", "max-autotune"):
        raise ValueError(
            "compile.mode must be default, reduce-overhead, or max-autotune."
        )
    for name in ("flush_secs", "max_queue"):
        if getattr(cfg.logging, name) < 1:
            raise ValueError(f"logging.{name} must be a positive integer.")
    if not 0 <= cfg.evaluation_seed < 2**32:
        raise ValueError("evaluation_seed must be an integer in [0, 2**32).")
    if not cfg.output_dir.strip():
        raise ValueError("output_dir cannot be empty.")
    if (
        cfg.device != "cpu"
        and cfg.device != "cuda"
        and not (cfg.device.startswith("cuda:") and cfg.device[5:].isdigit())
    ):
        raise ValueError("device must be cpu, cuda, or cuda:<index>.")
    if cfg.trainer.precision == "fp16" and cfg.device == "cpu":
        raise ValueError("fp16 training requires CUDA; use fp32 or bf16 on CPU.")
    opt = cfg.optimizer
    if any(not math.isfinite(v) or v <= 0 for v in (opt.lr, opt.eps)):
        raise ValueError("optimizer lr and eps must be finite and positive.")
    if not math.isfinite(opt.weight_decay) or opt.weight_decay < 0:
        raise ValueError("optimizer.weight_decay must be finite and nonnegative.")
    if len(opt.betas) != 2 or any(not 0 <= beta < 1 for beta in opt.betas):
        raise ValueError("optimizer.betas must contain two values in [0, 1).")
    if cfg.scheduler.name not in ("constant", "cosine"):
        raise ValueError("scheduler.name must be constant or cosine.")
    if not 0 <= cfg.scheduler.warmup_steps < cfg.trainer.max_steps:
        raise ValueError("Require 0 <= scheduler.warmup_steps < trainer.max_steps.")
    if cfg.scheduler.total_steps is not None and (
        type(cfg.scheduler.total_steps) is not int
        or cfg.scheduler.total_steps < cfg.trainer.max_steps
    ):
        raise ValueError(
            "scheduler.total_steps must be an integer >= trainer.max_steps."
        )
    if cfg.metrics.mode not in ("min", "max"):
        raise ValueError("metrics.mode must be min or max.")
    if cfg.metrics.monitor_key is not None and not cfg.metrics.monitor_key.strip():
        raise ValueError("metrics.monitor_key cannot be empty.")
    if cfg.metrics.protocols:
        from ..metrics import METRIC_PROTOCOLS

        if not cfg.metrics.monitor_key:
            raise ValueError("Metric protocols require metrics.monitor_key.")
        unknown = set(cfg.metrics.protocols) - METRIC_PROTOCOLS.keys()
        if unknown:
            raise ValueError(f"Unknown metric protocols: {sorted(unknown)}")
        if cfg.metrics.mode != "max":
            raise ValueError("The current RoboMimic protocols require mode=max.")
    return cfg


def register_experiment_configs() -> None:
    register_configs()
    ConfigStore.instance().store(name="experiment_schema", node=ExperimentConfig)


def validate_experiment(config: DictConfig | ExperimentConfig) -> ExperimentConfig:
    cfg = OmegaConf.merge(OmegaConf.structured(ExperimentConfig), config)
    OmegaConf.resolve(cfg)
    missing = OmegaConf.missing_keys(cfg)
    if missing:
        raise ValueError(f"Missing experiment configuration: {sorted(missing)}")
    cfg = OmegaConf.to_object(cfg)
    cfg.training = validate_config(cfg.training)
    if not 0 <= cfg.seed < 2**32 - 1:
        raise ValueError("seed must lie in [0, 2**32 - 1).")
    if not isinstance(cfg.recipe, str) or not cfg.recipe:
        raise ValueError("recipe must name an importable training builder.")
    for name in ("dataset", "policy", "objective"):
        if not isinstance(getattr(cfg, name).get("_target_"), str):
            raise ValueError(f"{name} requires a Hydra _target_.")
    if cfg.initialization is not None and not isinstance(
        cfg.initialization.get("_target_"), str
    ):
        raise ValueError("initialization requires a Hydra _target_.")
    if cfg.training.trainer.eval_every and not cfg.evaluation:
        raise ValueError("eval_every requires an evaluation runner configuration.")
    for loader in (cfg.dataloader.train, cfg.dataloader.validation):
        if (
            loader.batch_size < 1
            or loader.num_workers < 0
            or loader.prefetch_factor < 1
        ):
            raise ValueError(
                "Require positive batch_size/prefetch_factor and num_workers >= 0."
            )
    if (
        cfg.training.resume
        and cfg.training.resume_mode == "exact"
        and cfg.dataloader.train.num_workers
    ):
        raise ValueError("Exact resume requires dataloader.train.num_workers=0.")
    if cfg.dataloader.validation.drop_last:
        raise ValueError("Validation must include the final incomplete batch.")
    return cfg
