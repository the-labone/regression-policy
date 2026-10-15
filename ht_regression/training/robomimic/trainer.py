"""Single-device imitation-learning loop over any compatible PolicyPipeline.

The caller constructs/seeds the model and data, and supplies optional evaluation
and logging callbacks. Losses must be scalar means over examples; accumulation
weights unequal microbatches by their example counts. No diffusion math lives here.
"""

import copy
import math
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict
from numbers import Real
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

from ...pipelines import PolicyPipeline
from ..batch import infer_batch_size, move_to_device
from ..checkpoint import (
    capture_rng_state,
    isolated_rng,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from ..logging import TrainingLogger
from ..metrics import METRIC_PROTOCOLS
from .config import TrainingConfig, validate_config


def _class_name(value):
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _generator_states(loader):
    return {
        name: generator.get_state()
        for name, generator in (
            ("loader", loader.generator),
            ("sampler", getattr(loader.sampler, "generator", None)),
        )
        if generator is not None
    }


def _restore_generators(loader, states):
    for name, state in states.items():
        owner = loader if name == "loader" else loader.sampler
        if owner.generator is None:
            raise ValueError("Resume requires the original DataLoader generators.")
        owner.generator.set_state(state.cpu())


def _loader_signature(loader):
    return {
        "dataset": _class_name(loader.dataset),
        "dataset_length": len(loader.dataset),
        "batches": len(loader),
        "batch_size": loader.batch_size,
        "drop_last": loader.drop_last,
        "num_workers": loader.num_workers,
        "sampler": _class_name(loader.sampler),
        "generators": sorted(_generator_states(loader)),
    }


def _replayable(loader):
    sampler = loader.sampler
    return (
        loader.num_workers == 0
        and type(loader.batch_sampler) is BatchSampler
        and (
            type(sampler) is SequentialSampler
            or (
                type(sampler) is RandomSampler
                and not sampler.replacement
                and sampler.num_samples == len(loader.dataset)
            )
        )
    )


class ILTrainer:
    """Shared IL mechanics with explicit configuration and update-based cadence.

    ``evaluation_factory(output_dir)`` returns a fresh runner with ``run(pipeline)``.
    ``log_callback(record)`` receives JSON-serializable metrics. ``metadata`` stores
    component construction/data provenance for reconstruction and resume checks.
    ``batch_size_fn`` can override leading-tensor batch-size inference.
    ``initializer(pipeline)`` optionally calibrates a fresh model after device
    placement and before optimizer/EMA/compilation; returns a checkpointed report.
    It is skipped when resuming a checkpoint.

    Exact resume supports standard sequential/shuffled DataLoaders with zero
    workers and stateless datasets/collators (global Python/NumPy/Torch RNG is OK).
    Stateful datasets, custom generators inside transforms, custom samplers,
    distributed training, and worker-prefetch restoration are not implemented.
    """

    def __init__(
        self,
        pipeline: PolicyPipeline,
        train_loader: DataLoader,
        config: TrainingConfig,
        *,
        validation_loader: DataLoader | None = None,
        optimizer=None,
        lr_scheduler=None,
        evaluation_factory=None,
        log_callback=None,
        metadata: Mapping | None = None,
        experiment_config: Mapping | None = None,
        batch_size_fn=None,
        initializer=None,
    ):
        # Validate configuration and component requirements before allocating state.
        self.config = validate_config(config)
        cfg = self.config
        self._validate_inputs(
            pipeline, train_loader, validation_loader, evaluation_factory
        )
        self.device = self._resolve_device()
        if cfg.allow_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf32
            torch.backends.cudnn.allow_tf32 = cfg.allow_tf32

        # Place parameters before creating anything that holds references to them.
        if optimizer is not None and any(
            p.device != self.device for p in pipeline.parameters()
        ):
            raise ValueError(
                "Move the pipeline to the target device before supplying an optimizer."
            )
        self.pipeline = pipeline.to(self.device)
        # Data-dependent initialization belongs before optimizer/EMA/compilation.
        # Resume restores its saved bias and report; never recalibrate trained weights.
        self.initialization_report = None
        if initializer is not None and not cfg.resume:
            self.initialization_report = initializer(self.pipeline)
            if not isinstance(self.initialization_report, Mapping):
                raise TypeError("initializer must return a report mapping.")
            self.initialization_report = dict(self.initialization_report)

        # Keep data, evaluation, logging and output configuration together.
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.evaluation_factory = evaluation_factory
        self.metadata = dict(metadata or {})
        self.experiment_config = (
            dict(experiment_config) if experiment_config is not None else None
        )
        self.batch_size_fn = batch_size_fn or infer_batch_size
        self.output_dir = Path(cfg.output_dir).expanduser().resolve()
        self.logger = TrainingLogger(
            self.output_dir,
            callback=log_callback,
            tensorboard=cfg.logging.tensorboard,
            flush_secs=cfg.logging.flush_secs,
            max_queue=cfg.logging.max_queue,
        )

        # Optimizer -> LR scheduler; precision and EMA use the prepared pipeline.
        self.optimizer = optimizer if optimizer is not None else self._make_optimizer()
        self._check_optimizer()
        self.lr_scheduler = (
            lr_scheduler if lr_scheduler is not None else self._make_scheduler()
        )
        if self.lr_scheduler.optimizer is not self.optimizer:
            raise ValueError("LR scheduler must belong to the supplied optimizer.")
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=cfg.trainer.precision == "fp16"
        )
        self.ema = (
            copy.deepcopy(self.pipeline).eval().requires_grad_(False)
            if cfg.trainer.ema.enabled
            else None
        )

        # Establish fresh-run state before optionally restoring every component.
        self.progress = {
            "global_step": 0,
            "micro_step": 0,
            "epoch": 0,
            "batch_in_epoch": 0,
            "skipped_updates": 0,
            "examples_seen": 0,
        }
        self._iterator = None
        self._epoch_start = None
        self._at_boundary = True
        self._started = False
        self.evaluation_history = []
        self.best_metric = None
        self.best_step = None
        if cfg.resume:
            self.restore(cfg.resume)
        # Compile a callable, not the module: checkpoint keys and parameter identity
        # stay unchanged, and validation / EMA sampling keep their eager path.
        self._compute_loss = self.pipeline.compute_loss
        if cfg.compile.enabled:
            # Loading extra state can recreate non-module solver tensors on CPU.
            # Reapply placement after restore, before capturing the online loss.
            self.pipeline.objective.to(self.device)
            self._compute_loss = torch.compile(
                self._compute_loss, mode=cfg.compile.mode, fullgraph=True, dynamic=False
            )

    def _validate_inputs(
        self, pipeline, train_loader, validation_loader, evaluation_factory
    ):
        if not isinstance(pipeline, PolicyPipeline):
            raise TypeError("ILTrainer expects a PolicyPipeline.")
        if not isinstance(train_loader, DataLoader) or not len(train_loader):
            raise ValueError("train_loader must be a nonempty DataLoader.")
        if self.config.trainer.validate_every and (
            validation_loader is None or not len(validation_loader)
        ):
            raise ValueError("validate_every requires a nonempty validation_loader.")
        if validation_loader is train_loader:
            raise ValueError("Training and validation require separate DataLoaders.")
        if self.config.trainer.eval_every and evaluation_factory is None:
            raise ValueError("eval_every requires an evaluation_factory.")

    def _resolve_device(self) -> torch.device:
        device = torch.device(self.config.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA was requested but is unavailable.")
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            with torch.cuda.device(device):
                if (
                    self.config.trainer.precision == "bf16"
                    and not torch.cuda.is_bf16_supported()
                ):
                    raise ValueError("The selected CUDA device does not support bf16.")
        return device

    def _make_optimizer(self):
        cfg = self.config.optimizer
        groups = []
        for group in self.pipeline.get_optimizer_groups(cfg.weight_decay):
            group = dict(group)
            group["params"] = [p for p in group["params"] if p.requires_grad]
            if group["params"]:
                groups.append(group)
        covered = {id(p) for group in groups for p in group["params"]}
        objective_params = [
            p
            for p in self.pipeline.objective.parameters()
            if p.requires_grad and id(p) not in covered
        ]
        if objective_params:
            groups.append(
                {"params": objective_params, "weight_decay": cfg.weight_decay}
            )
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.eps)

    def _check_optimizer(self):
        expected = {id(p) for p in self.pipeline.parameters() if p.requires_grad}
        actual = [
            id(p) for group in self.optimizer.param_groups for p in group["params"]
        ]
        if not expected or len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError(
                "Optimizer must cover every trainable pipeline parameter exactly once."
            )

    def _make_scheduler(self):
        warmup = self.config.scheduler.warmup_steps
        total = self.config.scheduler.total_steps or self.config.trainer.max_steps

        def factor(step):
            if step < warmup:
                return step / max(1, warmup)
            if self.config.scheduler.name == "constant":
                return 1.0
            fraction = min(1.0, (step - warmup) / (total - warmup))
            return 0.5 * (1 + math.cos(math.pi * fraction))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, factor)

    def _next_batch(self):
        if self._iterator is None:
            self._epoch_start = {
                "rng": capture_rng_state(),
                "generators": _generator_states(self.train_loader),
            }
            self._iterator = iter(self.train_loader)
        try:
            batch = next(self._iterator)
        except StopIteration:
            self.progress["epoch"] += 1
            self.progress["batch_in_epoch"] = 0
            self._iterator = None
            return self._next_batch()
        self.progress["batch_in_epoch"] += 1
        self.progress["micro_step"] += 1
        size = self.batch_size_fn(batch)
        if not isinstance(size, int) or size < 1:
            raise ValueError(
                "Cannot infer a positive batch size; supply batch_size_fn."
            )
        self.progress["examples_seen"] += size
        return batch, size

    def _autocast(self):
        precision = self.config.trainer.precision
        return torch.autocast(
            self.device.type,
            enabled=precision != "fp32",
            dtype=torch.float16 if precision == "fp16" else torch.bfloat16,
        )

    @torch.no_grad()
    def _update_ema(self):
        if self.ema is None:
            return
        cfg = self.config.trainer.ema
        decay = cfg.decay
        if cfg.schedule == "power":
            # Upstream EMAModel uses its pre-update optimization_step (starts at 0).
            step = max(0, self.progress["global_step"] - cfg.update_after_step - 2)
            decay = (
                0.0
                if step == 0
                else max(
                    cfg.min_decay,
                    min(cfg.decay, 1 - (1 + step / cfg.inv_gamma) ** -cfg.power),
                )
            )
        source_params = dict(self.pipeline.named_parameters())
        for name, target in self.ema.named_parameters():
            source = source_params[name]
            if source.requires_grad:
                target.lerp_(source, 1 - decay)
            else:
                target.copy_(source)
        source_buffers = dict(self.pipeline.named_buffers())
        for name, target in self.ema.named_buffers():
            target.copy_(source_buffers[name])
        source_modules = dict(self.pipeline.named_modules())
        for name, target in self.ema.named_modules():
            source = source_modules[name]
            if type(source).get_extra_state is not nn.Module.get_extra_state:
                target.set_extra_state(copy.deepcopy(source.get_extra_state()))

    def _update(self):
        """Attempt one optimizer update; return metrics or None on AMP overflow."""
        # Accumulate one effective batch, weighting microbatches by example count.
        self._at_boundary = False
        batches = [
            self._next_batch()
            for _ in range(self.config.trainer.gradient_accumulation_steps)
        ]
        self.optimizer.zero_grad(set_to_none=True)
        losses = self._accumulate_gradients(batches)
        grad_norm = self._unscale_and_clip_gradients()

        # Record the LR used for this attempt, before advancing the scheduler.
        learning_rates = [group["lr"] for group in self.optimizer.param_groups]
        scale_before = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        skipped = self.scaler.get_scale() < scale_before
        self.optimizer.zero_grad(set_to_none=True)
        self._at_boundary = True

        # Only successful parameter updates advance the training schedule and EMA.
        if skipped:
            self.progress["skipped_updates"] += 1
            return None
        self.progress["global_step"] += 1
        self.lr_scheduler.step()
        self._update_ema()
        # Materialize metrics after the update, rather than synchronizing each
        # microbatch immediately after backward. Preserve Python weighted averaging.
        loss_values = torch.stack([loss for loss, _ in losses]).tolist()
        mean_loss = sum(
            value * weight for value, (_, weight) in zip(loss_values, losses)
        )
        return {
            "train/loss": mean_loss,
            "train/grad_norm": grad_norm.item(),
            "lr": learning_rates,
        }

    def _accumulate_gradients(self, batches):
        total_examples = sum(size for _, size in batches)
        losses = []
        for batch, size in batches:
            with self._autocast():
                loss = self._compute_loss(move_to_device(batch, self.device))
            # Intentional eager boundary: reject a bad loss before backward/update.
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError("Objective must return a finite scalar mean loss.")
            weight = size / total_examples
            self.scaler.scale(loss * weight).backward()
            # Clone because CUDA graph replay can reuse the compiled output buffer.
            losses.append((loss.detach().clone(), weight))
        return losses

    def _unscale_and_clip_gradients(self):
        # Clipping and reported norms must use the original, unscaled gradients.
        self.scaler.unscale_(self.optimizer)
        params = [p for p in self.pipeline.parameters() if p.requires_grad]
        max_norm = self.config.trainer.max_grad_norm
        return nn.utils.clip_grad_norm_(
            params,
            max_norm if max_norm is not None else float("inf"),
            error_if_nonfinite=not self.scaler.is_enabled(),
        )

    @torch.no_grad()
    def validate(self):
        if self.validation_loader is None or not len(self.validation_loader):
            raise ValueError("Validation requires a nonempty validation_loader.")
        loader = self.validation_loader
        generators = _generator_states(loader)
        try:
            with self._evaluation_pipeline() as pipeline:
                total, count = 0.0, 0
                for batch in loader:
                    size = self.batch_size_fn(batch)
                    if not isinstance(size, int) or size < 1:
                        raise ValueError("Invalid validation batch size.")
                    with self._autocast():
                        loss = pipeline.compute_loss(move_to_device(batch, self.device))
                    if loss.ndim != 0 or not torch.isfinite(loss):
                        raise ValueError("Validation loss must be a finite scalar.")
                    total += loss.item() * size
                    count += size
                return {"validation/loss": total / count}
        finally:
            _restore_generators(loader, generators)

    @torch.no_grad()
    def evaluate(self):
        if self.evaluation_factory is None:
            raise ValueError("Evaluation requires an evaluation_factory.")
        with self._evaluation_pipeline() as pipeline:
            runner = self.evaluation_factory(
                self.output_dir
                / "evaluation"
                / f"step_{self.progress['global_step']:08d}"
            )
            return runner.run(pipeline)

    @contextmanager
    def _evaluation_pipeline(self):
        """Select EMA and isolate evaluation modes/RNG, including on failure."""
        pipeline = self.ema if self.ema is not None else self.pipeline
        modes = [(module, module.training) for module in pipeline.modules()]
        try:
            pipeline.eval()
            with isolated_rng(self.config.evaluation_seed):
                yield pipeline
        finally:
            for module, training in modes:
                module.training = training

    def _record_evaluation(self, result):
        """Commit one successful evaluation, then select/save a new best model."""
        row = {
            key: float(value)
            for key, value in result.items()
            if isinstance(value, Real) and not isinstance(value, bool)
        }
        if not all(math.isfinite(value) for value in row.values()):
            raise ValueError("Evaluation returned a non-finite scalar metric.")
        row["global_step"] = self.progress["global_step"]
        cfg = self.config.metrics
        if cfg.monitor_key is not None and cfg.monitor_key not in row:
            raise ValueError(
                f"Evaluation did not return monitored metric {cfg.monitor_key!r}."
            )
        history = [*self.evaluation_history, row]
        summary = {}
        for protocol in cfg.protocols:
            values = METRIC_PROTOCOLS[protocol](history, key=cfg.monitor_key)
            summary.update(
                {f"{protocol}/{key}": value for key, value in values.items()}
            )
        improved = False
        if cfg.monitor_key is not None:
            score = row[cfg.monitor_key]
            improved = self.best_metric is None or (
                score > self.best_metric
                if cfg.mode == "max"
                else score < self.best_metric
            )
            if improved:
                self.best_metric, self.best_step = score, row["global_step"]
        self.evaluation_history = history
        report = {
            "monitor_key": cfg.monitor_key,
            "mode": cfg.mode,
            "best_metric": self.best_metric,
            "best_step": self.best_step,
            "summary": summary,
            "evaluations": history,
        }
        self.logger.save_evaluation(report)
        if improved:
            self.save(self.output_dir / "checkpoints" / "best.pt")
        return {
            **{
                f"evaluation/{key}": value
                for key, value in row.items()
                if key != "global_step"
            },
            **{f"summary/{key}": value for key, value in summary.items()},
        }

    def fit(self, *, until_step: int | None = None):
        """Run updates and release log writers on success, failure or interruption."""
        try:
            return self._fit(until_step=until_step)
        finally:
            self.logger.close()

    def _start_run(self):
        """Create run artifacts once, including when continuing a partial fit."""
        if self._started:
            return
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError("Use a new output_dir, including for resumed runs.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.save_configs(self.config, self.experiment_config)
        if self.initialization_report is not None:
            self.logger.save_initialization(self.initialization_report)
        self._started = True

    def _fit(self, *, until_step: int | None = None):
        """Train to an absolute update count; partial runs retain the full LR budget."""
        end = self.config.trainer.max_steps if until_step is None else until_step
        if (
            type(end) is not int
            or not self.progress["global_step"] <= end <= self.config.trainer.max_steps
        ):
            raise ValueError(
                "until_step must lie between the current and configured max_steps."
            )
        self._start_run()
        self.pipeline.train()
        cfg = self.config.trainer
        consecutive_skips = 0
        while self.progress["global_step"] < end:
            metrics = self._update()
            if metrics is None:
                consecutive_skips += 1
                if consecutive_skips >= 100:
                    raise FloatingPointError(
                        "AMP skipped 100 consecutive updates; inspect gradients."
                    )
                continue
            consecutive_skips = 0
            step = self.progress["global_step"]
            if cfg.validate_every and step % cfg.validate_every == 0:
                metrics.update(self.validate())
            if cfg.eval_every and step % cfg.eval_every == 0:
                metrics.update(self._record_evaluation(self.evaluate()))
            if step % cfg.log_every == 0 or step == end or len(metrics) > 3:
                self.logger.log_metrics({**self.progress, **metrics})
            if cfg.checkpoint_every and step % cfg.checkpoint_every == 0:
                self.save(self.output_dir / "checkpoints" / f"step_{step:08d}.pt")
        self.save(self.output_dir / "checkpoints" / "last.pt")
        return dict(self.progress)

    def _components(self):
        return {
            "policy": _class_name(self.pipeline.policy),
            "objective": _class_name(self.pipeline.objective),
            "optimizer": _class_name(self.optimizer),
            "lr_scheduler": _class_name(self.lr_scheduler),
        }

    def save(self, path):
        if not self._at_boundary:
            raise RuntimeError("Save only at optimizer-update boundaries.")
        return save_checkpoint(
            path,
            {
                "pipeline": self.pipeline.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict(),
                "ema": self.ema.state_dict() if self.ema is not None else None,
                "progress": dict(self.progress),
                "config": asdict(self.config),
                "components": self._components(),
                "metadata": self.metadata,
                "initialization": self.initialization_report,
                "evaluation": {
                    "history": self.evaluation_history,
                    "best_metric": self.best_metric,
                    "best_step": self.best_step,
                },
                "rng": capture_rng_state(),
                "data": {
                    "signature": _loader_signature(self.train_loader),
                    "replayable": _replayable(self.train_loader),
                    "epoch_start": self._epoch_start,
                    "generators": _generator_states(self.train_loader),
                },
            },
        )

    def restore(self, path):
        if self._started or self.progress["micro_step"]:
            raise RuntimeError("Restore into a fresh trainer before fit().")
        state = load_checkpoint(path)
        saved, current = asdict(validate_config(state["config"])), asdict(self.config)
        for config in (saved, current):
            config.pop("output_dir")
            config.pop("resume")
            config.pop("resume_mode")
            # Logging destinations/cadence of flushing do not affect optimization.
            config.pop("logging")
        if saved != current or state["components"] != self._components():
            raise ValueError(
                "Resume requires the same training configuration and component types."
            )
        data = state["data"]
        epoch_resume = self.config.resume_mode == "epoch_boundary"
        if epoch_resume:
            if state["progress"]["batch_in_epoch"] != len(self.train_loader):
                raise ValueError("epoch_boundary resume requires a completed epoch.")
        elif not data["replayable"] or not _replayable(self.train_loader):
            raise ValueError(
                "Exact resume requires num_workers=0 and a standard sequential/random sampler."
            )
        if (
            data["signature"] != _loader_signature(self.train_loader)
            or state["metadata"] != self.metadata
        ):
            raise ValueError("Resume data configuration/provenance does not match.")
        self.pipeline.load_state_dict(state["pipeline"], strict=True)
        self.initialization_report = copy.deepcopy(state.get("initialization"))
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        if self.ema is not None:
            self.ema.load_state_dict(state["ema"], strict=True)
        self.progress = dict(state["progress"])
        evaluation = state.get("evaluation")
        if evaluation is None and self.config.metrics.monitor_key is not None:
            raise ValueError("Checkpoint has no evaluation history to restore.")
        if evaluation is not None:
            self.evaluation_history = copy.deepcopy(evaluation["history"])
            self.best_metric = evaluation["best_metric"]
            self.best_step = evaluation["best_step"]
        self._epoch_start = data["epoch_start"]
        try:
            if epoch_resume:
                # Worker prefetch/RNG is not checkpointed. Start the next epoch
                # explicitly, retaining the saved parent RNG and loader generators.
                self.progress["epoch"] += 1
                self.progress["batch_in_epoch"] = 0
                self._epoch_start = None
            elif self._epoch_start is not None:
                restore_rng_state(self._epoch_start["rng"])
                _restore_generators(self.train_loader, self._epoch_start["generators"])
                self._iterator = iter(self.train_loader)
                for _ in range(self.progress["batch_in_epoch"]):
                    next(self._iterator)
            _restore_generators(self.train_loader, data["generators"])
        finally:
            restore_rng_state(state["rng"])
