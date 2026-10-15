"""Training log output: local artifacts, console formatting, and callbacks.

The trainer owns recording cadence, metric computation and checkpoint selection.
This module only writes the records it receives. Construction has no filesystem
or console side effects; the trainer creates the run directory before writing.
"""

import json
import logging
import math
from collections.abc import Callable, Mapping
from numbers import Real
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

LogCallback = Callable[[dict[str, Any]], None]


def scalar_metrics(record: Mapping[str, Any], prefix: str = ""):
    """Flatten numeric metrics, including optimizer-group LRs and summaries."""
    for key, value in record.items():
        tag = f"{prefix}/{key}" if prefix else str(key)
        if not prefix and key == "global_step":
            continue
        if isinstance(value, Mapping):
            yield from scalar_metrics(value, tag)
        elif isinstance(value, (list, tuple)):
            yield from scalar_metrics(dict(enumerate(value)), tag)
        elif isinstance(value, Real) and not isinstance(value, bool):
            if math.isfinite(value):
                yield tag, float(value)


def write_tensorboard_record(writer, record: Mapping[str, Any]) -> None:
    step = record["global_step"]
    if type(step) is not int or step < 0:
        raise ValueError("TensorBoard global_step must be a nonnegative integer.")
    for tag, value in scalar_metrics(record):
        writer.add_scalar(tag, value, global_step=step)


class TrainingLogger:
    """Durable JSONL records, optional asynchronous TensorBoard, and callbacks.

    Each metrics write closes its file before invoking the callback. Exceptions
    propagate to the caller; a failing callback does not discard the local record.
    TensorBoard opens lazily; close() flushes and releases its background writer.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        callback: LogCallback | None = None,
        tensorboard: bool = False,
        flush_secs: int = 30,
        max_queue: int = 100,
    ):
        self.output_dir = Path(output_dir)
        self.callback = callback
        self.tensorboard = tensorboard
        self.flush_secs = flush_secs
        self.max_queue = max_queue
        self._writer = None
        self._writer_class = None
        if tensorboard:
            # Fail before training if enabled without the optional dependencies.
            from torch.utils.tensorboard import SummaryWriter

            self._writer_class = SummaryWriter

    def log_metrics(self, record: Mapping[str, Any]) -> None:
        record = dict(record)
        # Validate serialization before opening the file, avoiding partial rows.
        serialized = json.dumps(record, allow_nan=False)
        with (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(serialized + "\n")
        if self.tensorboard:
            if self._writer is None:
                self._writer = self._writer_class(
                    log_dir=str(self.output_dir / "tensorboard"),
                    flush_secs=self.flush_secs,
                    max_queue=self.max_queue,
                )
            write_tensorboard_record(self._writer, record)
        if self.callback is not None:
            self.callback(record)

    def close(self) -> None:
        """Flush even after a failed or partial fit; a later fit can reopen it."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def save_evaluation(self, report: Mapping[str, Any]) -> None:
        """Replace the evaluation report atomically so readers see a full snapshot."""
        serialized = json.dumps(dict(report), indent=2, allow_nan=False) + "\n"
        path = self.output_dir / "evaluation_metrics.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(path)

    def save_initialization(self, report: Mapping) -> None:
        """Record measured initialization values separately from requested config."""
        serialized = json.dumps(dict(report), indent=2, allow_nan=False) + "\n"
        (self.output_dir / "initialization.json").write_text(
            serialized, encoding="utf-8"
        )

    def save_configs(self, training_config, experiment_config: Mapping | None) -> None:
        OmegaConf.save(
            OmegaConf.structured(training_config), self.output_dir / "config.yaml"
        )
        if experiment_config is not None:
            OmegaConf.save(
                OmegaConf.create(dict(experiment_config)),
                self.output_dir / "experiment.yaml",
            )


def configure_console_logging() -> logging.Logger:
    """Configure the CLI console explicitly, without import-time global changes."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    return logging.getLogger("ht_regression.training")


def log_to_console(record: Mapping[str, Any]) -> None:
    """Default CLI metric callback; full records remain available in metrics.jsonl."""
    logging.getLogger("ht_regression.training").info(
        "step=%d loss=%.6f lr=%s",
        record["global_step"],
        record["train/loss"],
        record["lr"],
    )
