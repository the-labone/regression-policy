"""RoboCasa GR1 evaluation with batched policy inference and exact episode counts."""

import os
import random
import time
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch

from ...adapters.inference.gr00t import GR00TAdapter
from ..artifacts import write_json
from .environment import create_environment
from .parallel import WORKER_HASH_SEED, EnvironmentBatch
from .protocol import ENV_SUFFIX, environment_id


@contextmanager
def evaluation_state(pipeline, processor, seed):
    """Restore individual module modes and caller RNGs, also after failures."""
    modes = [(module, module.training) for module in pipeline.modules()]
    processor_mode = processor.training
    numpy_state, python_state = np.random.get_state(), random.getstate()
    devices = [pipeline.device.index or 0] if pipeline.device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            for device in devices:
                torch.cuda.default_generators[device].manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            pipeline.eval()
            processor.eval()
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        for module, training in modes:
            module.training = training
        processor.train() if processor_mode else processor.eval()


class GR1DiagnosticRunner:
    """Trainer-compatible ``run(pipeline)`` for a prepared GR00T policy.

    CPU simulators may run in spawned processes; inference and the processor stay
    in the caller. Each requested (task, seed) runs exactly once. No auto-reset,
    dummy episodes, failed-episode filtering or stale actions for completed slots.
    Success means any step reported info['success']; termination/truncation and
    max_steps count individual environment steps, including partial last chunks.
    """

    def __init__(
        self,
        processor,
        output_dir,
        tasks,
        *,
        n_episodes=50,
        start_seed=0,
        seed=0,
        max_steps=720,
        n_envs=1,
        num_workers=0,
        worker_timeout=300.0,
        terminate_on_success=True,
        embodiment_tag="robocasa_gr1_tabletop",
        language_key="annotation.human.coarse_action",
        env_factory=None,
        progress=None,
    ):
        if isinstance(tasks, str):
            tasks = [tasks]
        self.tasks = tuple(environment_id(task) for task in tasks)
        if not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError("tasks must be nonempty and unique.")
        for name, value, minimum in (
            ("n_episodes", n_episodes, 1),
            ("max_steps", max_steps, 1),
            ("n_envs", n_envs, 1),
            ("num_workers", num_workers, 0),
            ("seed", seed, 0),
            ("start_seed", start_seed, 0),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if seed >= 2**32 or start_seed + n_episodes > 2**32:
            raise ValueError("Seeds must fit NumPy's uint32 seed range.")
        if not np.isfinite(worker_timeout) or worker_timeout <= 0:
            raise ValueError("worker_timeout must be finite and positive.")
        self.processor = processor
        self.adapter = GR00TAdapter(
            processor, embodiment_tag=embodiment_tag, language_key=language_key
        )
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.n_episodes, self.start_seed, self.seed = n_episodes, start_seed, seed
        self.max_steps, self.n_envs, self.num_workers = max_steps, n_envs, num_workers
        self.worker_timeout = worker_timeout
        self.terminate_on_success = terminate_on_success
        self.env_factory = env_factory or create_environment
        self.progress = progress

    def _protocol(self, pipeline):
        versions = {"torch": str(torch.__version__)}
        for package in (
            "numpy",
            "gymnasium",
            "robosuite",
            "robocasa",
            "mujoco",
            "transformers",
        ):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                pass
        return dict(
            tasks=self.tasks,
            n_episodes_per_task=self.n_episodes,
            seeds=list(range(self.start_seed, self.start_seed + self.n_episodes)),
            policy_seed=self.seed,
            max_steps=self.max_steps,
            n_envs=self.n_envs,
            num_workers=self.num_workers,
            simulator_python_hash_seed=(
                WORKER_HASH_SEED
                if self.num_workers
                else os.environ.get("PYTHONHASHSEED", "unknown")
            ),
            terminate_on_success=self.terminate_on_success,
            history=self.adapter.history_spec,
            language_key=self.adapter.language_key,
            embodiment_tag=self.adapter.embodiment_tag,
            action_start_index=pipeline.policy.action_start_index,
            n_action_steps=pipeline.policy.n_action_steps,
            task_aggregation="unweighted mean of per-task success rates",
            success="any info['success'] during the episode",
            versions=versions,
            simulator_reference="robocasa-gr1-tabletop-tasks@4840e671596f93ca03651524b9f72ffb1aadfeff",
        )

    def run(self, pipeline):
        self.adapter.validate_policy(pipeline.policy)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if any(
            (self.output_dir / name).exists()
            for name in ("metrics.json", "episodes.json", "error.json")
        ):
            raise FileExistsError("Use a fresh evaluation output directory.")
        write_json(self.output_dir / "protocol.json", self._protocol(pipeline))
        records = []
        timings = dict(
            setup_seconds=0.0,
            inference_seconds=0.0,
            environment_seconds=0.0,
            inference_calls=0,
        )
        started = time.perf_counter()
        try:
            with evaluation_state(pipeline, self.processor, self.seed):
                for task in self.tasks:
                    for offset in range(0, self.n_episodes, self.n_envs):
                        seeds = range(
                            self.start_seed + offset,
                            self.start_seed
                            + min(offset + self.n_envs, self.n_episodes),
                        )
                        self._run_batch(pipeline, task, seeds, records, timings)
        except BaseException as error:
            write_json(self.output_dir / "episodes.json", records)
            write_json(
                self.output_dir / "error.json",
                {"type": type(error).__name__, "message": str(error)},
            )
            raise
        finally:
            timings["total_seconds"] = time.perf_counter() - started
            write_json(self.output_dir / "timings.json", timings)
        records.sort(key=lambda row: (self.tasks.index(row["task"]), row["seed"]))
        write_json(self.output_dir / "episodes.json", records)
        metrics = self._metrics(records)
        write_json(self.output_dir / "metrics.json", metrics)
        return metrics

    def _run_batch(self, pipeline, task, seeds, records, timings):
        specs = [
            dict(
                factory=self.env_factory,
                task=task,
                seed=seed,
                history_spec=self.adapter.history_spec,
                language_key=self.adapter.language_key,
                max_steps=self.max_steps,
                terminate_on_success=self.terminate_on_success,
            )
            for seed in seeds
        ]
        start = time.perf_counter()
        with EnvironmentBatch(
            specs, self.num_workers, self.worker_timeout
        ) as environments:
            timings["setup_seconds"] += time.perf_counter() - start
            observations = environments.observations
            active = list(range(len(specs)))
            pipeline.reset()
            while active:
                start = time.perf_counter()
                batch = {
                    key: np.stack([observations[i][key] for i in active])
                    for key in self.adapter.history_spec
                }
                batch[self.adapter.language_key] = [
                    observations[i][self.adapter.language_key] for i in active
                ]
                chunks = self.adapter.predict(pipeline, batch)
                timings["inference_seconds"] += time.perf_counter() - start
                timings["inference_calls"] += 1
                actions = [None] * len(specs)
                for batch_index, slot_index in enumerate(active):
                    actions[slot_index] = {
                        key: value[batch_index] for key, value in chunks.items()
                    }
                start = time.perf_counter()
                responses = environments.step(actions)
                timings["environment_seconds"] += time.perf_counter() - start
                remaining = []
                for index in active:
                    observations[index], done, record = responses[index]
                    if done:
                        records.append(record)
                        if self.progress is not None:
                            self.progress(record)
                    else:
                        remaining.append(index)
                active = remaining

    def _metrics(self, records):
        if len(records) != len(self.tasks) * self.n_episodes:
            raise RuntimeError(
                "Incomplete evaluation; refusing to report success rates."
            )
        rates, metrics = [], {}
        for task in self.tasks:
            rows = [row for row in records if row["task"] == task]
            if len({row["seed"] for row in rows}) != self.n_episodes:
                raise RuntimeError("Missing or duplicated task seeds.")
            score = float(np.mean([row["success"] for row in rows]))
            rates.append(score)
            name = task.split("/")[1][: -len(ENV_SUFFIX)]
            metrics[f"test/{name}/mean_score"] = score
            metrics[f"test/{name}/num_episodes"] = len(rows)
        metrics.update(
            {
                "test/mean_score": float(np.mean(rates)),
                "test/num_episodes": len(records),
                "test/num_tasks": len(self.tasks),
                "test/mean_length": float(np.mean([row["steps"] for row in records])),
            }
        )
        return metrics
