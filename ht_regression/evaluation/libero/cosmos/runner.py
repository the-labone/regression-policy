"""LIBERO-10 evaluation through Cosmos's original simulation client."""

import json
import math
import subprocess
from pathlib import Path

from ....adapters.checkpoint.common import file_hash
from ...artifacts import write_json
from ..common import validate_rollout
from .server import policy_server


class CosmosLiberoRunner:
    """Preserve Cosmos camera, rotation, gripper and initial-state conventions.

    The native simulator runs in its separate Python environment. The loaded
    GPU model serves requests locally; no checkpoint or model is copied to the
    simulator process. By default, evaluate ten tasks with ten episodes each.
    """

    def __init__(
        self,
        output_dir,
        *,
        native_client,
        simulator_python,
        task_ids=None,
        n_episodes=10,
        n_envs=8,
        seed=0,
    ):
        self.task_ids = list(range(10)) if task_ids is None else list(task_ids)
        validate_rollout(
            n_episodes=n_episodes, n_envs=n_envs, seed=seed, task_ids=self.task_ids
        )
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.native_client = Path(native_client).expanduser().resolve(strict=True)
        self.simulator_python = Path(simulator_python).expanduser().absolute()
        if not self.native_client.is_file() or not self.simulator_python.is_file():
            raise ValueError("Supply the native client and its Python executable.")
        self.n_episodes, self.n_envs, self.seed = n_episodes, n_envs, seed

    def client_command(self, server_url, *, request_timeout=300):
        return [
            str(self.simulator_python),
            str(self.native_client),
            "--server_url",
            server_url,
            "--task_suite",
            "libero_10",
            "--task_ids",
            ",".join(map(str, self.task_ids)),
            "--num_trials_per_task",
            str(self.n_episodes),
            "--num_envs",
            str(self.n_envs),
            "--seed",
            str(self.seed),
            "--camera",
            "agentview,wrist",
            "--image_size",
            "256",
            "--env_image_size",
            "256",
            "--action_space",
            "frame_wise_relative",
            "--rotation_space",
            "6d",
            "--action_dim",
            "10",
            "--action_horizon",
            "0",
            "--gripper_mode",
            "zero_one",
            "--initial_states_path",
            "DEFAULT",
            "--warmup_steps",
            "10",
            "--max_steps",
            "520",
            "--mujoco_gl",
            "egl",
            "--timeout",
            str(request_timeout),
            "--output_dir",
            str(self.output_dir / "results"),
        ]

    def summarize(self, summary):
        tasks = summary["task_results"]
        if (
            summary["task_suite"] != "libero_10"
            or sorted(summary["selected_task_ids"]) != sorted(self.task_ids)
            or len(tasks) != len(self.task_ids)
            or summary["total_episodes"] != len(self.task_ids) * self.n_episodes
        ):
            raise ValueError(
                "Native evaluation did not complete the requested task/episode set."
            )
        if sorted(task["task_id"] for task in tasks) != sorted(self.task_ids):
            raise ValueError(
                "Native evaluation returned missing or duplicate task IDs."
            )
        for task in tasks:
            if (
                task["episodes"] != self.n_episodes
                or not 0 <= task["successes"] <= task["episodes"]
                or not math.isclose(
                    task["success_rate"], task["successes"] / task["episodes"]
                )
            ):
                raise ValueError(
                    "Inconsistent native task episode counts or success rate."
                )
        if summary["total_successes"] != sum(
            task["successes"] for task in tasks
        ) or not math.isclose(
            summary["overall_success_rate"],
            summary["total_successes"] / summary["total_episodes"],
        ):
            raise ValueError("Inconsistent native aggregate success rate.")
        metrics = {
            f"test/libero_10/task_{task['task_id']}/mean_score": task["success_rate"]
            for task in tasks
        }
        metrics.update(
            {
                "test/libero_10/mean_score": summary["overall_success_rate"],
                "test/mean_score": summary["overall_success_rate"],
                "test/num_episodes": summary["total_episodes"],
                "test/num_successes": summary["total_successes"],
            }
        )
        return metrics

    def run(self, bundle):
        if bundle.provenance["inference_seed"] != self.seed:
            raise ValueError(
                "Environment and inference seeds must match the historical protocol."
            )
        self.output_dir.mkdir(parents=True, exist_ok=False)
        try:
            with policy_server(bundle) as (url, service):
                command = self.client_command(
                    url, request_timeout=900 if bundle.method == "flow" else 300
                )
                write_json(
                    self.output_dir / "protocol.json",
                    dict(
                        checkpoint=bundle.provenance,
                        client={
                            "path": str(self.native_client),
                            "sha256": file_hash(self.native_client),
                        },
                        command=command,
                        task_ids=self.task_ids,
                        n_episodes=self.n_episodes,
                        n_envs=self.n_envs,
                        seed=self.seed,
                        rotate_180=True,
                        collector="unchanged Cosmos closed_loop_eval.py",
                    ),
                )
                with (self.output_dir / "eval.log").open("w") as log:
                    subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, check=True
                    )
            if service.errors:
                raise RuntimeError(f"Cosmos inference failures: {service.errors}")
            summary = json.loads((self.output_dir / "results/summary.json").read_text())
            metrics = self.summarize(summary)
            write_json(self.output_dir / "metrics.json", metrics)
            return metrics
        except BaseException as error:
            write_json(
                self.output_dir / "error.json",
                dict(type=type(error).__name__, message=str(error)),
            )
            raise
