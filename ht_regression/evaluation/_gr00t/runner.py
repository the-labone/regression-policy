"""Run a native GR00T benchmark protocol with separate model and simulator processes."""

import json
import os
import signal
import statistics
import subprocess
import time
from pathlib import Path

from ..artifacts import write_json


def _stop_process(process):
    """Also reap simulator workers if their parent failed or timed out."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


class NativeGR00TRunner:
    """Serve a local checkpoint and delegate every rollout to the native collector.

    Native collection finishes the final vector batch. Keep every returned
    episode (up to n_envs-1 extra) and average task success rates equally.
    Each task starts a fresh model process, as in the original experiment.
    """

    def __init__(
        self,
        output_dir,
        *,
        checkpoint,
        native_root,
        model_python,
        simulator_python,
        suite,
        embodiment,
        env_prefix,
        tasks,
        n_episodes,
        n_envs=5,
        n_action_steps,
        max_episode_steps,
        seed=None,
        device="cuda",
        local_files_only=False,
        startup_timeout=1800,
        task_timeout=14400,
    ):
        self.suite, self.embodiment, self.env_prefix = suite, embodiment, env_prefix
        self.tasks = list(tasks)
        if (
            not self.tasks
            or len(set(self.tasks)) != len(self.tasks)
            or any(
                not isinstance(t, str) or not t or "/" in t or t in (".", "..")
                for t in self.tasks
            )
        ):
            raise ValueError(
                "Select unique, nonempty task names without a path prefix."
            )
        self.n_episodes, self.n_action_steps = n_episodes, n_action_steps
        for count in (self.n_episodes, n_envs, self.n_action_steps, max_episode_steps):
            if type(count) is not int or count < 1:
                raise ValueError(
                    "Rollout counts and step budgets must be positive integers."
                )
        if n_envs > self.n_episodes:
            raise ValueError("n_envs cannot exceed requested episodes.")
        if seed is not None and (
            type(seed) is not int or not 0 <= seed <= 2**32 - n_envs
        ):
            raise ValueError("seed and all vector worker seeds must fit uint32.")
        if not all(
            0 < value < float("inf") for value in (startup_timeout, task_timeout)
        ):
            raise ValueError("Process timeouts must be finite and positive.")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
        self.native_root = Path(native_root).expanduser().resolve(strict=True)
        self.model_python = Path(model_python).expanduser().absolute()
        self.simulator_python = Path(simulator_python).expanduser().absolute()
        for executable in (self.model_python, self.simulator_python):
            if not executable.is_file():
                raise FileNotFoundError(executable)
        if not (self.native_root / "gr00t/eval/rollout_policy.py").is_file():
            raise FileNotFoundError(
                "native_root must contain the GR00T evaluation source."
            )
        self.n_envs, self.max_episode_steps, self.seed = n_envs, max_episode_steps, seed
        self.device, self.local_files_only = device, local_files_only
        self.startup_timeout, self.task_timeout = startup_timeout, task_timeout

    def _environment(self):
        env = os.environ.copy()
        # Prevent an ambient scene override from silently changing the protocol.
        for key in (
            "GR00T_EVAL_SEED",
            "NU_SCENE_MANIFEST_PATH",
            "NU_SCENE_MANIFEST_KEY",
        ):
            env.pop(key, None)
        root = Path(__file__).resolve().parents[3]
        env["PYTHONPATH"] = os.pathsep.join((str(root), str(self.native_root)))
        env.update(PYTHONUNBUFFERED="1", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
        if self.local_files_only:
            env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        return env

    def _server_command(self, directory):
        command = [
            str(self.model_python),
            "-m",
            "ht_regression.evaluation._gr00t.server",
            "--checkpoint",
            str(self.checkpoint),
            "--embodiment",
            self.embodiment,
            "--device",
            self.device,
            "--ready-file",
            str(directory / "server.json"),
        ]
        if self.seed is not None:
            command += ["--seed", str(self.seed)]
        if self.local_files_only:
            command += ["--local-files-only"]
        return command

    def _client_command(self, task, directory, port):
        command = [
            str(self.simulator_python),
            "-m",
            "ht_regression.evaluation._gr00t.client",
            "--env-name",
            f"{self.env_prefix}/{task}",
            "--port",
            str(port),
            "--n-episodes",
            str(self.n_episodes),
            "--n-envs",
            str(self.n_envs),
            "--n-action-steps",
            str(self.n_action_steps),
            "--max-episode-steps",
            str(self.max_episode_steps),
            "--output-dir",
            str(directory),
        ]
        if self.seed is not None:
            command += ["--seed", str(self.seed)]
        return command

    def _wait_ready(self, server, directory):
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(
                    f"GR00T server exited; see {directory / 'server.log'}."
                )
            ready = directory / "server.json"
            if ready.is_file():
                return json.loads(ready.read_text())["port"]
            time.sleep(0.2)
        raise TimeoutError(
            f"GR00T server startup timed out; see {directory / 'server.log'}."
        )

    def _run_task(self, task):
        directory = self.output_dir / task
        directory.mkdir()
        environment = self._environment()
        with (directory / "server.log").open("w") as server_log:
            server = subprocess.Popen(
                self._server_command(directory),
                cwd=self.native_root,
                env=environment,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                port = self._wait_ready(server, directory)
                with (directory / "rollout.log").open("w") as client_log:
                    client = subprocess.Popen(
                        self._client_command(task, directory, port),
                        cwd=self.native_root,
                        env=environment,
                        stdout=client_log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    try:
                        code = self._wait_client(client, server, directory)
                        if code:
                            raise RuntimeError(
                                f"GR00T rollout exited {code}; see {directory / 'rollout.log'}."
                            )
                    finally:
                        _stop_process(client)
            finally:
                _stop_process(server)
        return json.loads((directory / "episodes.json").read_text())

    def _wait_client(self, client, server, directory):
        deadline = time.monotonic() + self.task_timeout
        while True:
            if server.poll() is not None:
                raise RuntimeError(
                    f"GR00T server exited during rollout; see {directory / 'server.log'}."
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"GR00T rollout timed out; see {directory / 'rollout.log'}."
                )
            try:
                return client.wait(timeout=min(remaining, 0.5))
            except subprocess.TimeoutExpired:
                pass

    def summarize(self, results):
        expected = [f"{self.env_prefix}/{t}" for t in self.tasks]
        if len(results) != len(expected) or sorted(
            r["env_name"] for r in results
        ) != sorted(expected):
            raise ValueError("Missing, duplicate or unexpected GR00T task results.")
        tasks = []
        for result in results:
            outcomes = result["successes"]
            if (
                not isinstance(outcomes, list)
                or any(type(x) is not bool for x in outcomes)
                or not self.n_episodes <= len(outcomes) < self.n_episodes + self.n_envs
            ):
                raise ValueError(
                    "Invalid native episode outcomes or final-batch count."
                )
            tasks.append(
                dict(
                    env_name=result["env_name"],
                    requested_episodes=self.n_episodes,
                    episodes=len(outcomes),
                    successes=sum(outcomes),
                    success_rate=sum(outcomes) / len(outcomes),
                )
            )
        return dict(
            suite=self.suite,
            seed=self.seed,
            task_results=tasks,
            mean_success_rate=statistics.mean(t["success_rate"] for t in tasks),
            total_episodes=sum(t["episodes"] for t in tasks),
            total_successes=sum(t["successes"] for t in tasks),
        )

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=False)
        write_json(
            self.output_dir / "protocol.json",
            dict(
                suite=self.suite,
                tasks=self.tasks,
                checkpoint=str(self.checkpoint),
                native_root=str(self.native_root),
                model_python=str(self.model_python),
                simulator_python=str(self.simulator_python),
                seed=self.seed,
                n_episodes=self.n_episodes,
                n_envs=self.n_envs,
                n_action_steps=self.n_action_steps,
                max_episode_steps=self.max_episode_steps,
                aggregation="task_macro_average",
                collector="native_gr00t_with_final_batch_overshoot",
            ),
        )
        try:
            results = []
            for task in self.tasks:
                results.append(self._run_task(task))
            summary = self.summarize(results)
            write_json(self.output_dir / "summary.json", summary)
            write_json(
                self.output_dir / "metrics.json",
                {
                    "test/mean_score": summary["mean_success_rate"],
                    "test/num_episodes": summary["total_episodes"],
                    "test/num_successes": summary["total_successes"],
                    **{
                        f"test/{t['env_name']}/mean_score": t["success_rate"]
                        for t in summary["task_results"]
                    },
                },
            )
            return summary
        except BaseException as error:
            write_json(
                self.output_dir / "failure.json",
                dict(type=type(error).__name__, error=str(error)),
            )
            raise
