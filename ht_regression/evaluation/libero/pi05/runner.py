"""LIBERO evaluation using the historical LeRobot collector and processors."""

import random
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

from ....adapters.inference.pi05 import PI05Adapter
from ...artifacts import write_json
from ..common import SUITES, validate_rollout


@contextmanager
def evaluation_state(pipeline, native):
    """Scope native seeding/backend settings without perturbing the trainer."""
    import numpy as np

    modes = {
        module: module.training
        for root in (pipeline, native)
        for module in root.modules()
    }
    numpy_state, python_state = np.random.get_state(), random.getstate()
    benchmark = torch.backends.cudnn.benchmark
    tf32 = torch.backends.cuda.matmul.allow_tf32
    # Native set_seed seeds every CUDA device, not just the policy's device.
    devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cuda.matmul.allow_tf32 = tf32
        for module, mode in modes.items():
            module.training = mode


class PI05LiberoRunner:
    """Run a prepared pipeline with native LIBERO state-bank/reset semantics.

    No simulator stepping, episode masks or success aggregation are reimplemented.
    Environment defaults follow the paper's protocol; the policy supplies the
    execution horizon. max_parallel_tasks=1 preserves the native action queue.
    """

    def __init__(
        self,
        native_policy,
        preprocessor,
        postprocessor,
        output_dir,
        *,
        suites=SUITES,
        task_ids=None,
        n_episodes=100,
        n_envs=10,
        seed=0,
        use_async_envs=True,
        provenance=None,
    ):
        self.suites = (suites,) if isinstance(suites, str) else tuple(suites)
        if (
            not self.suites
            or len(set(self.suites)) != len(self.suites)
            or not set(self.suites) <= set(SUITES)
        ):
            raise ValueError(
                "Choose unique LIBERO suites from the four benchmark suites."
            )
        self.native_policy, self.preprocessor, self.postprocessor = (
            native_policy,
            preprocessor,
            postprocessor,
        )
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.task_ids = None if task_ids is None else list(task_ids)
        validate_rollout(
            n_episodes=n_episodes, n_envs=n_envs, seed=seed, task_ids=self.task_ids
        )
        self.n_episodes, self.n_envs, self.seed = n_episodes, n_envs, seed
        self.use_async_envs = use_async_envs
        self.provenance = provenance or {}

    def run(self, pipeline):
        with evaluation_state(pipeline, self.native_policy):
            return self._run(pipeline)

    def _run(self, pipeline):
        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.factory import make_env, make_env_pre_post_processors
        from lerobot.envs.utils import close_envs
        from lerobot.scripts.lerobot_eval import eval_policy_all
        from lerobot.utils.random_utils import set_seed

        self.output_dir.mkdir(parents=True, exist_ok=False)
        protocol = dict(
            suites=self.suites,
            task_ids=self.task_ids,
            n_episodes=self.n_episodes,
            n_envs=self.n_envs,
            seed=self.seed,
            init_states=True,
            hard_reset=True,
            observation_size=360,
            control_mode="relative",
            max_parallel_tasks=1,
            num_steps_wait=10,
            n_action_steps=pipeline.policy.n_action_steps,
            collector="LeRobot eval_policy_all; unchanged native LIBERO environments",
            model=self.provenance,
        )
        write_json(self.output_dir / "protocol.json", protocol)
        results = {}
        try:
            for suite in self.suites:
                # Historical evaluation started a separate seeded process per suite.
                set_seed(self.seed)
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                cfg = LiberoEnv(
                    task=suite,
                    task_ids=self.task_ids,
                    camera_name="agentview_image,robot0_eye_in_hand_image",
                    obs_type="pixels_agent_pos",
                    init_states=True,
                    hard_reset=True,
                    control_mode="relative",
                    observation_height=360,
                    observation_width=360,
                    max_parallel_tasks=1,
                )
                envs = make_env(
                    cfg, n_envs=self.n_envs, use_async_envs=self.use_async_envs
                )
                try:
                    env_pre, env_post = make_env_pre_post_processors(
                        env_cfg=cfg, policy_cfg=self.native_policy.config
                    )
                    amp = (
                        torch.autocast(device_type=pipeline.device.type)
                        if self.native_policy.config.use_amp
                        else nullcontext()
                    )
                    with (
                        PI05Adapter(self.native_policy).bind(pipeline),
                        torch.no_grad(),
                        amp,
                    ):
                        result = eval_policy_all(
                            envs=envs,
                            policy=self.native_policy,
                            env_preprocessor=env_pre,
                            env_postprocessor=env_post,
                            preprocessor=self.preprocessor,
                            postprocessor=self.postprocessor,
                            n_episodes=self.n_episodes,
                            max_episodes_rendered=0,
                            return_episode_data=False,
                            start_seed=self.seed,
                            max_parallel_tasks=1,
                        )
                finally:
                    close_envs(envs)
                write_json(self.output_dir / f"{suite}.json", result)
                results[suite] = result
        except BaseException as error:
            write_json(
                self.output_dir / "error.json",
                dict(
                    type=type(error).__name__,
                    message=str(error),
                    completed_suites=list(results),
                ),
            )
            raise
        # Each task has the same episode budget, so equal suite means reproduce
        # the full four-suite paper mean (ten tasks in each suite).
        scores = {
            suite: result["overall"]["pc_success"] / 100
            for suite, result in results.items()
        }
        metrics = {f"test/{suite}/mean_score": score for suite, score in scores.items()}
        metrics["test/mean_score"] = sum(scores.values()) / len(scores)
        write_json(self.output_dir / "metrics.json", metrics)
        return metrics
