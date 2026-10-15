"""GR00T's native RoboCasa-GR1 benchmark protocol."""

from .._gr00t.runner import NativeGR00TRunner
from .protocol import checkpoint_tasks, environment_id


class RoboCasaGR1Runner(NativeGR00TRunner):
    """Native autoreset, full action chunks, actual counts and task-macro mean."""

    def __init__(
        self,
        output_dir,
        *,
        checkpoint,
        tasks=None,
        n_episodes=20,
        n_action_steps=8,
        max_episode_steps=720,
        **runtime,
    ):
        tasks = checkpoint_tasks(checkpoint) if tasks is None else tasks
        if isinstance(tasks, str):
            tasks = [tasks]
        names = [environment_id(task).split("/", 1)[1] for task in tasks]
        super().__init__(
            output_dir,
            checkpoint=checkpoint,
            tasks=names,
            suite="robocasa_gr1",
            embodiment="robocasa_gr1_tabletop",
            env_prefix="gr1_unified",
            n_episodes=n_episodes,
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            **runtime,
        )
