"""Bridge/WidowX and Fractal/Google benchmark protocols."""

from .._gr00t.runner import NativeGR00TRunner

SUITES = {
    "bridge": dict(
        embodiment="simpler_env_widowx",
        n_episodes=50,
        n_action_steps=4,
        tasks=(
            "widowx_spoon_on_towel",
            "widowx_carrot_on_plate",
            "widowx_stack_cube",
            "widowx_put_eggplant_in_basket",
            "widowx_put_eggplant_in_sink",
            "widowx_open_drawer",
            "widowx_close_drawer",
        ),
    ),
    "fractal": dict(
        embodiment="simpler_env_google",
        n_episodes=100,
        n_action_steps=1,
        tasks=(
            "google_robot_pick_coke_can",
            "google_robot_pick_object",
            "google_robot_move_near",
            "google_robot_open_drawer",
            "google_robot_close_drawer",
            "google_robot_place_in_closed_drawer",
        ),
    ),
}


class GR00TSimplerRunner(NativeGR00TRunner):
    def __init__(
        self,
        output_dir,
        *,
        suite,
        tasks=None,
        n_episodes=None,
        n_action_steps=None,
        max_episode_steps=300,
        **runtime,
    ):
        if suite not in SUITES:
            raise ValueError(f"Unknown Simpler suite: {suite}.")
        defaults = SUITES[suite]
        tasks = list(defaults["tasks"] if tasks is None else tasks)
        if (
            not tasks
            or len(set(tasks)) != len(tasks)
            or any(t not in defaults["tasks"] for t in tasks)
        ):
            raise ValueError("Select unique task names from the chosen suite.")
        super().__init__(
            output_dir,
            suite=suite,
            tasks=tasks,
            embodiment=defaults["embodiment"],
            env_prefix=defaults["embodiment"],
            n_episodes=defaults["n_episodes"] if n_episodes is None else n_episodes,
            n_action_steps=defaults["n_action_steps"]
            if n_action_steps is None
            else n_action_steps,
            max_episode_steps=max_episode_steps,
            **runtime,
        )
