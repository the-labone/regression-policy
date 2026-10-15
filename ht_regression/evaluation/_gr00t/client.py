"""Simulator-environment subprocess: delegate rollout to native GR00T."""

import argparse
from pathlib import Path

from ..artifacts import write_json


def collect(
    *,
    env_name,
    port,
    n_episodes,
    n_envs,
    n_action_steps,
    max_episode_steps,
    output_dir,
    seed=None,
):
    import numpy as np
    from gr00t.eval.rollout_policy import run_gr00t_sim_policy

    returned_name, successes, infos = run_gr00t_sim_policy(
        env_name=env_name,
        n_episodes=n_episodes,
        n_envs=n_envs,
        n_action_steps=n_action_steps,
        max_episode_steps=max_episode_steps,
        policy_client_host="127.0.0.1",
        policy_client_port=port,
        seed=seed,
        video_dir=str(output_dir / "videos"),
    )
    if returned_name != env_name:
        raise ValueError("Native collector returned a different task.")
    if any(not isinstance(value, (bool, np.bool_)) for value in successes):
        raise TypeError("Expected boolean native episode successes.")

    def serializable(value):
        if isinstance(value, np.ndarray):
            return serializable(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: serializable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [serializable(item) for item in value]
        return value

    result = dict(
        env_name=returned_name,
        successes=[bool(v) for v in successes],
        episode_infos=serializable(infos),
    )
    write_json(output_dir / "episodes.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--n-episodes", required=True, type=int)
    parser.add_argument("--n-envs", required=True, type=int)
    parser.add_argument("--n-action-steps", required=True, type=int)
    parser.add_argument("--max-episode-steps", required=True, type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    collect(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
