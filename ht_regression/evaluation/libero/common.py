"""Shared LIBERO task and rollout selection validation."""

SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")


def validate_rollout(*, n_episodes, n_envs, seed, task_ids):
    if any(type(n) is not int or n < 1 for n in (n_episodes, n_envs)):
        raise ValueError("n_episodes and n_envs must be positive integers.")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must fit uint32.")
    if task_ids is not None and (
        not task_ids
        or any(type(i) is not int or not 0 <= i < 10 for i in task_ids)
        or len(set(task_ids)) != len(task_ids)
    ):
        raise ValueError("task_ids must contain unique integers in [0, 9].")
