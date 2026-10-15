"""Native GR1 environment names and checkpoint task selection."""

import re
from pathlib import Path

ENV_SUFFIX = "_GR1ArmsAndWaistFourierHands_Env"


def environment_id(task):
    if task.startswith("gr1_unified/"):
        task = task[len("gr1_unified/") :]
        if not task.endswith(ENV_SUFFIX):
            raise ValueError("Expected a GR1 arms-and-waist Fourier-hands environment.")
        task = task[: -len(ENV_SUFFIX)]
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", task):
        raise ValueError(f"Invalid GR1 task name: {task!r}.")
    return f"gr1_unified/{task}{ENV_SUFFIX}"


def checkpoint_tasks(checkpoint):
    """Keep the 24-task dataset selection recorded by the historical training run."""
    import yaml

    path = Path(checkpoint).expanduser() / "experiment_cfg/config.yaml"
    config = yaml.safe_load(path.read_text())
    paths = config["data"]["datasets"][0]["dataset_paths"]
    tasks = [Path(p).name.removeprefix("gr1_unified.") for p in paths]
    if len(tasks) != 24 or len(set(tasks)) != 24:
        raise ValueError(
            "Expected 24 distinct GR1 dataset tasks; otherwise supply --task explicitly."
        )
    return tasks
