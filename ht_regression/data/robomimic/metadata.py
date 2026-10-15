"""File provenance is separate from the environment's controller settings."""

import copy
import json

from .actions import ActionSpace, check_action_space

# Keep the original on-disk keys compatible with already prepared datasets.
ACTION_SPACE_ATTR = "praxis_action_space"
REPAIRED_OBS_ATTR = "praxis_tool_hang_relative_poses_recomputed"
SCHEMA_VERSION = 1


def text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_env_metadata(data_group) -> dict:
    raw = data_group.attrs.get("env_args")
    return json.loads(text(raw)) if raw is not None else {}


def resolve_source_action_space(
    data_group, declared: ActionSpace | None
) -> ActionSpace:
    """Do not infer label semantics from filenames or stale control_delta fields."""
    recorded = data_group.attrs.get(ACTION_SPACE_ATTR)
    if recorded is not None:
        recorded = check_action_space(text(recorded))
    if declared is not None:
        declared = check_action_space(declared)
    if recorded is not None and declared is not None and recorded != declared:
        raise ValueError(
            f"Declared {declared!r} actions conflict with file marker {recorded!r}."
        )
    if recorded is None and declared is None:
        raise ValueError(
            "File has no praxis_action_space marker. Specify source_action_space "
            "explicitly after verifying the labels, or prepare a marked dataset. "
            "env_args/control_delta is not reliable evidence of label semantics."
        )
    return recorded if recorded is not None else declared


def pose_controller_configs(env_meta: dict) -> list[dict]:
    configs = env_meta.get("env_kwargs", {}).get("controller_configs")
    if configs is None:
        raise ValueError("Missing controller_configs in env_args.")
    result = []
    for config in configs if isinstance(configs, list) else [configs]:
        parts = config.get("body_parts")
        if parts is None:
            candidates = [config]
        else:
            candidates = [
                value for value in parts.values() if value.get("type") == "OSC_POSE"
            ]
        if not candidates or any(
            value.get("type") != "OSC_POSE" for value in candidates
        ):
            raise ValueError(
                "Only fixed-impedance OSC_POSE arm controllers are supported."
            )
        for value in candidates:
            if value.get("impedance_mode", "fixed") != "fixed":
                raise ValueError("Variable-impedance actions are not supported.")
        result.extend(candidates)
    return result


def controller_env_metadata(env_meta: dict, action_space: ActionSpace) -> dict:
    """Return an independent environment config matching the chosen labels.

    ABS labels prepared here are world-frame end-effector poses. This handles
    both legacy flat controllers and robosuite 1.5's nested body_parts schema.
    """
    check_action_space(action_space)
    result = copy.deepcopy(env_meta)
    for config in pose_controller_configs(result):
        config["control_delta"] = action_space == "delta"
        config["input_type"] = "delta" if action_space == "delta" else "absolute"
        if action_space == "abs":
            config["input_ref_frame"] = "world"
    return result


def validate_abs_reference_frame(env_meta: dict) -> None:
    """Reject base-frame labels before anyone configures a world-frame controller."""
    configs = env_meta.get("env_kwargs", {}).get("controller_configs")
    if configs is None:
        return
    for config in configs if isinstance(configs, list) else [configs]:
        default_frame = "base" if "body_parts" in config else "world"
        for part in pose_controller_configs(
            {"env_kwargs": {"controller_configs": config}}
        ):
            if part.get("input_ref_frame", default_frame) != "world":
                raise ValueError(
                    "ABS loading requires world-frame labels; convert from the original delta data first."
                )
