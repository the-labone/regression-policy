"""Validate known Tool Hang observation-cache errors before fitting statistics."""

import numpy as np
from scipy.spatial.transform import Rotation


def encode_observation(
    observation: dict[str, np.ndarray],
    obs_keys: tuple[str, ...],
    obs_dim: int,
    env_meta: dict,
    *,
    repair_observations: bool = False,
) -> np.ndarray:
    """Pack raw frames in training order, with optional matching geometry repair.

    Accepts a single frame or arbitrary leading batch/time dimensions. This is
    shared by datasets and runners; it never fits or applies normalization.
    """
    leading_shape = np.asarray(observation[obs_keys[0]]).shape[:-1]
    keys = set(obs_keys)
    layout = tool_hang_layout(env_meta)
    if repair_observations and layout is not None:
        keys.update(("object", "robot0_eef_pos", "robot0_eef_quat"))
    flat = {}
    for key in keys:
        value = np.asarray(observation[key])
        if (
            value.ndim < 1
            or value.shape[:-1] != leading_shape
            or not np.isfinite(value).all()
        ):
            raise ValueError(
                f"Observation {key!r} has inconsistent shape or non-finite values."
            )
        flat[key] = value.reshape(-1, value.shape[-1])
    if repair_observations and layout is not None:
        flat["object"] = repair_tool_hang_relative_poses(flat, layout)
    result = np.concatenate([flat[key] for key in obs_keys], axis=-1)
    if result.shape[-1] != obs_dim:
        raise ValueError("Evaluation observation dimension differs from training.")
    return result.reshape(*leading_shape, obs_dim).astype(np.float32)


def tool_hang_layout(env_meta: dict) -> str | None:
    if env_meta.get("env_name", "").lower().replace("_", "") != "toolhang":
        return None
    controllers = env_meta.get("env_kwargs", {}).get("controller_configs", {})
    # The 1.5 schema puts relative pose before global pose for each object.
    return "relative_first" if "body_parts" in controllers else "global_first"


def repair_tool_hang_relative_poses(
    obs: dict[str, np.ndarray], layout: str
) -> np.ndarray:
    """Recompute relative poses from recorded GLOBAL object and EEF poses.

    This preserves field order and all non-pose fields. It repairs observations,
    not action labels. Quaternions follow robomimic's xyzw convention.
    """
    objects = np.asarray(obs["object"], dtype=np.float64)
    if objects.ndim != 2 or objects.shape[-1] != 44:
        raise ValueError(
            "Tool Hang validation supports the known single-arm 44D object layout."
        )
    if layout not in ("global_first", "relative_first"):
        raise ValueError(f"Unknown Tool Hang object layout: {layout}")
    eef_position = np.asarray(obs["robot0_eef_pos"])
    eef_rotation = Rotation.from_quat(obs["robot0_eef_quat"])
    result = objects.copy()
    global_offset, relative_offset = (0, 7) if layout == "global_first" else (7, 0)
    for start in (0, 14, 28):
        world = start + global_offset
        relative = start + relative_offset
        position = objects[:, world : world + 3]
        rotation = Rotation.from_quat(objects[:, world + 3 : world + 7])
        result[:, relative : relative + 3] = eef_rotation.inv().apply(
            position - eef_position
        )
        quat = (eef_rotation.inv() * rotation).as_quat()
        # Keep the original quaternion hemisphere wherever it is meaningful.
        sign = np.sum(quat * objects[:, relative + 3 : relative + 7], axis=-1) < 0
        quat[sign] *= -1
        result[:, relative + 3 : relative + 7] = quat
    return result.astype(obs["object"].dtype)


def validate_tool_hang_first_frame(
    obs: dict[str, np.ndarray], env_meta: dict, demo_name: str
) -> None:
    layout = tool_hang_layout(env_meta)
    if layout is None or "object" not in obs:
        return
    first = {key: value[:1] for key, value in obs.items()}
    expected = repair_tool_hang_relative_poses(first, layout)
    offset = 7 if layout == "global_first" else 0
    for start in (0, 14, 28):
        relative = start + offset
        pos = first["object"][:, relative : relative + 3]
        expected_pos = expected[:, relative : relative + 3]
        quat = first["object"][:, relative + 3 : relative + 7]
        expected_quat = expected[:, relative + 3 : relative + 7]
        pos_error = float(np.linalg.norm(pos - expected_pos))
        rot_error = float(
            (
                Rotation.from_quat(quat).inv() * Rotation.from_quat(expected_quat)
            ).magnitude()[0]
        )
        if pos_error > 1e-3 or rot_error > 1e-3:
            raise ValueError(
                f"{demo_name}: inconsistent Tool Hang first-frame object pose "
                f"(position error {pos_error:.4g} m, rotation error {rot_error:.4g} rad). "
                "Use verified original data or explicitly repair observations with "
                "the prepare command. No frames were silently dropped."
            )


def encode_rgb(value, shape):
    value = np.asarray(value)
    if value.dtype == np.uint8 and value.shape == (shape[1], shape[2], 3):
        value = value.transpose(2, 0, 1)
    elif np.issubdtype(value.dtype, np.floating) and value.shape == tuple(shape):
        if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
            raise ValueError("Processed simulator RGB must be finite in [0, 1].")
        value = np.rint(value * 255).astype(np.uint8)
    if value.shape != tuple(shape) or value.dtype != np.uint8:
        raise ValueError("Simulator camera shape/dtype does not match the dataset.")
    # RoboMimic already corrects renderer orientation; do not flip again.
    return value.copy()
