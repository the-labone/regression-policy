"""Explicit controller action semantics and DP's row-based rotation-6D layout."""

from typing import Literal

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ._dp_rotation import (
    matrix_to_quaternion,
    quaternion_to_axis_angle,
    rotation_6d_to_matrix,
)

ActionSpace = Literal["delta", "abs"]


def check_action_space(value: str) -> ActionSpace:
    if value not in ("delta", "abs"):
        raise ValueError(f"action_space must be 'delta' or 'abs', got {value!r}.")
    return value


def encode_actions(actions: np.ndarray, action_space: ActionSpace) -> np.ndarray:
    """Convert native 7D/arm labels to model actions; never convert delta to ABS.

    Delta controller commands stay 7D. Absolute axis-angle poses become 10D:
    position(3), the first two ROWS of the rotation matrix(6), gripper(1).
    """
    check_action_space(action_space)
    actions = np.asarray(actions)
    if actions.ndim < 2 or actions.shape[-1] not in (7, 14):
        raise ValueError("Expected native 7D or 14D OSC_POSE actions.")
    if not np.isfinite(actions).all():
        raise ValueError("Actions contain NaN or infinity.")
    if action_space == "delta":
        return actions.astype(np.float32, copy=True)
    arms = actions.reshape(*actions.shape[:-1], -1, 7)
    matrices = Rotation.from_rotvec(arms[..., 3:6].reshape(-1, 3)).as_matrix()
    rot6d = matrices[:, :2, :].reshape(*arms.shape[:-1], 6)
    encoded = np.concatenate((arms[..., :3], rot6d, arms[..., 6:7]), axis=-1)
    return encoded.reshape(*actions.shape[:-1], -1).astype(np.float32)


def decode_actions(actions: np.ndarray, action_space: ActionSpace) -> np.ndarray:
    """Convert UNNORMALIZED model actions to native controller commands.

    ABS uses the DP release's PyTorch3D v0.7.0 arithmetic and axis-angle branch,
    preserving float32/float64 input precision. Equivalent SciPy rotations can
    round differently enough to change closed-loop rollouts. Degenerate rotations
    fail explicitly; no action clipping is performed here. Use the matching
    controller mode from dataset.env_metadata().
    """
    check_action_space(action_space)
    actions = np.asarray(actions)
    expected = (7, 14) if action_space == "delta" else (10, 20)
    if actions.ndim < 1 or actions.shape[-1] not in expected:
        raise ValueError(f"Expected action dimension in {expected}.")
    if not np.isfinite(actions).all():
        raise ValueError("Actions contain NaN or infinity.")
    if action_space == "delta":
        return actions.astype(np.float64, copy=True)
    if actions.dtype not in (np.dtype("float32"), np.dtype("float64")):
        actions = actions.astype(np.float64)
    arms = actions.reshape(*actions.shape[:-1], -1, 10)
    # Validation is separate from conversion and does not modify model actions.
    rows = arms[..., 3:9].astype(np.float64).reshape(-1, 2, 3)
    first_norm = np.linalg.norm(rows[:, 0], axis=-1, keepdims=True)
    if np.any(first_norm < 1e-8):
        raise ValueError("Degenerate rotation-6D first row.")
    first = rows[:, 0] / first_norm
    second = rows[:, 1] - (first * rows[:, 1]).sum(-1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(second_norm < 1e-8):
        raise ValueError("Degenerate rotation-6D collinear rows.")
    # copy() also supports read-only arrays and views with negative strides.
    rotations = torch.from_numpy(arms[..., 3:9].copy())
    matrices = rotation_6d_to_matrix(rotations)
    rotvec = quaternion_to_axis_angle(matrix_to_quaternion(matrices)).numpy()
    native = np.concatenate((arms[..., :3], rotvec, arms[..., 9:10]), axis=-1)
    return native.reshape(*actions.shape[:-1], -1)
