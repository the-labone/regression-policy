"""Selectable RoboMimic affine normalization, fitted on training episodes only."""

import numpy as np
import torch
from torch import Tensor, nn

from .actions import ActionSpace, check_action_space


class RobomimicNormalizer(nn.Module):
    """DP conventions or per-channel min–max for all observations and actions.

    ``dp`` leaves ABS rotation and gripper unchanged. ``minmax`` follows Much
    Ado's MinMaxNormalizer: fit each channel to [-1, 1], replacing exactly zero
    ranges with one (constant training channels consequently map to -1).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_space: ActionSpace,
        *,
        normalization: str = "dp",
    ):
        super().__init__()
        if normalization not in ("dp", "minmax"):
            raise ValueError("normalization must be 'dp' or 'minmax'.")
        self.normalization = normalization
        self.action_space = check_action_space(action_space)
        if obs_dim < 1 or action_dim not in (
            (7, 14) if action_space == "delta" else (10, 20)
        ):
            raise ValueError(
                "Invalid observation/action dimensions for this action space."
            )
        self.register_buffer("obs_scale", torch.ones(obs_dim))
        self.register_buffer("obs_offset", torch.zeros(obs_dim))
        self.register_buffer("action_scale", torch.ones(action_dim))
        self.register_buffer("action_offset", torch.zeros(action_dim))

    @classmethod
    def fit(
        cls,
        observations: np.ndarray,
        actions: np.ndarray,
        action_space: ActionSpace,
        *,
        normalization: str = "dp",
    ):
        if (
            observations.ndim != 2
            or actions.ndim != 2
            or not len(observations)
            or len(observations) != len(actions)
        ):
            raise ValueError(
                "Normalizer fitting requires nonempty aligned 2D training arrays."
            )
        if not np.isfinite(observations).all() or not np.isfinite(actions).all():
            raise ValueError("Cannot fit a normalizer to non-finite data.")
        result = cls(
            observations.shape[-1],
            actions.shape[-1],
            action_space,
            normalization=normalization,
        )
        if normalization == "minmax":
            result._fit_minmax(observations, actions)
        else:
            result._fit_dp(observations, actions)
        return result

    def _fit_minmax(self, observations: np.ndarray, actions: np.ndarray) -> None:
        """Fit every channel; exactly constant channels map to -1 upstream."""
        for values, scale_buffer, offset_buffer in (
            (observations, self.obs_scale, self.obs_offset),
            (actions, self.action_scale, self.action_offset),
        ):
            values = values.astype(np.float32)
            low, high = values.min(axis=0), values.max(axis=0)
            span = high - low
            span = np.where(span == 0, 1.0, span)
            scale = 2.0 / span
            scale_buffer.copy_(torch.from_numpy(scale))
            offset_buffer.copy_(torch.from_numpy(-1.0 - scale * low))

    def _fit_dp(self, observations: np.ndarray, actions: np.ndarray) -> None:
        """Global observation scaling; ABS positions only, delta actions unchanged."""
        max_abs = float(np.abs(observations).max())
        self.obs_scale.fill_(1.0 / max_abs if max_abs > 1e-7 else 1.0)
        if self.action_space == "delta":
            return
        # Each ABS arm is position(3), rotation_6d(6), gripper(1).
        # Rotation and gripper retain the identity buffers created in __init__.
        for arm_start in range(0, actions.shape[-1], 10):
            position = slice(arm_start, arm_start + 3)
            low = actions[:, position].min(axis=0)
            high = actions[:, position].max(axis=0)
            span = high - low
            constant = span < 1e-7
            scale = 2.0 / np.where(constant, 2.0, span)
            offset = np.where(constant, -low, -1.0 - scale * low)
            self.action_scale[position] = torch.as_tensor(scale)
            self.action_offset[position] = torch.as_tensor(offset)

    def get_extra_state(self):
        return {"action_space": self.action_space, "normalization": self.normalization}

    def set_extra_state(self, state):
        # Older checkpoints omitted the normalization name and always used DP.
        expected = self.get_extra_state()
        if {"normalization": "dp", **state} != expected:
            raise ValueError(
                "Normalizer checkpoint action space or normalization does not match."
            )

    def normalize_obs(self, obs: Tensor) -> Tensor:
        return obs * self.obs_scale + self.obs_offset

    def unnormalize_obs(self, obs: Tensor) -> Tensor:
        return (obs - self.obs_offset) / self.obs_scale

    def normalize_action(self, action: Tensor) -> Tensor:
        return action * self.action_scale + self.action_offset

    def unnormalize_action(self, action: Tensor) -> Tensor:
        return (action - self.action_offset) / self.action_scale
