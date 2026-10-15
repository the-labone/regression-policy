"""Frozen affine statistics imported with DP model weights."""

import torch
from torch import Tensor, nn


class AffineNormalizer(nn.Module):
    """Frozen per-feature affine transforms, including imported DP statistics.

    Data normalizers with the same four methods can also be passed to the policy.
    No dataset fitting or action representation conversion happens here.
    """

    def __init__(self, obs_scale, obs_offset, action_scale, action_offset):
        super().__init__()
        for field, scale, offset in (
            ("obs", obs_scale, obs_offset),
            ("action", action_scale, action_offset),
        ):
            scale = torch.as_tensor(scale, dtype=torch.float32).detach().clone()
            offset = torch.as_tensor(offset, dtype=torch.float32).detach().clone()
            if (
                scale.ndim != 1
                or not scale.numel()
                or scale.shape != offset.shape
                or not torch.isfinite(scale).all()
                or not torch.isfinite(offset).all()
                or (scale == 0).any()
            ):
                raise ValueError(f"Invalid {field} affine normalization parameters.")
            self.register_buffer(f"{field}_scale", scale)
            self.register_buffer(f"{field}_offset", offset)

    def normalize_obs(self, obs: Tensor) -> Tensor:
        return obs * self.obs_scale + self.obs_offset

    def unnormalize_obs(self, obs: Tensor) -> Tensor:
        return (obs - self.obs_offset) / self.obs_scale

    def normalize_action(self, action: Tensor) -> Tensor:
        return action * self.action_scale + self.action_offset

    def unnormalize_action(self, action: Tensor) -> Tensor:
        return (action - self.action_offset) / self.action_scale
