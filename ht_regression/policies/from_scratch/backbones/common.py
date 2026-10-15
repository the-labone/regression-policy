"""Shared tensor conventions and Diffusion Policy's sinusoidal embedding.

Embedding adapted from real-stanford/diffusion_policy (MIT), revision
5ba07ac6661db573af695b419a7947ecb704690f. See ../../LICENSE.diffusion_policy.
"""

import math

import torch
from torch import Tensor, nn

Timestep = Tensor | float | int


class SinusoidalPosEmb(nn.Module):
    """DP's single-time embedding; timesteps are used without rescaling."""

    def __init__(self, dim: int):
        super().__init__()
        if dim < 4 or dim % 2:
            raise ValueError("Time embedding dimension must be even and at least 4.")
        self.dim = dim

    def forward(self, timestep: Tensor) -> Tensor:
        half_dim = self.dim // 2
        exponent = math.log(10000) / (half_dim - 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=timestep.device) * -exponent
        )
        angles = timestep[:, None] * frequencies[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


def validate_dimensions(
    action_dim: int, obs_dim: int, horizon: int, n_obs_steps: int
) -> None:
    for name, value in (
        ("action_dim", action_dim),
        ("obs_dim", obs_dim),
        ("horizon", horizon),
        ("n_obs_steps", n_obs_steps),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive, got {value}.")
    if n_obs_steps > horizon:
        raise ValueError("n_obs_steps must not exceed horizon.")


def validate_inputs(
    sample: Tensor,
    obs: Tensor,
    *,
    action_dim: int,
    obs_dim: int,
    horizon: int,
    n_obs_steps: int,
) -> None:
    if sample.ndim != 3 or sample.shape[1:] != (horizon, action_dim):
        raise ValueError(
            f"sample must have shape (B, {horizon}, {action_dim}), "
            f"got {tuple(sample.shape)}."
        )
    if obs.shape != (sample.shape[0], n_obs_steps, obs_dim):
        raise ValueError(
            f"obs must have shape ({sample.shape[0]}, {n_obs_steps}, {obs_dim}), "
            f"got {tuple(obs.shape)}."
        )
    if not sample.is_floating_point() or not obs.is_floating_point():
        raise TypeError("sample and obs must be floating-point tensors.")
    if sample.device != obs.device or sample.dtype != obs.dtype:
        raise ValueError("sample and obs must have the same device and dtype.")


def batch_timesteps(timestep: Timestep, sample: Tensor) -> Tensor:
    """Broadcast a scalar or length-one vector, preserving fractional times.

    The objective chooses the time scale: DDPM can pass integer steps, and flow
    or regression can pass continuous times. Python floats are never cast to int.
    """
    timestep = torch.as_tensor(timestep, device=sample.device)
    if timestep.ndim == 0:
        timestep = timestep[None]
    if timestep.ndim != 1 or timestep.shape[0] not in (1, sample.shape[0]):
        raise ValueError("timestep must be a scalar, (1,), or (B,) tensor.")
    return timestep.expand(sample.shape[0])
