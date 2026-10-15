"""Diffusion Policy's globally conditioned temporal U-Net.

Adapted from real-stanford/diffusion_policy (MIT), revision
5ba07ac6661db573af695b419a7947ecb704690f, conditional_unet1d.py and
conv1d_components.py. See ../../../LICENSE.diffusion_policy.
"""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from ..common import (
    SinusoidalPosEmb,
    Timestep,
    batch_timesteps,
    validate_dimensions,
    validate_inputs,
)


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class _UnsqueezeLast(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.unsqueeze(-1)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int,
        n_groups: int,
        cond_predict_scale: bool,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups),
            ]
        )
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * (2 if cond_predict_scale else 1)),
            _UnsqueezeLast(),
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            # Keep DP's scale * x + bias (not (1 + scale) * x + bias).
            out = embed[:, 0] * out + embed[:, 1]
        else:
            out = out + embed
        return self.blocks[1](out) + self.residual_conv(x)


class UNetBackbone(nn.Module):
    """Predict an action-shaped tensor using DP's global observation conditioning.

    Inputs: sample (B, horizon, action_dim), timestep scalar/(B,), and
    obs (B, n_obs_steps, obs_dim). Output: (B, horizon, action_dim).
    Observations are flattened directly and concatenated with the time features.
    Inputs must already be normalized; prediction targets and sampling live outside
    this module. This implements the global-conditioning path, not local/inpainting.
    """

    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        horizon: int = 16,
        n_obs_steps: int = 2,
        *,
        diffusion_step_embed_dim: int = 256,
        down_dims: Sequence[int] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ):
        super().__init__()
        validate_dimensions(action_dim, obs_dim, horizon, n_obs_steps)
        down_dims = tuple(down_dims)
        if not down_dims or any(dim < 1 for dim in down_dims):
            raise ValueError("down_dims must contain positive channel dimensions.")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be positive and odd.")
        if n_groups < 1 or any(dim % n_groups for dim in down_dims):
            raise ValueError("Every down_dims entry must be divisible by n_groups.")
        divisor = 2 ** (len(down_dims) - 1)
        if horizon % divisor:
            raise ValueError(
                f"horizon must be divisible by {divisor} for these down_dims; "
                "the DP U-Net does not pad or crop action sequences."
            )
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.feature_dim = down_dims[0]
        self.bottleneck_dim = down_dims[-1]

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + obs_dim * n_obs_steps

        def residual(in_channels: int, out_channels: int) -> nn.Module:
            return ConditionalResidualBlock1D(
                in_channels,
                out_channels,
                cond_dim,
                kernel_size,
                n_groups,
                cond_predict_scale,
            )

        in_out = list(zip((action_dim, *down_dims[:-1]), down_dims))
        self.mid_modules = nn.ModuleList(
            [residual(down_dims[-1], down_dims[-1]) for _ in range(2)]
        )
        self.down_modules = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        residual(dim_in, dim_out),
                        residual(dim_out, dim_out),
                        Downsample1d(dim_out)
                        if index < len(in_out) - 1
                        else nn.Identity(),
                    ]
                )
                for index, (dim_in, dim_out) in enumerate(in_out)
            ]
        )
        # DP consumes the deepest skips and upsamples at every decoder stage.
        self.up_modules = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        residual(dim_out * 2, dim_in),
                        residual(dim_in, dim_in),
                        Upsample1d(dim_in),
                    ]
                )
                for dim_in, dim_out in reversed(in_out[1:])
            ]
        )
        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups),
            nn.Conv1d(down_dims[0], action_dim, 1),
        )

    def forward(self, sample: Tensor, timestep: Timestep, obs: Tensor) -> Tensor:
        return self.forward_head(self.forward_features(sample, timestep, obs))

    def forward_head(self, features: Tensor) -> Tensor:
        return self.final_conv[1](features.transpose(1, 2)).transpose(1, 2)

    def forward_features(
        self, sample: Tensor, timestep: Timestep, obs: Tensor
    ) -> Tensor:
        """Shared B x horizon x feature_dim representation before the action head."""
        features, _ = self.forward_features_with_context(sample, timestep, obs)
        return features

    def forward_features_with_context(
        self, sample: Tensor, timestep: Timestep, obs: Tensor
    ) -> tuple[Tensor, tuple[Tensor]]:
        """Return action features plus the bottleneck before the up modules."""
        validate_inputs(
            sample,
            obs,
            action_dim=self.action_dim,
            obs_dim=self.obs_dim,
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps,
        )
        timesteps = batch_timesteps(timestep, sample)
        time_features = self.diffusion_step_encoder[0](timesteps).to(sample.dtype)
        for index in range(1, len(self.diffusion_step_encoder)):
            time_features = self.diffusion_step_encoder[index](time_features)
        cond = torch.cat((time_features, obs.flatten(start_dim=1)), dim=-1)

        x = sample.transpose(1, 2)
        skips = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet2(resnet(x, cond), cond)
            skips.append(x)
            x = downsample(x)
        for module in self.mid_modules:
            x = module(x, cond)
        bottleneck = x
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, skips.pop()), dim=1)
            x = upsample(resnet2(resnet(x, cond), cond))
        return self.final_conv[0](x).transpose(1, 2), (bottleneck,)

    def get_optimizer_groups(self, weight_decay: float = 1e-6) -> list[dict]:
        """DP's U-Net applies weight decay to all model parameters."""
        return [{"params": list(self.parameters()), "weight_decay": weight_decay}]
