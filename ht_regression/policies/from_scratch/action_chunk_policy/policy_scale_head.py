"""Chunk-level scalar heads adapted from much-ado-about-noising's Chi networks.

Reference: mip/networks/{chitfm,chiunet}.py at revision
41085ae108a7a5aebaf145b4281885cd806eb585 (MIT). See ../../LICENSE.much_ado.
"""

import torch
from torch import Tensor, nn

from ...base import Timestep
from ..backbones import TransformerBackbone, UNetBackbone
from .policy import ActionChunkPolicy


class TransformerScaleHead(nn.Module):
    """Pool input, pre-LayerNorm decoder output and encoded condition memory."""

    def __init__(self, action_dim: int, feature_dim: int):
        super().__init__()
        if feature_dim < 4:
            raise ValueError("Transformer scale head requires feature_dim >= 4.")
        self.input_processor = nn.Linear(action_dim, feature_dim // 4)
        self.final_processor = nn.Linear(feature_dim, feature_dim // 4)
        self.output = nn.Linear(2 * (feature_dim // 4) + feature_dim, 1)
        for projection in (self.input_processor, self.final_processor):
            nn.init.normal_(projection.weight, std=0.02)
            nn.init.zeros_(projection.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, sample: Tensor, decoder_output: Tensor, memory: Tensor) -> Tensor:
        features = torch.cat(
            (
                self.input_processor(sample.mean(dim=1)),
                self.final_processor(decoder_output.mean(dim=1)),
                memory.mean(dim=1),
            ),
            dim=-1,
        )
        return self.output(features)


class UNetScaleHead(nn.Module):
    """Read a chunk scalar from the globally pooled U-Net bottleneck."""

    def __init__(self, bottleneck_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(bottleneck_dim, 128),
            nn.SiLU(),
        )
        self.output = nn.Linear(128, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, sample: Tensor, bottleneck: Tensor) -> Tensor:
        return self.output(self.features(bottleneck))


class ActionChunkPolicyWithScale(ActionChunkPolicy):
    """Share one backbone pass between the mean and one raw scalar per chunk.

    Features are pooled before the scalar projection. The objective owns the
    positive scale transform and likelihood; no feature branch is detached.
    Inherited forward/action decoding use only the mean for deployment.
    """

    def __init__(
        self,
        model: nn.Module,
        normalizer: nn.Module,
        *,
        n_action_steps: int = 8,
        action_start_index: int | None = None,
        validate_values: bool = True,
        obs_encoder: nn.Module | None = None,
    ):
        if not isinstance(model, (TransformerBackbone, UNetBackbone)):
            raise TypeError(
                "ActionChunkPolicyWithScale requires a TransformerBackbone or UNetBackbone."
            )
        super().__init__(
            model,
            normalizer,
            n_action_steps=n_action_steps,
            action_start_index=action_start_index,
            validate_values=validate_values,
            obs_encoder=obs_encoder,
        )
        if isinstance(model, TransformerBackbone):
            self.scale_head = TransformerScaleHead(model.action_dim, model.feature_dim)
        else:
            self.scale_head = UNetScaleHead(model.bottleneck_dim)
        self.scale_head.to(device=self.device, dtype=self.dtype)

    def forward_with_scale(
        self, sample: Tensor, time: Timestep, condition: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return the mean and (B, 1) raw chunk scale in one backbone pass."""
        features, context = self.model.forward_features_with_context(
            sample, time, condition
        )
        return self.model.forward_head(features), self.scale_head(sample, *context)

    def get_optimizer_groups(self, weight_decay: float):
        return super().get_optimizer_groups(weight_decay) + [
            {
                "params": [
                    p
                    for name, p in self.scale_head.named_parameters()
                    if name.endswith("weight")
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for name, p in self.scale_head.named_parameters()
                    if name.endswith("bias")
                ],
                "weight_decay": 0.0,
            },
        ]
