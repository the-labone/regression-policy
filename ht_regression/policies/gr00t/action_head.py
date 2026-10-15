# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Single GR00T N1.7 network prediction, separated from corruption and losses.

Adapted from GROOT_HT/gr00t/model/gr00t_n1d7/gr00t_n1d7.py at
b861ec9090b841cc2bbae3cdc6b8d2a384a3660b. See LICENSE.gr00t.
Upstream code license:
https://github.com/NVIDIA/Isaac-GR00T/blob/main/LICENSE
The supplied head's modules are reused, not copied or reinitialized. Do not keep
training the original head after transferring it to this adapter.
"""

import torch
from torch import Tensor, nn

from ..base import Timestep
from .processing import GR00TCondition


class GR00TActionHead(nn.Module):
    """Reuse upstream network modules without retaining its loss or sampler.

    ``normalized`` time maps t to floor(t * num_timestep_buckets), as in GR00T
    flow. ``discrete`` accepts bucket indices directly (e.g. a diffusion recipe).
    Choose explicitly when composing objectives; no time distribution lives here.
    """

    def __init__(
        self,
        head: nn.Module,
        *,
        time_mode: str = "normalized",
        validate_values: bool = True,
    ):
        super().__init__()
        cfg = head.config
        self.horizon = cfg.action_horizon
        self.action_dim = cfg.max_action_dim
        self.state_dim = cfg.max_state_dim
        self.state_history_length = cfg.state_history_length
        self.num_embodiments = cfg.max_num_embodiments
        self.num_timestep_buckets = cfg.num_timestep_buckets
        self.state_dropout_prob = cfg.state_dropout_prob
        self.use_alternate_vl_dit = cfg.use_alternate_vl_dit
        if time_mode not in ("normalized", "discrete"):
            raise ValueError("time_mode must be 'normalized' or 'discrete'.")
        self.time_mode = time_mode
        self.validate_values = validate_values
        if not 0 <= self.state_dropout_prob <= 1:
            raise ValueError("state_dropout_prob must be in [0, 1].")
        for name in (
            "state_encoder",
            "action_encoder",
            "action_decoder",
            "model",
            "vlln",
            "vl_self_attention",
        ):
            setattr(self, name, getattr(head, name))
        self.position_embedding = head.position_embedding if cfg.add_pos_embed else None
        self.tune_projector = head.tune_projector
        self.tune_diffusion_model = head.tune_diffusion_model
        self.tune_vlln = head.tune_vlln
        self.train(head.training)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            groups = (
                (
                    self.tune_projector,
                    (
                        self.state_encoder,
                        self.action_encoder,
                        self.action_decoder,
                        self.position_embedding,
                    ),
                ),
                (self.tune_diffusion_model, (self.model,)),
                (self.tune_vlln, (self.vlln, self.vl_self_attention)),
            )
            for trainable, modules in groups:
                if not trainable:
                    for module in modules:
                        if module is not None:
                            module.eval()
        return self

    def encode_condition(self, output, state: Tensor, embodiment_id: Tensor):
        """Encode condition once; preserve gradients and apply state dropout once."""
        features = self.vl_self_attention(self.vlln(output["backbone_features"]))
        state_features = self.state_encoder(
            state.reshape(state.shape[0], 1, -1), embodiment_id
        )
        if self.training and self.state_dropout_prob > 0:
            dropped = (
                torch.rand(state.shape[0], device=state.device)
                < self.state_dropout_prob
            )
            state_features = state_features * (~dropped[:, None, None]).to(
                state_features.dtype
            )
        return GR00TCondition(
            features,
            state_features,
            embodiment_id,
            output["backbone_attention_mask"],
            output.get("image_mask"),
        )

    def encode_time(self, time: Timestep, sample: Tensor) -> Tensor:
        time = torch.as_tensor(time, device=sample.device)
        if time.ndim == 0 or time.shape == (1,):
            time = time.expand(sample.shape[0])
        if time.shape != (sample.shape[0],):
            raise ValueError("time must be a scalar or shape (B,).")
        if self.validate_values and not torch.isfinite(time).all():
            raise ValueError("time must be finite.")
        if self.time_mode == "normalized":
            if self.validate_values and ((time < 0).any() or (time > 1).any()):
                raise ValueError("Normalized time must be in [0, 1].")
            return (time * self.num_timestep_buckets).long()
        if self.validate_values and (
            (time < 0).any()
            or (time > self.num_timestep_buckets).any()
            or (time != time.long()).any()
        ):
            raise ValueError(
                "Discrete time must be an integer bucket in [0, num_timestep_buckets]."
            )
        return time.long()

    def get_extra_state(self):
        return {
            name: getattr(self, name)
            for name in (
                "time_mode",
                "horizon",
                "action_dim",
                "state_dim",
                "state_history_length",
                "num_embodiments",
                "num_timestep_buckets",
                "state_dropout_prob",
                "use_alternate_vl_dit",
                "tune_projector",
                "tune_diffusion_model",
                "tune_vlln",
            )
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError(
                "GR00T action-head settings differ from the saved checkpoint."
            )

    def forward_features(
        self, sample: Tensor, time: Timestep, condition: GR00TCondition
    ):
        expected = (condition.state_features.shape[0], self.horizon, self.action_dim)
        if sample.shape != expected:
            raise ValueError(f"sample must have shape {expected}.")
        if sample.device != self.device or not sample.is_floating_point():
            raise ValueError("sample must be floating point on the action-head device.")
        timestep = self.encode_time(time, sample)
        action_features = self.action_encoder(sample, timestep, condition.embodiment_id)
        if self.position_embedding is not None:
            positions = torch.arange(sample.shape[1], device=sample.device)
            action_features = action_features + self.position_embedding(positions)[None]
        state_action = torch.cat((condition.state_features, action_features), dim=1)
        kwargs = {}
        if self.use_alternate_vl_dit:
            if condition.image_mask is None:
                raise ValueError("AlternateVLDiT requires image_mask.")
            kwargs = {
                "image_mask": condition.image_mask,
                "backbone_attention_mask": condition.backbone_attention_mask,
            }
        features, _ = self.model(
            hidden_states=state_action,
            encoder_hidden_states=condition.backbone_features,
            encoder_attention_mask=condition.backbone_attention_mask,
            timestep=timestep,
            return_all_hidden_states=True,
            **kwargs,
        )
        return features

    def decode_features(self, features: Tensor, condition: GR00TCondition):
        return self.action_decoder(features, condition.embodiment_id)[
            :, -self.horizon :
        ]

    def forward(self, sample: Tensor, time: Timestep, condition: GR00TCondition):
        return self.decode_features(
            self.forward_features(sample, time, condition), condition
        )
