# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-compatible regression head built from official GR00T modules.

The constructor follows Isaac-GR00T b995540 with an optional scale decoder.
Its ordering preserves the released HT initialization and parameter names.
Flow computation remains owned by the official action head.
"""

import torch
import torch.nn.functional as F
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
from gr00t.model.modules.dit import AlternateVLDiT, DiT, SelfAttentionTransformer
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
from torch import nn
from torch.distributions import Beta
from transformers.feature_extraction_utils import BatchFeature

from ...objectives.regression.HT.native import gr00t_loss


class GR00TNativeActionHead(Gr00tN1d7ActionHead):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim
        model_kwargs = dict(
            config.diffusion_model_cfg,
            cross_attention_dim=config.backbone_embedding_dim,
        )
        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **model_kwargs,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
        else:
            self.model = DiT(**model_kwargs)
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps
        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = self._new_decoder()
        if config.loss_type == "hetero_t":
            self.sigma_decoder = self._new_decoder()
        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim)
            if config.use_vlln
            else nn.Identity()
        )
        vl_config = getattr(config, "vl_self_attention_cfg", None)
        self.vl_self_attention = (
            SelfAttentionTransformer(**vl_config)
            if vl_config and vl_config.get("num_layers", 0) > 0
            else nn.Identity()
        )
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(
                config.max_seq_len, self.input_embedding_dim
            )
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.state_dropout_prob = config.state_dropout_prob
        self.beta_dist = Beta(
            torch.tensor(
                float(config.noise_beta_alpha), dtype=torch.float32, device="cpu"
            ),
            torch.tensor(
                float(config.noise_beta_beta), dtype=torch.float32, device="cpu"
            ),
        )
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    def _new_decoder(self):
        return CategorySpecificMLP(
            num_categories=self.config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

    def set_trainable_parameters(self, tune_projector, tune_diffusion_model, tune_vlln):
        super().set_trainable_parameters(
            tune_projector, tune_diffusion_model, tune_vlln
        )
        if hasattr(self, "sigma_decoder"):
            self.sigma_decoder.requires_grad_(tune_projector)

    def set_frozen_modules_to_eval_mode(self):
        super().set_frozen_modules_to_eval_mode()
        if self.training and not self.tune_projector and hasattr(self, "sigma_decoder"):
            self.sigma_decoder.eval()

    def _regression_features(
        self, vl, state, embodiment, backbone_output, trajectory, *, training_forward
    ):
        times = torch.zeros(
            trajectory.shape[0], dtype=torch.long, device=trajectory.device
        )
        features = self.action_encoder(trajectory, times, embodiment)
        if self.config.add_pos_embed:
            positions = torch.arange(features.shape[1], device=features.device)
            features = features + self.position_embedding(positions).unsqueeze(0)
        kwargs = {}
        if training_forward:
            kwargs.update(
                encoder_attention_mask=backbone_output.backbone_attention_mask,
                return_all_hidden_states=True,
            )
        if self.config.use_alternate_vl_dit:
            kwargs.update(
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
            )
        result = self.model(
            hidden_states=torch.cat((state, features), dim=1),
            encoder_hidden_states=vl,
            timestep=times,
            **kwargs,
        )
        return result[0] if training_forward else result

    def forward(self, backbone_output, action_input):
        if self.config.loss_type == "flow":
            return super().forward(backbone_output, action_input)
        self.set_frozen_modules_to_eval_mode()
        backbone_output = self.process_backbone_output(backbone_output)
        vl = backbone_output.backbone_features
        embodiment = action_input.embodiment_id
        assert action_input.state.shape[1] == self.config.state_history_length
        state = action_input.state.reshape(action_input.state.shape[0], 1, -1)
        state = self.state_encoder(state, embodiment)
        if self.training and self.state_dropout_prob > 0:
            dropped = (
                torch.rand(state.shape[0], device=state.device)
                < self.state_dropout_prob
            )
            state = state * (1 - dropped[:, None, None].to(state.dtype))
        actions = action_input.action
        features = self._regression_features(
            vl,
            state,
            embodiment,
            backbone_output,
            torch.zeros_like(actions),
            training_forward=True,
        )
        prediction = self.action_decoder(features, embodiment)[:, -actions.shape[1] :]
        mask = action_input.action_mask
        if self.config.loss_type == "hetero_t":
            scale = self.sigma_decoder(features, embodiment)[:, -actions.shape[1] :]
            result = gr00t_loss(
                prediction,
                actions,
                scale,
                mask,
                df=self.config.ht_df,
                multivariate=self.config.ht_mvt,
                scale_bias=self.config.ht_sbias,
            )
            loss, action_loss = result["loss"], result["action_loss"]
        else:
            criterion = F.mse_loss if self.config.loss_type == "mse" else F.l1_loss
            action_loss = criterion(prediction, actions, reduction="none") * mask
            loss = action_loss.sum() / (mask.sum() + 1e-6)
        return dict(
            loss=loss,
            action_loss=action_loss,
            action_mask=mask,
            backbone_features=vl,
            state_features=state,
        )

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features,
        state_features,
        embodiment_id,
        backbone_output,
        action_input,
        options=None,
    ):
        if self.config.loss_type == "flow":
            return super().get_action_with_features(
                backbone_features,
                state_features,
                embodiment_id,
                backbone_output,
                action_input,
                options,
            )
        if options:
            raise ValueError("Regression inference does not support RTC options.")
        zeros = backbone_features.new_zeros(
            (backbone_features.shape[0], self.action_horizon, self.action_dim)
        )
        features = self._regression_features(
            backbone_features,
            state_features,
            embodiment_id,
            backbone_output,
            zeros,
            training_forward=False,
        )
        prediction = self.action_decoder(features, embodiment_id)[
            :, -self.action_horizon :
        ]
        return BatchFeature(
            dict(
                action_pred=prediction,
                backbone_features=backbone_features,
                state_features=state_features,
            )
        )
