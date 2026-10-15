"""Diffusion Policy's observation-conditioned Transformer decoder.

Adapted from real-stanford/diffusion_policy (MIT), revision
5ba07ac6661db573af695b419a7947ecb704690f, transformer_for_diffusion.py.
See ../../../LICENSE.diffusion_policy.
"""

import torch
from torch import Tensor, nn

from ..common import (
    SinusoidalPosEmb,
    Timestep,
    batch_timesteps,
    validate_dimensions,
    validate_inputs,
)


class TransformerBackbone(nn.Module):
    """DP decoder with a time token and independently projected observation tokens.

    Inputs: sample (B, horizon, action_dim), timestep scalar/(B,), and
    obs (B, n_obs_steps, obs_dim). Output: (B, horizon, action_dim).
    With the default n_cond_layers=0, observation tokens never mix before
    cross-attention. Inputs must already be normalized. No objective or sampler
    is attached, so the output may represent noise, velocity, or actions.
    """

    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        horizon: int = 10,
        n_obs_steps: int = 2,
        *,
        n_layer: int = 8,
        n_head: int = 4,
        n_emb: int = 256,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.3,
        causal_attn: bool = True,
        n_cond_layers: int = 0,
    ):
        super().__init__()
        validate_dimensions(action_dim, obs_dim, horizon, n_obs_steps)
        if n_layer < 1 or n_cond_layers < 0:
            raise ValueError("n_layer must be positive and n_cond_layers nonnegative.")
        if n_head < 1 or n_emb % n_head:
            raise ValueError("n_emb must be divisible by a positive n_head.")
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.feature_dim = n_emb

        self.input_emb = nn.Linear(action_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))
        self.drop = nn.Dropout(p_drop_emb)
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = nn.Linear(obs_dim, n_emb)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, 1 + n_obs_steps, n_emb))

        if n_cond_layers:
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
                    dropout=p_drop_attn,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=n_cond_layers,
                enable_nested_tensor=False,
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(n_emb, 4 * n_emb),
                nn.Mish(),
                nn.Linear(4 * n_emb, n_emb),
            )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=n_layer,
        )

        mask = memory_mask = None
        if causal_attn:
            mask = torch.full((horizon, horizon), float("-inf")).triu(diagonal=1)
            action_index = torch.arange(horizon)[:, None]
            condition_index = torch.arange(1 + n_obs_steps)[None, :]
            # Condition token 0 is time; token j+1 corresponds to observation j.
            memory_mask = torch.zeros(horizon, 1 + n_obs_steps).masked_fill(
                action_index < condition_index - 1, float("-inf")
            )
        self.register_buffer("mask", mask)
        self.register_buffer("memory_mask", memory_mask)
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, action_dim)

        self.apply(self._init_weights)
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.cond_pos_emb, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            for name in (
                "in_proj_weight",
                "q_proj_weight",
                "k_proj_weight",
                "v_proj_weight",
            ):
                weight = getattr(module, name, None)
                if weight is not None:
                    nn.init.normal_(weight, std=0.02)
            for name in ("in_proj_bias", "bias_k", "bias_v"):
                bias = getattr(module, name, None)
                if bias is not None:
                    nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, sample: Tensor, timestep: Timestep, obs: Tensor) -> Tensor:
        return self.forward_head(self.forward_features(sample, timestep, obs))

    def forward_head(self, features: Tensor) -> Tensor:
        return self.head(features)

    def forward_features(
        self, sample: Tensor, timestep: Timestep, obs: Tensor
    ) -> Tensor:
        """Shared B x horizon x feature_dim representation before the action head."""
        features, _ = self.forward_features_with_context(sample, timestep, obs)
        return features

    def forward_features_with_context(
        self, sample: Tensor, timestep: Timestep, obs: Tensor
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Also expose pre-LayerNorm decoder output and memory to the chunk head."""
        validate_inputs(
            sample,
            obs,
            action_dim=self.action_dim,
            obs_dim=self.obs_dim,
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps,
        )
        time = self.time_emb(batch_timesteps(timestep, sample)).to(sample.dtype)
        condition = torch.cat((time[:, None], self.cond_obs_emb(obs)), dim=1)
        memory = self.drop(condition + self.cond_pos_emb)
        # Keep the stock layers and state-dict names, but avoid the stack wrappers'
        # runtime mask detection and _get_seq_len (untraceable on some Torch builds).
        if isinstance(self.encoder, nn.TransformerEncoder):
            for layer in self.encoder.layers:
                memory = layer(memory, is_causal=False)
            if self.encoder.norm is not None:
                memory = self.encoder.norm(memory)
        else:
            memory = self.encoder(memory)
        x = self.drop(self.input_emb(sample) + self.pos_emb)
        for layer in self.decoder.layers:
            x = layer(
                tgt=x,
                memory=memory,
                tgt_mask=self.mask,
                memory_mask=self.memory_mask,
                tgt_is_causal=self.mask is not None,
                # DP's observation-time visibility mask is NOT a causal triangle.
                memory_is_causal=False,
            )
        if self.decoder.norm is not None:
            x = self.decoder.norm(x)
        return self.ln_f(x), (x, memory)

    def get_optimizer_groups(self, weight_decay: float = 1e-3) -> list[dict]:
        """DP's AdamW grouping: decay linear/attention weights, not bias/norm/pos."""
        decay = set()
        for module_name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.MultiheadAttention)):
                for name, _ in module.named_parameters(recurse=False):
                    if name.endswith("weight"):
                        decay.add(f"{module_name}.{name}" if module_name else name)
        parameters = dict(self.named_parameters())
        return [
            {
                "params": [parameters[name] for name in sorted(decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    parameters[name] for name in sorted(parameters.keys() - decay)
                ],
                "weight_decay": 0.0,
            },
        ]
