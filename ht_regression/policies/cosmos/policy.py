"""Cosmos joint video/action network behind the shared policy interface."""

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from ..base import BasePolicy, PolicyTarget, SampleSpec, Timestep
from .processing import CosmosCondition


class CosmosPolicy(BasePolicy):
    """One network call, independent of the loss and sampling algorithm.

    Accept a constructed native Cosmos3VFMNetwork and processor-prepared packed
    sequences. Samples/targets are normalized, padded action chunks. Decoding
    removes padding; native ActionProcessor still owns physical action decoding.

    ``clean_action`` checkpoints consume direct time and emit clean actions.
    ``velocity`` checkpoints use native data-to-noise time and velocity; forward
    maps canonical noise-to-data time s to (1-s)*time_scale and negates velocity.
    This is a checkpoint parameterization, not a choice of training objective.

    Video context is supplied by the caller. forward_modalities additionally
    exposes native video predictions for an external joint video/action loss or
    solver. An action-only shared objective does not reproduce the historical
    auxiliary video loss or the joint video/action UniPC sampling recipe.
    """

    def __init__(
        self,
        network: nn.Module,
        *,
        horizon: int = 16,
        action_dim: int | None = None,
        output_action_dim: int = 10,
        n_action_steps: int | None = None,
        prediction_type: str = "clean_action",
        time_scale: float = 1000.0,
    ):
        super().__init__()
        self.network = network
        self.horizon = horizon
        self.action_dim = network.action_dim if action_dim is None else action_dim
        self.output_action_dim = output_action_dim
        self.n_action_steps = horizon if n_action_steps is None else n_action_steps
        self.action_start_index = 0
        for value in (horizon, self.action_dim, output_action_dim, self.n_action_steps):
            if type(value) is not int or value < 1:
                raise ValueError(
                    "Action dimensions and horizons must be positive integers."
                )
        if output_action_dim > self.action_dim or self.n_action_steps > horizon:
            raise ValueError("Action decoding must fit the padded model chunk.")
        if prediction_type not in ("clean_action", "velocity"):
            raise ValueError("prediction_type must be clean_action or velocity.")
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("time_scale must be finite and positive.")
        self.prediction_type = prediction_type
        self.time_scale = float(time_scale)
        self.train(network.training)

    def encode_condition(
        self, observation: Mapping[str, Any] | CosmosCondition
    ) -> CosmosCondition:
        condition = (
            observation
            if isinstance(observation, CosmosCondition)
            else CosmosCondition(
                observation["packed_sequence"],
                observation.get("memory"),
                observation.get("video_temporal_causal"),
            )
        )
        pack = condition.packed_sequence
        if pack.action is None or pack.vision is None or pack.sound is not None:
            raise ValueError("Expected a packed video/action sequence without sound.")
        if not pack.action.tokens or any(
            token.shape != (self.horizon, self.action_dim)
            for token in pack.action.tokens
        ):
            raise ValueError("Packed actions must match the configured chunk shape.")
        if any(torch.count_nonzero(mask).item() for mask in pack.action.condition_mask):
            raise ValueError("Action conditioning/inpainting is not supported.")
        return condition

    def encode_target(self, batch: Mapping[str, Any]) -> PolicyTarget:
        action = batch["action"]
        if (
            action.ndim != 3
            or action.shape[1:] != (self.horizon, self.output_action_dim)
            or not action.is_floating_point()
        ):
            raise ValueError(
                "action must be normalized floating point (B,H,real_action_dim)."
            )
        action = action.to(device=self.device, dtype=self.dtype)
        target = torch.nn.functional.pad(
            action, (0, self.action_dim - self.output_action_dim)
        )
        weights = torch.zeros_like(target)
        weights[..., : self.output_action_dim] = 1
        return PolicyTarget(target, weights)

    def sample_spec(self, condition: CosmosCondition) -> SampleSpec:
        tokens = condition.packed_sequence.action.tokens
        return SampleSpec(
            (len(tokens), self.horizon, self.action_dim),
            tokens[0].device,
            tokens[0].dtype,
        )

    def _native_time(self, time: Timestep, sample: Tensor) -> Tensor:
        time = torch.as_tensor(time, device=sample.device, dtype=torch.float32)
        if time.ndim == 0 or time.shape == (1,):
            time = time.expand(sample.shape[0])
        if time.shape != (sample.shape[0],):
            raise ValueError("time must be scalar or shape (B,).")
        if self.prediction_type == "velocity":
            time = 1 - time
        return time * self.time_scale

    def forward_modalities(
        self,
        sample: Tensor,
        time: Timestep,
        condition: CosmosCondition,
        *,
        network=None,
    ):
        """Return native readouts, preserving video gradients and velocity signs."""
        self.sample_spec(condition).validate(sample, check_values=False)
        prepared = condition.with_actions(sample, self._native_time(time, sample))
        return self.forward_native(prepared, network=network)

    def forward_native(self, condition: CosmosCondition, *, network=None):
        """One native prediction, preserving packed times for external joint solvers."""
        network = self.network if network is None else network
        return network(
            packed_seq=condition.packed_sequence,
            memory=condition.memory,
            video_temporal_causal=condition.video_temporal_causal,
        )

    def forward(
        self, sample: Tensor, time: Timestep, condition: CosmosCondition
    ) -> Tensor:
        outputs = self.forward_modalities(sample, time, condition)
        prediction = torch.stack(outputs["preds_action"])
        if prediction.shape != sample.shape:
            raise ValueError("Cosmos action readout must match the input chunk.")
        return -prediction if self.prediction_type == "velocity" else prediction

    def decode_prediction(self, prediction: Tensor) -> dict[str, Tensor]:
        if prediction.ndim != 3 or prediction.shape[1:] != (
            self.horizon,
            self.action_dim,
        ):
            raise ValueError("Prediction must match the padded model chunk.")
        action = prediction[..., : self.output_action_dim]
        return {"action_pred": action, "action": action[:, : self.n_action_steps]}

    def get_extra_state(self):
        return dict(
            version=1,
            horizon=self.horizon,
            action_dim=self.action_dim,
            output_action_dim=self.output_action_dim,
            n_action_steps=self.n_action_steps,
            prediction_type=self.prediction_type,
            time_scale=self.time_scale,
        )

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Cosmos policy settings differ from the saved checkpoint.")
