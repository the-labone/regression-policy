"""Direct clean-target regression through the shared policy interface."""

import math
from typing import Any

import torch
from torch import Tensor

from ....policies.base import BasePolicy, SampleSpec
from ...base import BaseObjective
from ...common import loss_precision, loss_weights, validate_prediction, validate_target


class MSEObjective(BaseObjective):
    """Train and infer with one call: prediction = policy(zeros, time=0, condition).

    Targets are clean samples encoded by the policy. The default loss is mean
    squared error over every batch, horizon and feature element. A broadcastable
    loss mask supplies nonnegative weights; its expanded sum is the denominator.
    There is no feature-sum multiplier, noise, refinement or sampling schedule.
    Normalization and decoding belong to the policy.

    Set validate_values=False for fullgraph compilation, retaining shape/device
    checks while leaving nonfinite loss/gradient detection to the trainer.
    """

    def __init__(self, *, loss_scale: float = 1.0, validate_values: bool = True):
        super().__init__()
        self.validate_values = validate_values
        self._configure(loss_scale)

    def compute_loss(
        self,
        policy: BasePolicy,
        target: Tensor,
        condition: Any,
        *,
        loss_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Regress clean targets; generator is accepted but not consumed."""
        validate_target(target, check_values=self.validate_values)
        prediction = self._predict(policy, torch.zeros_like(target), condition)

        # Accumulate squared errors in fp32 under autocast to avoid fp16 overflow.
        prediction = loss_precision(prediction)
        target = loss_precision(target)
        error = (prediction - target).square()
        if loss_mask is None:
            return self.loss_scale * error.mean()
        weights = loss_weights(loss_mask, error, check_values=self.validate_values)
        weighted_mean = (error * weights).sum() / weights.sum()
        return self.loss_scale * weighted_mean

    @torch.no_grad()
    def sample(
        self,
        policy: BasePolicy,
        condition: Any,
        *,
        sample_spec: SampleSpec,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """One deterministic prediction; generator is accepted but not consumed."""
        if policy.training:
            raise RuntimeError("Call policy.eval() before sampling to disable dropout.")
        sample = torch.zeros(
            sample_spec.shape, device=sample_spec.device, dtype=sample_spec.dtype
        )
        return self._predict(policy, sample, condition).to(sample_spec.dtype)

    @staticmethod
    def _predict(policy: BasePolicy, sample: Tensor, condition: Any) -> Tensor:
        time = torch.zeros(sample.shape[0], device=sample.device, dtype=torch.float32)
        prediction = policy(sample, time, condition)
        validate_prediction(prediction, sample)
        return prediction

    def _configure(self, loss_scale: float) -> None:
        if not math.isfinite(loss_scale) or loss_scale <= 0:
            raise ValueError("loss_scale must be finite and positive.")
        self.loss_scale = float(loss_scale)

    def get_extra_state(self):
        return {"version": 1, "loss_scale": self.loss_scale}

    def set_extra_state(self, state):
        if state.get("version") != 1:
            raise ValueError("Unsupported MSE objective state version.")
        self._configure(state["loss_scale"])
