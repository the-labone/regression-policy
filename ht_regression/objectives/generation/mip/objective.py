"""MIP iterative generation through the objective-independent policy interface.

Matches the current mip_loss / mip_sampler formulation in
https://github.com/simchowitzlabpublic/much-ado-about-noising/tree/main/mip
The earlier paper formulation has different target/input scaling.
"""

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from ....policies.base import BasePolicy, SampleSpec
from ...base import BaseObjective
from ...common import (
    gaussian_noise,
    loss_precision,
    loss_weights,
    validate_prediction,
    validate_target,
)


class MIPObjective(BaseObjective):
    """Two clean-target regressions with different inputs and residual scales.

    With encoded clean target x, tau=t_two_step and e~N(0,I), training uses
        p0 = policy(0, 0, condition)
        p1 = policy(x + (1-tau)*e, tau, condition)
        loss = scale * mean(sum(norm((p0-x)/tau) + norm((p1-x)/(1-tau))))
    Here norm is elementwise (squared error by default), summed over the last
    feature axis and averaged over the remaining axes. The second training input
    is constructed from ground truth, not from p0. Both calls share the encoded
    condition and retain gradients into its encoder.

    Inference starts from zero and returns policy(policy(0,0,c),tau,c).
    Predictions are clean samples in the policy's representation, not velocities;
    there is no Euler integration, random initialization or inference scheduler.
    Normalization and action decoding remain policy responsibilities.

    Masks weight the loss only, with the same reduction convention as flow.
    validate_values=False keeps metadata checks but skips synchronizing value
    checks for fullgraph compilation; the trainer then checks loss/gradients.
    """

    def __init__(
        self,
        *,
        t_two_step: float = 0.9,
        norm_type: str = "l2",
        loss_scale: float = 0.1,
        validate_values: bool = True,
    ):
        super().__init__()
        self.validate_values = validate_values
        self._configure(
            t_two_step=t_two_step, norm_type=norm_type, loss_scale=loss_scale
        )

    def compute_loss(
        self,
        policy: BasePolicy,
        target: Tensor,
        condition: Any,
        *,
        loss_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Supervise both passes with clean targets; noise overrides the Gaussian draw."""
        validate_target(target, check_values=self.validate_values)
        noise = gaussian_noise(
            target, noise, generator=generator, check_values=self.validate_values
        )
        initial_time, refinement_time = self._times(target.shape[0], target.device)
        refinement_input = target + (1 - self.t_two_step) * noise

        initial_prediction = self._predict(
            policy, torch.zeros_like(target), initial_time, condition
        )
        refined_prediction = self._predict(
            policy, refinement_input, refinement_time, condition
        )

        # Scale residuals BEFORE the norm (matters for L1 and smooth L1).
        initial_error = self._regression_error(
            initial_prediction, target, interval=self.t_two_step
        )
        refinement_error = self._regression_error(
            refined_prediction, target, interval=1 - self.t_two_step
        )
        return self._reduce_loss(initial_error, refinement_error, loss_mask)

    @torch.no_grad()
    def sample(
        self,
        policy: BasePolicy,
        condition: Any,
        *,
        sample_spec: SampleSpec,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Two deterministic predictions; the optional generator is not consumed."""
        if policy.training:
            raise RuntimeError("Call policy.eval() before sampling to disable dropout.")
        initial_sample = torch.zeros(
            sample_spec.shape, device=sample_spec.device, dtype=sample_spec.dtype
        )
        initial_time, refinement_time = self._times(
            sample_spec.shape[0], sample_spec.device
        )
        initial_prediction = self._predict(
            policy, initial_sample, initial_time, condition
        )
        # Preserve the model-space input dtype under autocast, as flow sampling does.
        refinement_input = initial_prediction.to(sample_spec.dtype)
        return self._predict(policy, refinement_input, refinement_time, condition).to(
            sample_spec.dtype
        )

    def _times(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
        initial = torch.zeros(batch_size, device=device, dtype=torch.float32)
        refinement = torch.full(
            (batch_size,), self.t_two_step, device=device, dtype=torch.float32
        )
        return initial, refinement

    @staticmethod
    def _predict(
        policy: BasePolicy, sample: Tensor, time: Tensor, condition: Any
    ) -> Tensor:
        prediction = policy(sample, time, condition)
        validate_prediction(prediction, sample)
        return prediction

    def _regression_error(
        self, prediction: Tensor, target: Tensor, *, interval: float
    ) -> Tensor:
        # Compute scaled errors in fp32 for low-precision predictions/targets.
        prediction = loss_precision(prediction)
        target = loss_precision(target)
        residual = (prediction - target) / interval
        if self.norm_type == "l2":
            return residual.square()
        if self.norm_type == "l1":
            return residual.abs()
        return F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none")

    def _reduce_loss(
        self, initial_error: Tensor, refinement_error: Tensor, loss_mask: Tensor | None
    ) -> Tensor:
        """Sum features and average other axes; all-ones masks preserve that scale."""
        if loss_mask is None:
            # Preserve upstream's reduction order: feature sums, then stage sum.
            return (
                self.loss_scale
                * (initial_error.sum(dim=-1) + refinement_error.sum(dim=-1)).mean()
            )
        error = initial_error + refinement_error
        weights = loss_weights(loss_mask, error, check_values=self.validate_values)
        denominator = weights.sum()
        return self.loss_scale * error.shape[-1] * (error * weights).sum() / denominator

    def _configure(
        self, *, t_two_step: float, norm_type: str, loss_scale: float
    ) -> None:
        if not math.isfinite(t_two_step) or not 0 < t_two_step < 1:
            raise ValueError("t_two_step must be finite and strictly between 0 and 1.")
        if norm_type not in ("l2", "l1", "smooth_l1"):
            raise ValueError("norm_type must be 'l2', 'l1' or 'smooth_l1'.")
        if not math.isfinite(loss_scale) or loss_scale <= 0:
            raise ValueError("loss_scale must be finite and positive.")
        self.t_two_step = float(t_two_step)
        self.norm_type = norm_type
        self.loss_scale = float(loss_scale)

    def get_extra_state(self):
        return {
            "version": 1,
            "t_two_step": self.t_two_step,
            "norm_type": self.norm_type,
            "loss_scale": self.loss_scale,
        }

    def set_extra_state(self, state):
        if state.get("version") != 1:
            raise ValueError("Unsupported MIP objective state version.")
        self._configure(
            t_two_step=state["t_two_step"],
            norm_type=state["norm_type"],
            loss_scale=state["loss_scale"],
        )
