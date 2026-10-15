"""Heteroscedastic multivariate Student-t regression, normalized per element."""

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from ....policies.base import BasePolicy, SampleSpec
from ...base import BaseObjective
from ...common import loss_precision, loss_weights, validate_prediction, validate_target


class HTObjective(BaseObjective):
    """One-pass clean prediction with a learned scalar uncertainty per sample.

    sigma = softplus(raw_chunk_scale + scale_bias) + min_scale
    S = sum((prediction - target)^2) over ALL non-batch dimensions
    L = 0.5*(nu+d)*log1p(S/(nu*sigma^2)) + d*log(sigma)
    loss = sum_B(L) / sum_B(d)

    d counts valid elements in the chunk. nu is the actual multivariate degrees
    of freedom; None selects the guide's nu=4*d prescription per sample. There
    is no arbitrary loss multiplier or auxiliary MSE. Terms constant with respect
    to the mean/scale parameters are omitted. nu is fixed, not learned.

    scale_bias is an explicit initialization setting: calibrate it from initial
    residual RMS via inverse softplus. The neutral default 0 is not a calibration.

    Policies return exactly one raw chunk scale per sample, shaped (B, 1).
    Masks apply to residuals and effective dimension; sigma is shared by the chunk.
    Reduction is over valid elements across the batch; empty samples contribute
    zero, while a wholly empty mask is invalid. Fractional weights are supported.
    Low-precision outputs are promoted to fp32 for the likelihood computation.
    Inference uses one mean prediction, with no scale readout.
    """

    def __init__(
        self,
        *,
        nu: float | None = None,
        scale_bias: float = 0.0,
        min_scale: float = 1e-3,
        validate_values: bool = True,
    ):
        super().__init__()
        self.validate_values = validate_values
        self._configure(nu, scale_bias, min_scale)

    def compute_loss(
        self,
        policy: BasePolicy,
        target: Tensor,
        condition: Any,
        *,
        loss_mask: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Joint mean/scale NLL; neither branch nor the residual is detached."""
        validate_target(target, check_values=self.validate_values)
        sample = torch.zeros_like(target)
        time = torch.zeros(target.shape[0], device=target.device, dtype=torch.float32)
        prediction, raw_scale = policy.forward_with_scale(sample, time, condition)
        validate_prediction(prediction, target)
        if (
            not isinstance(raw_scale, Tensor)
            or raw_scale.device != target.device
            or not raw_scale.is_floating_point()
        ):
            raise ValueError("raw_scale must be floating point on the target device.")
        if raw_scale.shape != (target.shape[0], 1):
            raise ValueError("raw_scale must have shape (B, 1): one scalar per chunk.")

        return self._likelihood_loss(prediction, raw_scale, target, loss_mask)

    def _likelihood_loss(
        self,
        prediction: Tensor,
        raw_scale: Tensor,
        target: Tensor,
        loss_mask: Tensor | None,
    ) -> Tensor:
        """Transform the chunk scale, reduce residuals, and evaluate the NLL."""
        with torch.autocast(device_type=target.device.type, enabled=False):
            prediction = loss_precision(prediction)
            raw_scale = loss_precision(raw_scale)
            target = loss_precision(target)
            sigma = F.softplus(raw_scale[:, 0] + self.scale_bias) + self.min_scale
            squared_error = (prediction - target).square()
            if loss_mask is None:
                count = target[0].numel()
                residual_sum = squared_error.flatten(1).sum(1)
                return self._nll(residual_sum, sigma, count).mean() / count

            weights = loss_weights(
                loss_mask, squared_error, check_values=self.validate_values
            )
            count = weights.flatten(1).sum(1)
            residual_sum = (squared_error * weights).flatten(1).sum(1)
            per_sample = self._nll(residual_sum, sigma, count)
            return per_sample.sum() / count.sum()

    def _nll(self, residual_sum: Tensor, sigma: Tensor, count: Tensor | int) -> Tensor:
        # For an empty masked sample, choose a positive nu: S=d=0 then gives L=0.
        dimension = (
            torch.where(count > 0, count, torch.ones_like(count))
            if isinstance(count, Tensor)
            else count
        )
        nu = 4 * dimension if self.nu is None else self.nu
        return (
            0.5 * (nu + count) * torch.log1p(residual_sum / (nu * sigma.square()))
            + count * sigma.log()
        )

    @torch.no_grad()
    def sample(
        self,
        policy: BasePolicy,
        condition: Any,
        *,
        sample_spec: SampleSpec,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Predict the mean once at zero input/time; generator is not consumed."""
        if policy.training:
            raise RuntimeError("Call policy.eval() before sampling to disable dropout.")
        sample = torch.zeros(
            sample_spec.shape, device=sample_spec.device, dtype=sample_spec.dtype
        )
        time = torch.zeros(
            sample_spec.shape[0], device=sample_spec.device, dtype=torch.float32
        )
        prediction = policy(sample, time, condition)
        validate_prediction(prediction, sample)
        return prediction.to(sample_spec.dtype)

    def _configure(self, nu, scale_bias, min_scale):
        if nu is not None and (not math.isfinite(nu) or nu <= 0):
            raise ValueError("nu must be None or finite and positive.")
        if not math.isfinite(scale_bias):
            raise ValueError("scale_bias must be finite.")
        if not math.isfinite(min_scale) or min_scale <= 0:
            raise ValueError("min_scale must be finite and positive.")
        self.nu = None if nu is None else float(nu)
        self.scale_bias = float(scale_bias)
        self.min_scale = float(min_scale)

    def get_extra_state(self):
        return {
            "version": 2,
            "nu": self.nu,
            "scale_bias": self.scale_bias,
            "min_scale": self.min_scale,
        }

    def set_extra_state(self, state):
        if state.get("version") != 2:
            raise ValueError(
                "Unsupported HT objective state: expected scalar-head version 2; "
                "scaled-loss and elementwise-head checkpoints require their original source snapshot."
            )
        self._configure(state["nu"], state["scale_bias"], state["min_scale"])
