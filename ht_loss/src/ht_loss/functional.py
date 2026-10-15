"""Heteroscedastic multivariate Student-t loss; depends only on PyTorch."""

import math
from numbers import Real

import torch
import torch.nn.functional as F
from torch import Tensor


def _validate_options(nu, scale_bias, min_scale, reduction):
    for name, value in (
        ("nu", nu),
        ("scale_bias", scale_bias),
        ("min_scale", min_scale),
    ):
        if name == "nu" and value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite real number.")
        if name != "scale_bias" and value <= 0:
            raise ValueError(f"{name} must be positive.")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'.")


def _loss_precision(value: Tensor) -> Tensor:
    return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value


def ht_loss(
    input: Tensor,
    target: Tensor,
    raw_scale: Tensor,
    *,
    nu: float | None = None,
    scale_bias: float = 0.0,
    min_scale: float = 1e-3,
    reduction: str = "mean",
    mask: Tensor | None = None,
    validate_values: bool = True,
) -> Tensor:
    """Compute the HT regression loss with one learned scale per sample.

    ``input`` and ``target`` have the same nonempty shape ``(B, ...)``.
    All non-batch dimensions form ONE multivariate event. ``(B,)`` is also
    supported for scalar regression. ``raw_scale`` has shape ``(B,)`` or
    ``(B, 1)`` and is the unconstrained output of a learned scale head.

    With ``sigma = softplus(raw_scale + scale_bias) + min_scale``, squared
    residual sum S, and event dimension d, the per-sample loss is
    ``L = 0.5 * (nu + d) * log1p(S / (nu * sigma**2)) + d * log(sigma)``.
    Terms independent of the predictions and scale are omitted. ``nu`` is
    fixed; ``None`` selects ``4 * d`` for each sample.

    ``mask`` supplies nonnegative weights broadcastable to ``input.shape``;
    both S and d use the expanded weights. At least one element must have
    positive weight. Empty samples contribute zero loss and zero gradients.

    Reductions:
        ``'mean'``: sum of L divided by the total valid element weight.
        ``'sum'``: sum of L across samples.
        ``'none'``: L for each sample, shape ``(B,)``.

    Half/bfloat16 inputs are promoted to float32 for loss arithmetic; float64
    is preserved. Gradients flow through both input and raw_scale. The default
    scale_bias=0 is not data calibration. Set validate_values=False for trusted
    inputs under torch.compile; shape, dtype and device checks remain enabled.
    """
    _validate_options(nu, scale_bias, min_scale, reduction)
    for name, value in (("input", input), ("target", target), ("raw_scale", raw_scale)):
        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise ValueError(f"{name} must be a floating-point tensor.")
    if input.ndim < 1 or not input.numel():
        raise ValueError("input must be a nonempty batch with shape (B, ...).")
    if target.shape != input.shape or target.device != input.device:
        raise ValueError("target must have the same shape and device as input.")
    batch_size = input.shape[0]
    if raw_scale.shape not in ((batch_size,), (batch_size, 1)):
        raise ValueError("raw_scale must have shape (B,) or (B, 1).")
    if raw_scale.device != input.device:
        raise ValueError("raw_scale must be on the input device.")
    if validate_values:
        for name, value in (
            ("input", input),
            ("target", target),
            ("raw_scale", raw_scale),
        ):
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite.")

    with torch.autocast(device_type=input.device.type, enabled=False):
        input, target, raw_scale = map(_loss_precision, (input, target, raw_scale))
        sigma = F.softplus(raw_scale.reshape(batch_size) + scale_bias) + min_scale
        squared_error = (input - target).square()
        if mask is None:
            count = target[0].numel()
            residual_sum = squared_error.reshape(batch_size, -1).sum(1)
            dimension = count
        else:
            if (
                not isinstance(mask, Tensor)
                or mask.device != input.device
                or mask.is_complex()
            ):
                raise ValueError("mask must be a real tensor on the input device.")
            if validate_values and (not torch.isfinite(mask).all() or (mask < 0).any()):
                raise ValueError("mask must be finite and nonnegative.")
            try:
                weights = torch.broadcast_to(mask, input.shape).to(squared_error.dtype)
            except RuntimeError as exc:
                raise ValueError("mask must broadcast to input.shape.") from exc
            count = weights.reshape(batch_size, -1).sum(1)
            if validate_values and count.sum() <= 0:
                raise ValueError("mask must select at least one element.")
            residual_sum = (squared_error * weights).reshape(batch_size, -1).sum(1)
            # Empty samples use positive degrees of freedom and contribute L=0.
            dimension = torch.where(count > 0, count, torch.ones_like(count))
        degrees = 4 * dimension if nu is None else nu
        per_sample = (
            0.5
            * (degrees + count)
            * torch.log1p(residual_sum / (degrees * sigma.square()))
            + count * sigma.log()
        )
        if reduction == "none":
            return per_sample
        if reduction == "sum":
            return per_sample.sum()
        if mask is None:
            return per_sample.mean() / count
        return per_sample.sum() / count.sum()
