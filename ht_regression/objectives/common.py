"""Tensor utilities shared by objectives; each objective owns its math and reduction.

Value checks can synchronize CUDA and are disabled for trusted, compiled training.
Shape, dtype and device checks remain enabled in either mode.
"""

import torch
from torch import Tensor


def validate_target(target: Tensor, *, check_values: bool) -> None:
    if target.ndim < 2 or not target.numel() or not target.is_floating_point():
        raise ValueError("target must be a nonempty floating-point batch, rank >= 2.")
    if check_values and not torch.isfinite(target).all():
        raise ValueError("target must be finite.")


def validate_prediction(prediction: Tensor, sample: Tensor) -> None:
    # Autocast may change the output dtype, so require floating point, not equality.
    if (
        not isinstance(prediction, Tensor)
        or prediction.shape != sample.shape
        or prediction.device != sample.device
        or not prediction.is_floating_point()
    ):
        raise ValueError(
            "policy prediction must be floating point with the sample shape and device."
        )


def loss_precision(value: Tensor) -> Tensor:
    """Promote low precision for loss arithmetic while preserving fp64 inputs."""
    return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value


def gaussian_noise(
    target: Tensor,
    noise: Tensor | None,
    *,
    generator: torch.Generator | None,
    check_values: bool,
) -> Tensor:
    """Use supplied noise or draw N(0, I), matching the target's representation.

    Callers control when the draw happens relative to time sampling so each
    objective preserves its seeded random sequence.
    """
    if noise is None:
        noise = torch.randn(
            target.shape,
            device=target.device,
            dtype=target.dtype,
            generator=generator,
        )
    if (
        noise.shape != target.shape
        or noise.device != target.device
        or noise.dtype != target.dtype
    ):
        raise ValueError("noise must match the target shape, dtype and device.")
    if check_values and not torch.isfinite(noise).all():
        raise ValueError("noise must be finite.")
    return noise


def loss_weights(mask: Tensor, reference: Tensor, *, check_values: bool) -> Tensor:
    """Expand nonnegative weights to the loss shape, without choosing a reduction."""
    if mask.device != reference.device or (
        check_values and (not torch.isfinite(mask).all() or (mask < 0).any())
    ):
        raise ValueError(
            "loss_mask must be finite, nonnegative and on the target device."
        )
    try:
        weights = torch.broadcast_to(mask, reference.shape).to(reference.dtype)
    except RuntimeError as exc:
        raise ValueError("loss_mask must broadcast to the target shape.") from exc
    if check_values and weights.sum() <= 0:
        raise ValueError("loss_mask must select at least one target element.")
    return weights
