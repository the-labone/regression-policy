"""Training-data calibration of the initial HT scale, before optimization."""

import logging
import math
import random

import torch
from torch.utils.data import DataLoader, Subset

from ....pipelines import PolicyPipeline
from ....policies.base import BasePolicy
from ....training.batch import move_to_device
from ....training.checkpoint import isolated_rng
from .objective import HTObjective


@torch.no_grad()
def calibrate_sigma(
    pipeline: PolicyPipeline,
    dataset,
    *,
    num_samples: int = 300,
    batch_size: int = 64,
    seed: int = 0,
) -> dict:
    """Set sigma_0 to normalized residual RMS on sampled training windows.

    Call on a fresh model, on its training device, before creating EMA or
    compiling the loss. Pass the training split only. The policy encodes both
    observations and targets, so calibration uses precisely its loss space and
    mask. The raw scale readout must be zero-initialized; an already trained head
    is rejected instead of reset. Model weights and existing gradients are kept.

    Select windows without replacement, disable dropout, use full precision, and
    restore model modes and Python/NumPy/Torch RNG states even if calibration fails.
    This uses its own loader and does not consume the training loader/sampler.
    Stateful dataset side effects beyond global RNG are the caller's responsibility.

    Returns a JSON-serializable report. The bias is stored in objective extra state;
    the trainer also checkpoints the report and writes initialization.json.
    """
    if not isinstance(pipeline.objective, HTObjective):
        raise TypeError("HT sigma calibration requires HTObjective.")
    for name, value in (("num_samples", num_samples), ("batch_size", batch_size)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    if not len(dataset):
        raise ValueError("Cannot calibrate sigma on an empty training dataset.")

    objective = pipeline.objective
    samples = min(num_samples, len(dataset))
    residual_sum, valid_elements, indices = _measure_residuals(
        pipeline, dataset, samples=samples, batch_size=batch_size, seed=seed
    )

    if valid_elements <= 0:
        raise ValueError("Calibration must include at least one valid target element.")
    rms = math.sqrt(residual_sum / valid_elements)
    if not math.isfinite(rms) or rms <= objective.min_scale:
        raise ValueError(
            "Initial residual RMS must exceed min_scale; lower the scale floor before calibration."
        )
    positive_scale = rms - objective.min_scale
    # Stable inverse softplus, including small positive values and large residuals.
    bias = positive_scale + math.log(-math.expm1(-positive_scale))
    report = {
        "kind": "ht_sigma",
        "seed": seed,
        "dataset_size": len(dataset),
        "num_samples": samples,
        "sample_indices": indices,
        "valid_elements": valid_elements,
        "residual_rms": rms,
        "initial_sigma": rms,
        "previous_scale_bias": objective.scale_bias,
        "scale_bias": bias,
        "min_scale": objective.min_scale,
        "model_mode": "eval",
    }
    objective.scale_bias = bias
    logging.getLogger("ht_regression.training").info(
        "HT sigma initialized: %d training windows, residual RMS=%.6g, sigma=%.6g, bias=%.6g",
        samples,
        rms,
        rms,
        bias,
    )
    return report


def _measure_residuals(
    pipeline: PolicyPipeline, dataset, *, samples: int, batch_size: int, seed: int
) -> tuple[float, float, list[int]]:
    """Measure a reproducible subset without changing training modes or RNG."""
    policy = pipeline.policy
    modes = [(module, module.training) for module in pipeline.modules()]
    residual_sum = 0.0
    valid_elements = 0.0
    try:
        pipeline.eval()
        with isolated_rng(seed), torch.autocast(policy.device.type, enabled=False):
            indices = random.sample(range(len(dataset)), samples)
            loader = DataLoader(
                Subset(dataset, indices),
                batch_size=batch_size,
                num_workers=0,
                shuffle=False,
                generator=torch.Generator().manual_seed(seed),
            )
            for batch in loader:
                batch_sum, batch_count = _batch_residuals(policy, batch)
                residual_sum += batch_sum
                valid_elements += batch_count
    finally:
        for module, training in modes:
            module.training = training

    return residual_sum, valid_elements, indices


def _batch_residuals(policy: BasePolicy, batch) -> tuple[float, float]:
    """Return squared-residual sum and valid weight in the policy's loss space."""
    batch = move_to_device(batch, policy.device)
    condition = policy.encode_condition(batch)
    encoded = policy.encode_target(batch)
    target = encoded.sample
    time = torch.zeros(target.shape[0], device=target.device, dtype=torch.float32)
    mean, raw_scale = policy.forward_with_scale(
        torch.zeros_like(target), time, condition
    )
    if (
        mean.shape != target.shape
        or mean.device != target.device
        or not mean.is_floating_point()
    ):
        raise ValueError(
            "Mean prediction must match the target shape/device and be floating point."
        )
    if not torch.isfinite(target).all() or not torch.isfinite(mean).all():
        raise ValueError("Calibration targets and mean predictions must be finite.")
    if raw_scale.device != target.device or not raw_scale.is_floating_point():
        raise ValueError("Raw scale must be floating point on the target device.")
    if raw_scale.shape != (target.shape[0], 1):
        raise ValueError("raw_scale must have shape (B, 1): one scalar per chunk.")
    if not torch.isfinite(raw_scale).all() or torch.count_nonzero(raw_scale):
        raise ValueError(
            "Sigma calibration requires a zero-initialized raw scale head; do not recalibrate a trained head."
        )
    error = (mean.double() - target.double()).square()
    if encoded.loss_mask is None:
        return error.sum().item(), float(error.numel())
    mask = encoded.loss_mask
    if (
        mask.device != target.device
        or not torch.isfinite(mask).all()
        or (mask < 0).any()
    ):
        raise ValueError(
            "Calibration loss mask must be finite, nonnegative and on the target device."
        )
    weights = torch.broadcast_to(mask, target.shape).double()
    return (error * weights).sum().item(), weights.sum().item()
