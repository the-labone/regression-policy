"""HT likelihood primitives used by native VLA trainers.

Keep model-specific scale readouts and pooling outside the likelihood. In
particular, GR00T pools positive elementwise scales; PI0.5 pools raw scales
in its model's dtype; Cosmos reads detached predictions. These are distinct
historical recipes, not interchangeable defaults of the RoboMimic objective.
"""

import torch
import torch.nn.functional as F


def chunk_nll(squared_residual, sigma, dimension, nu):
    """Unnormalized joint Student-t NLL with literal multivariate nu."""
    return 0.5 * (nu + dimension) * torch.log1p(
        squared_residual / (nu * sigma**2)
    ) + dimension * torch.log(sigma)


def scaled_df_nll(squared_residual, sigma, dimension, df):
    """Historical df encoding: effective joint nu = df * dimension.

    Preserve the original operation order as well as the mathematical formula.
    """
    return 0.5 * (df + 1.0) * torch.log1p(
        squared_residual / (df * sigma**2 * dimension)
    ) * dimension + dimension * torch.log(sigma)


def gr00t_loss(prediction, target, raw_scale, mask, *, df, multivariate, scale_bias):
    """Native padding-aware GR00T loss, including its denominator epsilon."""
    if not prediction.shape == target.shape == raw_scale.shape == mask.shape:
        raise ValueError("Prediction, target, raw scale and mask shapes must match.")
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        mask = mask.float()
        axes = tuple(range(1, mask.ndim))
        dimension = mask.sum(dim=axes).clamp_min(1.0)
        sigma = (F.softplus(raw_scale.float() + scale_bias) * mask).sum(
            dim=axes
        ) / dimension + 1e-3
        residual = (((target.float() - prediction.float()) ** 2) * mask).sum(dim=axes)
        nll = chunk_nll if multivariate else scaled_df_nll
        per_sample = nll(residual, sigma, dimension, df)
        return {
            "loss": per_sample.sum() / (mask.sum() + 1e-6),
            "action_loss": per_sample,
            "sigma": sigma,
        }
