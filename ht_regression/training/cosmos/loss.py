"""Historical Cosmos calibration and scale readout, shared HT likelihood.

Ported from much-ado scripts/cluster/cosmos_direct_action.py. The sigma head
reads detached, padded predictions; padding is excluded only from residuals.
"""

import math

import torch
import torch.nn.functional as F

from ...objectives.regression.HT.native import chunk_nll

NU = 640.0
EPS = 1e-3


def masked_residual_stats(pred, target, masks, raw_dims):
    sse = pred[0].new_zeros((), dtype=torch.float64)
    count = sse.clone()
    per_sample = []
    for p, a, cm, width in zip(pred, target, masks, raw_dims, strict=True):
        if p.shape != a.shape or p.ndim != 2:
            raise ValueError(f"Action shape mismatch {p.shape} vs {a.shape}")
        width = int(width)
        if width != 10 or a.shape[0] != 16:
            raise ValueError(
                f"Expected LIBERO 16x10 valid actions, got {a.shape}, width={width}"
            )
        mask = torch.ones_like(a, dtype=torch.float32)
        if cm is not None:
            mask *= 1 - cm.float().reshape(-1, 1)
        mask[:, width:] = 0
        d = mask.sum()
        if d.item() != 160:
            raise ValueError(f"All 16 action steps must be targets; d={d.item()}")
        r2 = (((p.float() - a.float()) * mask) ** 2).sum()
        per_sample.append((r2, d))
        sse += r2.detach().double()
        count += d.detach().double()
    return sse, count, per_sample


def bias_for_rms(rms):
    value = float(rms) - EPS
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid initial residual RMS {rms}")
    return value + math.log(-math.expm1(-value))


def direct_loss(model, pred, target, pack):
    _, _, stats = masked_residual_stats(
        pred, target, pack.action.condition_mask, pack.action.raw_action_dim
    )
    head = model.net.llm2action_ht_sigma_head
    losses, sigmas = [], []
    for p, (sse, d) in zip(pred, stats, strict=True):
        # Preserve the former Cosmos sigma-head architecture and detach policy output.
        sigma = F.softplus(head(p.float().detach()).mean()) + EPS
        loss = chunk_nll(sse, sigma, d, NU)
        losses.append(loss / d)
        sigmas.append(sigma.detach())
    loss = torch.stack(losses).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite direct-action likelihood")
    return loss, torch.stack(sigmas).mean()
