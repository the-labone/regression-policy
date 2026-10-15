"""Evaluation-history summaries, grouped by benchmark and baseline protocol.

Each history row represents one evaluation and contains raw runner metric keys
(e.g. ``test/mean_score``), optionally with ``global_step`` and aligned losses.
Rows must be chronological and come from the same run/evaluation protocol.
Training-only rows without the score key are ignored. Average across independent
training seeds only AFTER summarizing each run; do not concatenate their histories.

These pure functions do not run rollouts, select trajectory counts, average model
weights, or save checkpoints. METRIC_PROTOCOLS exposes the available summaries.
"""

from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any

import numpy as np

History = Sequence[Mapping[str, Any]]


# ---------------------------------------------------------------------------
# Shared input handling
# ---------------------------------------------------------------------------


def _finite_values(values, name: str) -> np.ndarray:
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        for value in values
    ):
        raise ValueError(f"{name} must contain numeric scalar values.")
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite scalar values.")
    return result


def _evaluation_rows(history: History, key: str):
    rows = [row for row in history if row.get(key) is not None]
    scores = _finite_values([row[key] for row in rows], key)
    # Reject ambiguous ordering when update indices are supplied. Without them,
    # the caller's row order defines the evaluation sequence.
    steps = [row.get("global_step") for row in rows]
    if any(step is not None for step in steps):
        if any(
            isinstance(step, bool) or not isinstance(step, Integral) or step < 0
            for step in steps
        ):
            raise ValueError(
                "Every evaluation must have a nonnegative integer global_step."
            )
        if any(a >= b for a, b in zip(steps, steps[1:])):
            raise ValueError(
                "Evaluation global_step values must be strictly increasing."
            )
    return rows, scores


# ---------------------------------------------------------------------------
# RoboMimic / common reporting across baselines (ht_regression convention)
# ---------------------------------------------------------------------------
# mean_score = mean of each episode's maximum reward, produced by the runner.
# It equals success rate only for the corresponding binary-reward tasks.
# success_rate is the runner's separate env.is_success() aggregate; shaped reward
# experiments must not silently substitute it for DP's mean_score.
# All requested episodes are counted; historical first-batch truncation is not
# reproduced here. n_test, seeds and inference batch size belong to runner config.

ROBOMIMIC_SCORE_KEY = "test/mean_score"
ROBOMIMIC_SUCCESS_KEY = "test/success_rate"


def robomimic_metrics(
    history: History, *, key: str = ROBOMIMIC_SCORE_KEY
) -> dict[str, float | int]:
    """Best, last, and trailing-ten mean for a metric where larger is better.

    ``last10_mean`` averages evaluation scores, not the final ten trajectories.
    Before ten evaluations, average the available scores and report last10_count.
    This trailing window is a ht_regression convention, not DP's max_k_window statistic.
    """
    _, scores = _evaluation_rows(history, key)
    if not len(scores):
        return {}
    name = key.replace("/", "_")
    return {
        f"best/{name}": float(scores.max()),
        f"last/{name}": float(scores[-1]),
        f"last10_mean/{name}": float(scores[-10:].mean()),
        f"num_evaluations/{name}": len(scores),
        f"last10_count/{name}": min(10, len(scores)),
    }


# ---------------------------------------------------------------------------
# RoboMimic / Diffusion Policy (official reporting, default window sizes = 10)
# ---------------------------------------------------------------------------
# Source: real-stanford/diffusion_policy, revision
# 5ba07ac6661db573af695b419a7947ecb704690f, multirun_metrics.py::compute_metrics.
# Adapted under MIT; see ../policies/LICENSE.diffusion_policy.
#
# max: maximum evaluation score, independently selected within each training seed.
# k_around_max: upstream's exact slice around the FIRST maximum; near a boundary
#              it may contain fewer than ten evaluations, and is not recentered.
# max_k_window: maximum centered size-ten mean, with nearest-value edge padding
#               (scipy.ndimage.uniform_filter1d, size=10, mode="nearest").
# last: final evaluation score. None of these is the trailing-ten mean above.
# *_train_loss / *_val_loss: scores selected by aligned epoch-average losses.
# The current trainer's train/loss is an update loss, so it is NOT automatically
# treated as DP's epoch-average train_loss. Callers must supply the aligned losses.


def _aligned_losses(rows, key):
    values = [row.get(key) for row in rows]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(
            f"{key} must be supplied for every evaluation or omitted entirely."
        )
    return _finite_values(values, key)


def robomimic_diffusion_policy_metrics(
    history: History, *, key: str = ROBOMIMIC_SCORE_KEY
) -> dict[str, float | int]:
    """DP's per-run summary, retaining official metric names and edge semantics.

    Loss-selection metrics are emitted only when the corresponding train_loss or
    val_loss is supplied for every evaluation. metric_global_step is emitted only
    when actual update indices are available; row positions are not invented steps.
    Empty evaluation histories return an empty dictionary.
    """
    rows, scores = _evaluation_rows(history, key)
    if not len(scores):
        return {}
    name = key.replace("/", "_")
    peak = int(np.argmax(scores))
    end = min(peak + 5, len(scores))
    start = max(end - 10, 0)
    # For an even size-ten kernel, SciPy's default origin spans offsets [-5, 4].
    padded = np.pad(scores, (5, 4), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 10).mean(axis=-1)
    result = {
        f"max/{name}": float(scores.max()),
        f"k_around_max/{name}": float(scores[start:end].mean()),
        f"max_k_window/{name}": float(windows.max()),
        f"last/{name}": float(scores[-1]),
    }
    for loss_key in ("train_loss", "val_loss"):
        losses = _aligned_losses(rows, loss_key)
        if losses is not None:
            result[f"min_{loss_key}/{name}"] = float(scores[np.argmin(losses)])
            # Match upstream NumPy argsort's default tie behavior.
            result[f"k_min_{loss_key}/{name}"] = float(
                scores[np.argsort(losses)[:10]].mean()
            )
    if rows[-1].get("global_step") is not None:
        result[f"metric_global_step/{name}"] = int(rows[-1]["global_step"])
    return result


# ---------------------------------------------------------------------------
# Protocol lookup: benchmark / baseline
# Add new benchmark/baseline blocks with verified definitions above, then register
# them here. Future baselines do not implicitly inherit DP's reporting convention.
# ---------------------------------------------------------------------------

METRIC_PROTOCOLS = {
    "robomimic/common": robomimic_metrics,
    "robomimic/diffusion_policy": robomimic_diffusion_policy_metrics,
}
