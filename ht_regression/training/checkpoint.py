"""Native training checkpoints and RNG isolation (not upstream DP import).

Checkpoints restore into already constructed, compatible components. Files use
torch/pickle to store Python and NumPy RNG state: only load trusted checkpoints.
"""

import os
import random
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

FORMAT_VERSION = 1
_REQUIRED = {
    "pipeline",
    "optimizer",
    "lr_scheduler",
    "scaler",
    "ema",
    "progress",
    "config",
    "rng",
    "data",
    "components",
    "metadata",
}


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if (
            not torch.cuda.is_available()
            or len(state["cuda"]) != torch.cuda.device_count()
        ):
            raise ValueError(
                "Exact RNG restoration requires the original CUDA topology."
            )
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


@contextmanager
def isolated_rng(seed: int):
    """Validation/evaluation must not consume the training RNG stream."""
    state = capture_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        if torch.cuda.is_initialized():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        restore_rng_state(state)


def _validate(state: dict) -> None:
    if not isinstance(state, dict) or state.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported native ht_regression training checkpoint format.")
    missing = _REQUIRED - state.keys()
    if missing:
        raise ValueError(f"Incomplete training checkpoint: missing {sorted(missing)}")


def save_checkpoint(path: str | Path, state: dict) -> Path:
    """Atomically replace a checkpoint; a failed write leaves the old file intact."""
    payload = dict(state, format_version=FORMAT_VERSION)
    _validate(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".tmp", delete=False
        ) as f:
            temporary = Path(f.name)
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def load_checkpoint(path: str | Path) -> dict:
    """Load a trusted native checkpoint on CPU before restoring its components."""
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    _validate(state)
    return state
