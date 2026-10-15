"""Shared device transfer and batch-size inference for nested training batches."""

from collections.abc import Mapping

import torch


def move_to_device(value, device: torch.device):
    """Move tensors, preserve sequence structure, and leave metadata unchanged."""
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        items = [move_to_device(item, device) for item in value]
        return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def infer_batch_size(batch) -> int | None:
    """Find the first tensor's leading dimension, ignoring scalar metadata."""
    if isinstance(batch, torch.Tensor):
        return batch.shape[0] if batch.ndim else None
    values = batch.values() if isinstance(batch, Mapping) else batch
    if isinstance(values, (str, bytes)):
        return None
    try:
        iterator = iter(values)
    except TypeError:
        return None
    for value in iterator:
        size = infer_batch_size(value)
        if size is not None:
            return size
    return None
