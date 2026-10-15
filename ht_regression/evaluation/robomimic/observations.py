"""Batch state arrays or camera/proprioception dictionaries without losing dtype."""

import numpy as np
import torch


def stack_observations(values, *, concatenate=False):
    values = list(values)
    if isinstance(values[0], dict):
        return {
            key: stack_observations(
                [value[key] for value in values], concatenate=concatenate
            )
            for key in values[0]
        }
    return (np.concatenate if concatenate else np.stack)(values)


def observation_to_device(value, device, dtype):
    if isinstance(value, dict):
        return {
            key: observation_to_device(item, device, dtype)
            for key, item in value.items()
        }
    return torch.as_tensor(
        value, device=device, dtype=torch.uint8 if value.dtype == np.uint8 else dtype
    )
