"""Tensor boundary between the GR00T processor and the objective-free policy.

Inputs are already tokenized, normalized and padded by a GR00T processor. This
module does not tokenize text, augment images or fit normalization statistics.
Integer token/embodiment IDs and boolean attention masks must stay integer/bool.
"""

from collections.abc import Mapping
from typing import NamedTuple

from torch import Tensor


class GR00TCondition(NamedTuple):
    backbone_features: Tensor
    state_features: Tensor
    embodiment_id: Tensor
    backbone_attention_mask: Tensor
    image_mask: Tensor | None = None


def model_inputs(value, *, device, dtype):
    """Move a prepared tensor tree without converting IDs or mutating inputs."""
    if isinstance(value, Tensor):
        return value.to(
            device=device, dtype=dtype if value.is_floating_point() else None
        )
    if isinstance(value, Mapping):
        return {
            key: model_inputs(item, device=device, dtype=dtype)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return type(value)(
            model_inputs(item, device=device, dtype=dtype) for item in value
        )
    return value
