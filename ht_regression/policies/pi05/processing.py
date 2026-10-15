# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
"""Prepared image/token inputs and attention masks for the PI0.5 adapter.

Adapted from LeRobot PI0.5; see LICENSE.pi05 for the upstream license.
"""

from typing import Any, NamedTuple

import torch
from torch import Tensor


class PI05Condition(NamedTuple):
    embeddings: Tensor
    padding_mask: Tensor
    attention_mask: Tensor
    past_key_values: Any = None


def attention_mask(padding: Tensor, blocks: Tensor) -> Tensor:
    """OpenPI block-causal attention, including both query and key padding."""
    groups = torch.cumsum(blocks, dim=1)
    visible = groups[:, None, :] <= groups[:, :, None]
    return visible & padding[:, None, :] & padding[:, :, None]


def additive_mask(mask: Tensor) -> Tensor:
    # Same fp32 sentinel as LeRobot's prepare_attention_masks_4d.
    return torch.where(mask[:, None], 0.0, -2.3819763e38)
