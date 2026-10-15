"""Prepared Cosmos inputs; native processors own images, text and normalization."""

from copy import copy
from dataclasses import dataclass, replace
from typing import Any

from torch import Tensor


@dataclass(frozen=True)
class CosmosCondition:
    """Native packed video/text context with action layout metadata.

    Video tokens and their noise levels are prepared externally. Action tokens
    and action times are replaced on every policy call, without modifying this
    pack. Memory is an optional native inference cache; omit it in training.
    """

    packed_sequence: Any
    memory: Any = None
    video_temporal_causal: bool | None = None

    def with_actions(self, sample: Tensor, native_time: Tensor) -> "CosmosCondition":
        """Replace action inputs while retaining video/text tensors and gradients."""
        pack = copy(self.packed_sequence)
        pack.action = copy(pack.action)
        pack.action.tokens = list(sample.unbind(0))
        # Native packs hold one timestep per action token, not per example.
        pack.action.timesteps = (
            native_time.repeat_interleave(sample.shape[1])
            .reshape_as(pack.action.timesteps)
            .to(device=pack.action.timesteps.device, dtype=pack.action.timesteps.dtype)
        )
        return replace(self, packed_sequence=pack)
