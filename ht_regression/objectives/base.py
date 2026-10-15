"""Objective interface; no dependency on any concrete policy family."""

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor, nn

from ..policies.base import BasePolicy, SampleSpec


class BaseObjective(nn.Module, ABC):
    @abstractmethod
    def compute_loss(
        self,
        policy: BasePolicy,
        target: Tensor,
        condition: Any,
        *,
        loss_mask: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """Compute a training loss on encoded inputs; preserve encoder gradients."""

    @abstractmethod
    def sample(
        self,
        policy: BasePolicy,
        condition: Any,
        *,
        sample_spec: SampleSpec,
        **kwargs,
    ) -> Tensor:
        """Generate a model-space sample for policy.decode_prediction()."""
