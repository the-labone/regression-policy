"""Objective-independent policy contract and model-space tensor descriptions."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
from torch import Tensor, nn

Timestep = Tensor | float | int


@dataclass(frozen=True)
class SampleSpec:
    """Shape (including batch), device and dtype of the generated representation.

    Shapes may describe action sequences, images or video latents. Conditioning
    tensors can have different shapes, devices and dtypes, owned by the policy.
    """

    shape: tuple[int, ...]
    device: torch.device
    dtype: torch.dtype

    def __post_init__(self):
        if not self.shape or any(type(n) is not int or n < 1 for n in self.shape):
            raise ValueError("sample shape must contain positive dimensions.")
        if not self.dtype.is_floating_point:
            raise ValueError("sample dtype must be floating point.")
        object.__setattr__(self, "device", torch.device(self.device))

    def validate(
        self, value: Tensor, name: str = "sample", *, check_values: bool = True
    ) -> None:
        """Check metadata; optionally inspect values (synchronizes CUDA tensors)."""
        if not isinstance(value, Tensor) or tuple(value.shape) != self.shape:
            raise ValueError(f"{name} must have shape {self.shape}.")
        if value.dtype != self.dtype or value.device != self.device:
            raise ValueError(f"{name} must match the sample dtype and device.")
        if check_values and not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite.")


class PolicyTarget(NamedTuple):
    """Clean model-space training sample and optional broadcastable loss weights.

    ``loss_mask`` is nonnegative and weights the loss, not the noise process or
    conditioning. None includes every element (including DP's edge padding).
    A named tuple keeps this immutable without a frozen-dataclass constructor's
    object.__setattr__, which older Torch compilers cannot trace.
    """

    sample: Tensor
    loss_mask: Tensor | None = None


class BasePolicy(nn.Module, ABC):
    """Model family interface; objectives own corruption, targets and sampling.

    Conditions are opaque to objectives and may contain tokens, masks, states or
    caches. Encoding must preserve gradients unless explicitly frozen. ``time``
    and the meaning of ``forward``'s output are supplied by the objective; any
    architecture-specific time encoding belongs to the policy/model adapter.
    """

    @abstractmethod
    def encode_condition(self, observation: Any) -> Any:
        """Encode raw observations; accept a training batch as well as eval input."""

    @abstractmethod
    def encode_target(self, batch: Mapping[str, Any]) -> PolicyTarget:
        """Encode clean targets and their loss weights, without adding noise."""

    @abstractmethod
    def sample_spec(self, condition: Any) -> SampleSpec:
        """Describe a generated sample without requiring ground-truth targets."""

    @abstractmethod
    def forward(self, sample: Tensor, time: Timestep, condition: Any) -> Tensor:
        """One model prediction, with the same shape as sample; no loss or solver."""

    def forward_with_scale(
        self, sample: Tensor, time: Timestep, condition: Any
    ) -> tuple[Tensor, Tensor]:
        """Optional joint readout for likelihood objectives: mean and raw scale.

        Mean has sample.shape. Raw scale is one scalar per chunk (B x 1).
        Both preserve gradients into the model. The objective owns the positive
        transform and likelihood.
        Ordinary forward still returns only the mean for deployment.
        """
        raise NotImplementedError("This policy does not provide a learned scale head.")

    @abstractmethod
    def decode_prediction(self, prediction: Tensor) -> dict[str, Tensor]:
        """Decode model output for consumers, including an execution action chunk."""

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def reset(self) -> None:
        """Clear rollout state if the policy owns any; stateless by default."""

    def get_optimizer_groups(self, weight_decay: float):
        """Default trainable group; model families can provide specialized groups."""
        return [
            {
                "params": [p for p in self.parameters() if p.requires_grad],
                "weight_decay": weight_decay,
            }
        ]
