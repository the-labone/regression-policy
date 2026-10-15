"""Historical Cosmos HT readout: a scalar head on detached padded predictions."""

import torch
from torch import nn

from .policy import CosmosPolicy


class CosmosPolicyWithScale(CosmosPolicy):
    def __init__(self, network, *, scale_head=None, **kwargs):
        super().__init__(network, **kwargs)
        if self.prediction_type != "clean_action":
            raise ValueError(
                "The historical Cosmos sigma head requires clean-action predictions."
            )
        native_head = getattr(network, "llm2action_ht_sigma_head", None)
        if scale_head is not None and native_head is not None:
            raise ValueError(
                "Supply a separate scale head only if the network has none."
            )
        head = native_head if native_head is not None else scale_head
        if head is None:
            head = nn.Linear(
                self.action_dim, 1, device=self.device, dtype=torch.float32
            )
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        if not isinstance(head, nn.Linear) or (head.in_features, head.out_features) != (
            self.action_dim,
            1,
        ):
            raise ValueError("Cosmos HT requires a linear action_dim-to-1 scale head.")
        # A native head is already registered under network; keep one state-dict path.
        self.scale_head = head if native_head is None else None

    def forward_with_scale(self, sample, time, condition):
        prediction = self(sample, time, condition)
        head = self.scale_head
        if head is None:
            head = self.network.llm2action_ht_sigma_head
        # Match the reference: all padded features enter the scalar head; only
        # the residual loss excludes padding. Pool raw outputs before softplus.
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            raw = torch.nn.functional.linear(
                prediction.float().detach(),
                head.weight.float(),
                None if head.bias is None else head.bias.float(),
            ).mean(dim=(1, 2))[:, None]
        return prediction, raw
