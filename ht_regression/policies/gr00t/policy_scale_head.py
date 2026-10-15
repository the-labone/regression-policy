"""GR00T's optional per-element raw scale readout; no likelihood math."""

from torch import nn

from .policy import GR00TPolicy


class GR00TPolicyWithScale(GR00TPolicy):
    """Share DiT features between action and upstream sigma decoders.

    The original GR00T HT path predicts (B,H,D) raw scales, then the objective
    applies softplus before masked pooling. It is NOT equivalent to the current
    RoboMimic HT scalar head. Expose raw elementwise values explicitly; an HT
    objective supporting this representation is required for training.
    """

    def __init__(self, action_head, *, sigma_decoder: nn.Module, **kwargs):
        super().__init__(action_head, **kwargs)
        self.sigma_decoder = sigma_decoder
        if not self.action_head.tune_projector:
            self.sigma_decoder.requires_grad_(False)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if (
            hasattr(self, "sigma_decoder")
            and mode
            and not self.action_head.tune_projector
        ):
            self.sigma_decoder.eval()
        return self

    def forward_with_elementwise_scale(self, sample, time, condition):
        features = self.action_head.forward_features(sample, time, condition)
        mean = self.action_head.decode_features(features, condition)
        raw_scale = self.sigma_decoder(features, condition.embodiment_id)[
            :, -self.horizon :
        ]
        if raw_scale.shape != mean.shape:
            raise ValueError(
                "sigma_decoder must return one raw scale per action element."
            )
        return mean, raw_scale

    def forward_with_scale(self, sample, time, condition):
        raise NotImplementedError(
            "GR00T has elementwise raw scales. Use an objective supporting "
            "forward_with_elementwise_scale and masked pooling after softplus; "
            "the existing scalar-head HTObjective is not equivalent."
        )
