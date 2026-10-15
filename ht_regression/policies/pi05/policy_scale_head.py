"""The historical PI0.5 HT scalar readout: pool raw token outputs first."""

from .policy import PI05Policy


class PI05PolicyWithScale(PI05Policy):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        if getattr(model, "sigma_out_proj", None) is None:
            raise ValueError("The checkpoint has no sigma_out_proj.")
        if getattr(model.config, "ht_sigma_pooling", "before_softplus") not in (
            "before_softplus",
            "mean_before_softplus",
        ):
            raise ValueError("Only the historical raw-pooling sigma head is supported.")

    def forward_with_scale(self, sample, time, condition):
        features = self.forward_features(sample, time, condition)
        mean = -self.model._apply_checkpoint(self.model.action_out_proj, features)
        raw_scale = self.model.sigma_out_proj(features).mean(dim=(1, 2))[:, None]
        return mean, raw_scale
