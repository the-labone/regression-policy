"""HT configuration layered on the pinned official GR00T configuration."""

from dataclasses import dataclass

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config


@dataclass(init=False)
class GR00TConfig(Gr00tN1d7Config):
    # Keep the released model_type and parameter names for old checkpoints.
    loss_type: str = "flow"
    ht_sbias: float = 3.0
    ht_df: float = 2.0
    ht_mvt: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.loss_type not in ("flow", "hetero_t", "mse", "l1"):
            raise ValueError(f"Unsupported GR00T loss_type: {self.loss_type!r}")
        if self.ht_df <= 0:
            raise ValueError("ht_df must be positive.")
