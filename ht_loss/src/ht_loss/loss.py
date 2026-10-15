"""PyTorch module interface for the HT loss."""

from torch import Tensor, nn

from .functional import _validate_options, ht_loss


class HTLoss(nn.Module):
    """Heteroscedastic Student-t loss with a learned scalar scale per sample.

    Call ``criterion(input, target, raw_scale, mask=...)``. See :func:`ht_loss`
    for tensor shapes, scale parameterization, masks and reduction semantics.

    Example::

        criterion = HTLoss(nu=200)
        prediction, raw_scale = model(observations)
        loss = criterion(prediction, target, raw_scale)
        loss.backward()
    """

    def __init__(
        self,
        nu: float | None = None,
        *,
        scale_bias: float = 0.0,
        min_scale: float = 1e-3,
        reduction: str = "mean",
        validate_values: bool = True,
    ):
        super().__init__()
        _validate_options(nu, scale_bias, min_scale, reduction)
        self.nu = None if nu is None else float(nu)
        self.scale_bias = float(scale_bias)
        self.min_scale = float(min_scale)
        self.reduction = reduction
        self.validate_values = validate_values

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        raw_scale: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> Tensor:
        return ht_loss(
            input,
            target,
            raw_scale,
            nu=self.nu,
            scale_bias=self.scale_bias,
            min_scale=self.min_scale,
            reduction=self.reduction,
            mask=mask,
            validate_values=self.validate_values,
        )

    def extra_repr(self) -> str:
        return (
            f"nu={self.nu}, scale_bias={self.scale_bias}, min_scale={self.min_scale}, "
            f"reduction={self.reduction!r}"
        )
