# HT-Loss

HT-Loss is a heteroscedastic Student-t regression loss for PyTorch. The `ht_loss` package depends only on PyTorch and can be installed independently of `ht_regression`.

## Installation

From the repository root, install into your Python environment:

```bash
uv pip install ./ht_loss
# or: pip install ./ht_loss
```

## Usage

```python
from ht_loss import HTLoss

criterion = HTLoss()
prediction, raw_scale = model(observations)
loss = criterion(prediction, target, raw_scale)
loss.backward()
```

The functional interface uses the same arguments:

```python
from ht_loss import ht_loss

loss = ht_loss(prediction, target, raw_scale)
```

`prediction` and `target` have shape `[B, ...]`, such as `[B, T, D]` for action chunks. `raw_scale` contains one unconstrained scalar per sample, shaped `[B, 1]` or `[B]`.

### Choosing `nu`

Empirically, **`nu = 4d` is a good starting point**, where `d` is the number of predicted scalar values per sample. For action chunks shaped `[B, T, D]`, `d = T * D`; the batch size is excluded. For example, an 8-step chunk with 7 action dimensions gives `d = 56` and `nu = 224`.

```python
criterion = HTLoss()        # Automatically uses nu = 4d.
criterion = HTLoss(nu=224)  # Or specify a fixed value.
```

With a mask, the automatic choice uses each sample's valid dimension, computed as the sum of its expanded mask weights.

### Adding a scale head

Attach an action head and a scale head in parallel to the backbone's shared features. For sequence features shaped `[B, T, F]`, pool over time before the scale head to produce one scalar for the entire action chunk:

```python
from torch import nn


class ActionHeads(nn.Module):
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.action_head = nn.Linear(feature_dim, action_dim)
        self.scale_head = nn.Linear(feature_dim, 1)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)

    def forward(self, features):  # [B, T, F]
        prediction = self.action_head(features)  # [B, T, D]
        raw_scale = self.scale_head(features.mean(dim=1))  # [B, 1]
        return prediction, raw_scale
```

For features shaped `[B, F]`, feed them directly into both heads. Keep the features attached to the computation graph so both branches can train the backbone. Inference only needs the action head.

Return `raw_scale` directly: the loss computes `sigma = softplus(raw_scale + scale_bias) + min_scale`. With the zero-initialized scale head above, `scale_bias` can be set from the initial residual RMS. The default `scale_bias=0` does not perform calibration.

## Options

| Argument | Default | Meaning |
| --- | --- | --- |
| `nu` | `None` | Fixed Student-t degrees of freedom; `None` uses four times each sample's valid dimension. |
| `scale_bias` | `0.0` | Bias applied before softplus. |
| `min_scale` | `1e-3` | Positive floor added after softplus. |
| `reduction` | `"mean"` | `"mean"`, `"sum"`, or `"none"`. |
| `mask` | `None` | Nonnegative weights broadcastable to the target shape, passed when computing the loss. |
| `validate_values` | `True` | Set to `False` for trusted inputs with `torch.compile`. |

All non-batch dimensions form one multivariate event. For each sample, the loss is

```text
d = number of valid elements
S = sum of squared residuals over valid elements
L = 0.5 * (nu + d) * log1p(S / (nu * sigma²)) + d * log(sigma)
```

With a mask, both `d` and `S` use the expanded weights. `mean` returns `sum(L) / sum(d)`, `sum` returns `sum(L)`, and `none` returns one joint loss per sample with shape `[B]`. Terms independent of the predictions and scale are omitted.

Half and bfloat16 inputs use float32 loss arithmetic. Float64 inputs retain their precision.
