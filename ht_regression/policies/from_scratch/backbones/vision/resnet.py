"""ResNet18 + spatial keypoints, matching DP's RoboMimic visual architecture.

Uses torchvision's ResNet implementation. Spatial pooling computes expected
normalized (x, y) coordinates, followed by a linear feature map and ReLU.
"""

import torch
from torch import nn


class SpatialSoftmax(nn.Module):
    def __init__(self, channels, height, width, num_keypoints=32):
        super().__init__()
        self.projection = nn.Conv2d(channels, num_keypoints, 1)
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, dtype=torch.float64).float(),
            torch.linspace(-1, 1, width, dtype=torch.float64).float(),
            indexing="ij",
        )
        self.register_buffer(
            "coordinates", torch.stack((x.flatten(), y.flatten()), dim=-1)
        )

    def forward(self, features):
        projected = self.projection(features)
        logits = projected.reshape(-1, projected.shape[-2] * projected.shape[-1])
        # Keep softmax/expected coordinates stable under autocast.
        attention = logits.float().softmax(-1)
        # DP multiplies and sums each coordinate separately. A mathematically
        # equivalent GEMM changes rounding, which can change a closed-loop run.
        coordinates = self.coordinates.float()
        x = (coordinates[:, 0].reshape(1, -1) * attention).sum(-1, keepdim=True)
        y = (coordinates[:, 1].reshape(1, -1) * attention).sum(-1, keepdim=True)
        return torch.cat((x, y), dim=1).reshape(features.shape[0], -1)


class ResNetImageEncoder(nn.Module):
    def __init__(self, image_shape, feature_dim=64, num_keypoints=32):
        super().__init__()
        from torchvision.models import resnet18

        if feature_dim < 1 or num_keypoints < 1:
            raise ValueError("Visual feature dimensions must be positive.")
        net = resnet18(weights=None)
        self.trunk = nn.Sequential(*list(net.children())[:-2])
        self._replace_batch_norm(self.trunk)
        height, width = ((v + 31) // 32 for v in image_shape[-2:])
        self.pool = SpatialSoftmax(512, height, width, num_keypoints)
        self.output = nn.Linear(2 * num_keypoints, feature_dim)

    @classmethod
    def _replace_batch_norm(cls, module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                setattr(
                    module,
                    name,
                    nn.GroupNorm(child.num_features // 16, child.num_features),
                )
            else:
                cls._replace_batch_norm(child)

    def forward(self, image):
        return torch.relu(self.output(self.pool(self.trunk(image))))
