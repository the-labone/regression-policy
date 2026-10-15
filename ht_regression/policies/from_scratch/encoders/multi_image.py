"""Independent DP-style camera branches plus normalized proprioception."""

from collections.abc import Mapping

import torch
from torch import nn

from ..backbones.vision.resnet import ResNetImageEncoder
from .transforms import ImageCrop


class MultiImageEncoder(nn.Module):
    def __init__(
        self,
        camera_shapes,
        proprio_dim,
        feature_dim=64,
        num_keypoints=32,
        crop_shape=(76, 76),
        weight_decay=1e-6,
        proprio_keys=None,
    ):
        super().__init__()
        self.camera_shapes = {key: tuple(value) for key, value in camera_shapes.items()}
        if not self.camera_shapes or proprio_dim < 1 or "proprio" in self.camera_shapes:
            raise ValueError(
                "Require cameras and positive proprio_dim; proprio is a reserved key."
            )
        self.proprio_dim = proprio_dim
        self.proprio_keys = tuple(proprio_keys) if proprio_keys is not None else None
        self.feature_dim = feature_dim
        self.output_dim = proprio_dim + len(self.camera_shapes) * feature_dim
        self.weight_decay = weight_decay
        if weight_decay < 0:
            raise ValueError("Encoder weight_decay must be nonnegative.")
        self.crops = nn.ModuleDict(
            {
                key: ImageCrop(shape, crop_shape)
                for key, shape in self.camera_shapes.items()
            }
        )
        self.cameras = nn.ModuleDict(
            {
                key: ResNetImageEncoder(
                    (3, *self.crops[key].crop_shape), feature_dim, num_keypoints
                )
                for key in self.camera_shapes
            }
        )

    def get_extra_state(self):
        return {
            "version": 1,
            "camera_shapes": self.camera_shapes,
            "camera_order": list(self.camera_shapes),
            "proprio_dim": self.proprio_dim,
            "proprio_keys": self.proprio_keys,
            "crops": {k: v.crop_shape for k, v in self.crops.items()},
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError(
                "Visual encoder camera/crop configuration does not match checkpoint."
            )

    def forward(self, observation, normalizer, n_obs_steps):
        # Keep key validation traceable on the cluster's PyTorch 2.4: Dynamo
        # cannot capture set construction/comparison from dictionary keys.
        if (
            not isinstance(observation, Mapping)
            or len(observation) != len(self.camera_shapes) + 1
            or "proprio" not in observation
            or any(key not in observation for key in self.camera_shapes)
        ):
            raise ValueError(
                "Visual observations must contain exactly the configured cameras and proprio."
            )
        proprio = observation["proprio"]
        if (
            proprio.ndim != 3
            or proprio.shape[1] < n_obs_steps
            or proprio.shape[2] != self.proprio_dim
            or not proprio.is_floating_point()
        ):
            raise ValueError(
                "proprio must have shape (B, T >= n_obs_steps, proprio_dim)."
            )
        batch = proprio.shape[0]
        features = []
        for key, shape in self.camera_shapes.items():
            image = observation[key]
            if (
                image.shape != (batch, proprio.shape[1], *shape)
                or image.device != proprio.device
            ):
                raise ValueError(
                    f"Camera {key} must align with proprioception in batch/time/device."
                )
            frames = image[:, :n_obs_steps].reshape(-1, *shape)
            feature = self.cameras[key](self.crops[key](frames))
            features.append(feature.reshape(batch, n_obs_steps, -1))
        features.append(normalizer.normalize_obs(proprio[:, :n_obs_steps]))
        return torch.cat(features, dim=-1)
