"""DP crop convention: random per frame in training, center crop in evaluation.

Reference: real-stanford/diffusion_policy, revision 5ba07ac (MIT).
See ../../LICENSE.diffusion_policy. Inputs remain uint8 until policy encoding.
"""

import torch
from torch import nn


class ImageCrop(nn.Module):
    def __init__(self, image_shape, crop_shape=None):
        super().__init__()
        self.image_shape = tuple(image_shape)
        self.crop_shape = tuple(crop_shape or image_shape[-2:])
        if len(self.image_shape) != 3 or self.image_shape[0] != 3:
            raise ValueError("Images must have shape (3, H, W).")
        if len(self.crop_shape) != 2 or any(
            type(v) is not int or not 1 <= v <= size
            for v, size in zip(self.crop_shape, self.image_shape[-2:])
        ):
            raise ValueError("crop_shape must fit inside the image.")
        # RoboMimic divides RGB bytes by 255 in float32 on the CPU, then DP
        # applies x * 2 - 1. GPU division by 127.5 can round differently; TF32
        # convolutions amplify those one-ULP changes. This fixed table preserves
        # the reference values while keeping observation transport in uint8.
        values = torch.arange(256, dtype=torch.float32, device="cpu")
        self.register_buffer(
            "_rgb_values", values.div(255).mul(2).sub(1), persistent=False
        )

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != self.image_shape:
            raise ValueError("Unexpected camera image shape.")
        if images.dtype != torch.uint8:
            raise ValueError("Camera input must be uint8 RGB in [0, 255].")
        n, _, h, w = images.shape
        ch, cw = self.crop_shape
        normalized = self._rgb_values[images.long()].contiguous()
        if not self.training:
            # Match DP's NCHW center-crop view, including its strides. The old
            # NHWC gather selected different TF32 convolution kernels.
            y, x = round((h - ch) / 2), round((w - cw) / 2)
            return normalized[:, :, y : y + ch, x : x + cw]
        # Match DP's floor(rand * (size-crop)), including its exclusive edge.
        y = (torch.rand(n, device=images.device) * (h - ch)).long()
        x = (torch.rand(n, device=images.device) * (w - cw)).long()
        rows = y[:, None, None] + torch.arange(ch, device=images.device)[None, :, None]
        cols = x[:, None, None] + torch.arange(cw, device=images.device)[None, None, :]
        cropped = normalized.permute(0, 2, 3, 1)[
            torch.arange(n, device=images.device)[:, None, None], rows, cols
        ]
        return cropped.permute(0, 3, 1, 2).contiguous()
