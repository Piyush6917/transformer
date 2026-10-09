"""A small DCGAN for learning to generate 64x64 images."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ImageGenerator(nn.Module):
    """Turn a latent vector into an RGB image with values in ``[-1, 1]``."""

    image_size = 64

    def __init__(
        self,
        latent_dimension: int = 100,
        image_channels: int = 3,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        if latent_dimension <= 0 or image_channels <= 0 or base_channels <= 0:
            raise ValueError("latent_dimension, image_channels, and base_channels must be positive")

        self.latent_dimension = latent_dimension
        self.image_channels = image_channels
        self.base_channels = base_channels
        self.network = nn.Sequential(
            nn.ConvTranspose2d(latent_dimension, base_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels, image_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, latent_vectors: Tensor) -> Tensor:
        """Generate images from ``(batch, latent_dimension)`` vectors."""
        if latent_vectors.ndim != 2:
            raise ValueError("latent_vectors must have shape (batch, latent_dimension)")
        if latent_vectors.shape[1] != self.latent_dimension:
            raise ValueError(
                f"expected latent dimension {self.latent_dimension}, "
                f"got {latent_vectors.shape[1]}"
            )
        return self.network(latent_vectors[:, :, None, None])


class ImageDiscriminator(nn.Module):
    """Score whether a 64x64 image looks like it came from the dataset."""

    image_size = 64

    def __init__(self, image_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        if image_channels <= 0 or base_channels <= 0:
            raise ValueError("image_channels and base_channels must be positive")

        self.image_channels = image_channels
        self.base_channels = base_channels
        self.network = nn.Sequential(
            nn.Conv2d(image_channels, base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Return one real/fake logit for each ``(batch, channels, 64, 64)`` image."""
        if images.ndim != 4:
            raise ValueError("images must have shape (batch, channels, height, width)")
        if images.shape[1] != self.image_channels:
            raise ValueError(f"expected {self.image_channels} image channels, got {images.shape[1]}")
        if images.shape[2:] != (self.image_size, self.image_size):
            raise ValueError(f"images must be {self.image_size}x{self.image_size}")
        return self.network(images).flatten()


def initialize_dcgan_weights(module: nn.Module) -> None:
    """Apply the usual small-normal initialization used by DCGANs."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)
