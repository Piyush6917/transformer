"""Utilities shared by image-model training and generation scripts."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import Tensor


def save_image_grid(images: Tensor, output_path: Path, columns: int = 4) -> None:
    """Save images in ``[-1, 1]`` as a PNG grid without torchvision."""
    if images.ndim != 4:
        raise ValueError("images must have shape (batch, channels, height, width)")
    if images.shape[0] == 0:
        raise ValueError("at least one image is required")
    if images.shape[1] not in (1, 3):
        raise ValueError("images must have one or three channels")
    if columns <= 0:
        raise ValueError("columns must be positive")

    count, channels, height, width = images.shape
    rows = (count + columns - 1) // columns
    pixels = (
        images.detach()
        .clamp(-1, 1)
        .add(1)
        .mul(127.5)
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .contiguous()
    )
    mode = "L" if channels == 1 else "RGB"
    canvas = Image.new(mode, (columns * width, rows * height), color=0)
    for index in range(count):
        data = pixels[index].numpy().tobytes()
        if channels == 1:
            data = pixels[index, :, :, 0].numpy().tobytes()
        image = Image.frombytes(mode, (width, height), data)
        canvas.paste(image, ((index % columns) * width, (index // columns) * height))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
