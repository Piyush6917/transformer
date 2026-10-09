"""Generate new images from a trained DCGAN checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from image_model import ImageGenerator
from image_utils import save_image_grid


def generate(
    checkpoint_path: Path,
    output_path: Path,
    *,
    count: int = 1,
    seed: int | None = None,
) -> None:
    """Generate ``count`` images and save them as one PNG grid."""
    if count <= 0:
        raise ValueError("count must be positive")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Train the model first with "
            "train_image_model.py."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["model_config"]
    if config.get("image_size") != ImageGenerator.image_size:
        raise ValueError(
            f"This script expects {ImageGenerator.image_size}x{ImageGenerator.image_size} images; "
            f"checkpoint uses {config.get('image_size')}x{config.get('image_size')}"
        )
    generator = ImageGenerator(
        latent_dimension=config["latent_dimension"],
        image_channels=config["image_channels"],
        base_channels=config["base_channels"],
    ).to(device)
    generator.load_state_dict(checkpoint["model_state"])
    generator.eval()

    if seed is not None:
        torch.manual_seed(seed)
    noise = torch.randn(count, config["latent_dimension"], device=device)
    with torch.no_grad():
        images = generator(noise)
    columns = min(4, count)
    save_image_grid(images, output_path, columns=columns)
    print(f"Saved {count} generated image(s) to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).with_name("checkpoints") / "image_dcgan.pt",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("generated") / "images.png"
    )
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    generate(
        arguments.checkpoint,
        arguments.output,
        count=arguments.count,
        seed=arguments.seed,
    )
