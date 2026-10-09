"""Train a DCGAN on images in a local directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from image_model import ImageDiscriminator, ImageGenerator, initialize_dcgan_weights
from image_utils import save_image_grid


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_SIZE = 64


class ImageFolderDataset(Dataset[Tensor]):
    """Load and normalize image files from a directory and its subdirectories."""

    def __init__(self, image_directory: Path) -> None:
        self.image_paths = sorted(
            path
            for path in image_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise FileNotFoundError(
                f"No images found in {image_directory}; supported formats: "
                f"{', '.join(sorted(IMAGE_EXTENSIONS))}"
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tensor:
        with Image.open(self.image_paths[index]) as source:
            image = ImageOps.fit(
                source.convert("RGB"),
                (IMAGE_SIZE, IMAGE_SIZE),
                method=Image.Resampling.BICUBIC,
            )
            pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        pixels = pixels.view(IMAGE_SIZE, IMAGE_SIZE, 3).permute(2, 0, 1).float()
        return pixels.div(127.5).sub(1.0)


def train(
    image_directory: Path,
    checkpoint_path: Path,
    samples_directory: Path,
    *,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.0002,
    latent_dimension: int = 100,
    base_channels: int = 64,
    seed: int = 42,
) -> None:
    """Train a generator and discriminator, saving preview images and a checkpoint."""
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    if latent_dimension <= 0 or base_channels <= 0:
        raise ValueError("latent_dimension and base_channels must be positive")

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageFolderDataset(image_directory)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    generator = ImageGenerator(latent_dimension, base_channels=base_channels).to(device)
    discriminator = ImageDiscriminator(base_channels=base_channels).to(device)
    generator.apply(initialize_dcgan_weights)
    discriminator.apply(initialize_dcgan_weights)

    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999)
    )
    loss_function = nn.BCEWithLogitsLoss()
    fixed_noise = torch.randn(16, latent_dimension, device=device)

    print(f"Training on {len(dataset)} images with {device}.")
    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        total_generator_loss = 0.0
        total_discriminator_loss = 0.0

        for real_images in loader:
            real_images = real_images.to(device)
            batch_count = real_images.shape[0]

            discriminator_optimizer.zero_grad(set_to_none=True)
            real_logits = discriminator(real_images)
            real_loss = loss_function(real_logits, torch.ones_like(real_logits))

            noise = torch.randn(batch_count, latent_dimension, device=device)
            fake_images = generator(noise)
            fake_logits = discriminator(fake_images.detach())
            fake_loss = loss_function(fake_logits, torch.zeros_like(fake_logits))
            discriminator_loss = real_loss + fake_loss
            discriminator_loss.backward()
            discriminator_optimizer.step()

            generator_optimizer.zero_grad(set_to_none=True)
            generated_logits = discriminator(fake_images)
            generator_loss = loss_function(
                generated_logits, torch.ones_like(generated_logits)
            )
            generator_loss.backward()
            generator_optimizer.step()

            total_discriminator_loss += discriminator_loss.item()
            total_generator_loss += generator_loss.item()

        generator.eval()
        with torch.no_grad():
            preview_images = generator(fixed_noise)
        preview_path = samples_directory / f"epoch_{epoch:04d}.png"
        save_image_grid(preview_images, preview_path, columns=4)
        steps = len(loader)
        print(
            f"Epoch {epoch:3d}/{epochs} - "
            f"D loss: {total_discriminator_loss / steps:.4f} - "
            f"G loss: {total_generator_loss / steps:.4f} - "
            f"preview: {preview_path}"
        )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": generator.state_dict(),
            "model_config": {
                "latent_dimension": latent_dimension,
                "image_channels": 3,
                "base_channels": base_channels,
                "image_size": IMAGE_SIZE,
            },
        },
        checkpoint_path,
    )
    print(f"Saved generator checkpoint to {checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).with_name("images"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).with_name("checkpoints") / "image_dcgan.pt",
    )
    parser.add_argument(
        "--samples-dir", type=Path, default=Path(__file__).with_name("generated") / "training"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--latent-dimension", type=int, default=100)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train(
        arguments.data_dir,
        arguments.checkpoint,
        arguments.samples_dir,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        latent_dimension=arguments.latent_dimension,
        base_channels=arguments.base_channels,
        seed=arguments.seed,
    )
