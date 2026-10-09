"""Generate an image from a text prompt with a pretrained text-to-image model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


DEFAULT_MODEL = "segmind/tiny-sd"


def generate_from_prompt(
    prompt: str,
    output_path: Path,
    *,
    model_id: str = DEFAULT_MODEL,
    steps: int = 25,
    guidance_scale: float = 7.5,
    seed: int = 42,
) -> None:
    """Generate and save one image guided by ``prompt``."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if guidance_scale < 0:
        raise ValueError("guidance_scale must not be negative")

    try:
        from diffusers import DiffusionPipeline
    except ImportError as error:
        raise RuntimeError(
            "Text-to-image dependencies are missing. Install them with "
            "`python -m pip install -r requirements-diffusion.txt`."
        ) from error

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading {model_id} on {device}. First use downloads the model weights.")
    pipeline = DiffusionPipeline.from_pretrained(model_id, dtype=dtype)
    pipeline.to(device)

    random_generator = torch.Generator(device=device).manual_seed(seed)
    print(f'Generating image for: "{prompt}"')
    with torch.inference_mode():
        result = pipeline(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=random_generator,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(output_path)
    print(f"Saved image to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", help="What image should be generated?")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("generated") / "prompt.png"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Diffusers model ID or local path")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    prompt = arguments.prompt or input("Describe the image to generate: ").strip()
    generate_from_prompt(
        prompt,
        arguments.output,
        model_id=arguments.model,
        steps=arguments.steps,
        guidance_scale=arguments.guidance_scale,
        seed=arguments.seed,
    )
