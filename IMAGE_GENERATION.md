# Image generation model

This project has two image-generation paths:

* The custom DCGAN learns from example images and samples 64×64 RGB images. It
  does not use text prompts.
* The pretrained text-to-image model accepts prompts such as “a red sports car
  driving through a city at sunset”. It uses general pretrained knowledge and
  does not train on the local car folder.

## Generate from a text prompt

Install the optional text-to-image dependencies:

```powershell
python -m pip install -r requirements-diffusion.txt
```

Generate an image from a prompt:

```powershell
python generate_from_prompt.py --prompt "a red car" --output generated/red_car.png
```

Or run the script without `--prompt` and enter the description when asked:

```powershell
python generate_from_prompt.py
```

The default model is [Segmind Tiny-SD](https://huggingface.co/segmind/tiny-sd).
Its weights are about 1 GB and download on first use. Generation runs on CUDA
when available and otherwise on CPU; CPU generation can take a while. The
model card lists its usage license and model details.

The current PyTorch Windows guide supports Python 3.10–3.14, which includes
this project's Python version. The project already has PyTorch installed. This
computer has no CUDA GPU, so generation will use CPU.

## Train the custom DCGAN

Put training images (JPG, PNG, BMP, or WebP) under `images/`. Images may also be
organized in subdirectories. The loader center-crops and resizes them to 64×64.

Train the custom model:

```powershell
python train_image_model.py --data-dir images --epochs 50 --batch-size 64
```

The training script saves periodic sample grids in `generated/training/` and
the trained generator in `checkpoints/image_dcgan.pt`. Sample from that model:

```powershell
python generate_image.py --count 1 --output generated/image.png
```

Use `--count 16` to save a grid of samples. Pass `--seed` to reproduce the
same random samples. Training a small GAN from scratch usually requires many
representative images and several epochs; the quality depends on the dataset.
