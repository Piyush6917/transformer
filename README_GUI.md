# Transformer Studio

A local browser interface for asking questions about PDFs and generating images from text prompts.

## Start the app

From the project folder in PowerShell:

    python -m pip install -r requirements-gui.txt
    python -m pip install -r requirements-diffusion.txt
    python app.py

Open http://127.0.0.1:5000 in your browser. The server listens only on your computer by default.

Put PDFs in the pdf folder or upload them from the PDF Assistant tab. Ask questions and the assistant will show the pages it used.

Choose Image Studio, type a prompt such as "a red car driving through a rainy neon city", and select Create image. The first generation downloads the pretrained model weights; later prompts reuse the model while the server is running. Image generation works on CPU or CUDA, though CPU can take several minutes.

Generated images are saved in generated. To select another Diffusers model, set IMAGE_MODEL before starting the app. For example:

    $env:IMAGE_MODEL = "segmind/tiny-sd"
    python app.py

To access the app from other devices on your network, set FLASK_HOST to 0.0.0.0 before launch. Only do this on a trusted network.

