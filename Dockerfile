FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install packages used by the PDF QA and image-generation scripts.
RUN pip install --no-cache-dir torch PyMuPDF Pillow

# Copy the application, model scripts, extracted corpus, and source PDFs.
COPY model.py tokenizer.py chat.py train.py image_model.py image_utils.py ./
COPY train_image_model.py generate_image.py data.txt ./
COPY checkpoints ./checkpoints
COPY pdf ./pdf

# Start interactive PDF question answering.
CMD ["python", "chat.py"]
