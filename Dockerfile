FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FLASK_HOST=0.0.0.0

WORKDIR /app
RUN pip install --no-cache-dir torch PyMuPDF Pillow Flask diffusers transformers accelerate safetensors

COPY app.py chat.py model.py tokenizer.py generate_from_prompt.py ./
COPY templates ./templates
COPY static ./static
COPY pdf ./pdf
COPY generated ./generated

EXPOSE 5000
CMD ["python", "app.py"]