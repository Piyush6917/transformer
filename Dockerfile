FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install only the packages used by the PDF QA application.
RUN pip install --no-cache-dir torch PyMuPDF

# Copy the application, trained checkpoint, extracted corpus, and source PDFs.
COPY model.py tokenizer.py chat.py train.py data.txt ./
COPY checkpoints ./checkpoints
COPY pdf ./pdf

# Start interactive PDF question answering.
CMD ["python", "chat.py"]
