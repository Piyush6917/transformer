"""Local Flask interface for PDF Q&A and prompt-based image generation."""
from __future__ import annotations
import os, threading, time, uuid
from functools import lru_cache
from pathlib import Path
import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from chat import answer_question, load_pdf_chunks

ROOT = Path(__file__).resolve().parent
PDF_DIR, GENERATED_DIR = ROOT / "pdf", ROOT / "generated"
PDF_DIR.mkdir(exist_ok=True)
GENERATED_DIR.mkdir(exist_ok=True)
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
MODEL_ID = os.environ.get("IMAGE_MODEL", "segmind/tiny-sd")
jobs: dict[str, dict] = {}
jobs_lock, generation_lock, chunks_lock = threading.Lock(), threading.Lock(), threading.Lock()
cached_chunks: list | None = None
cached_signature: tuple | None = None

def pdf_signature() -> tuple:
    return tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(PDF_DIR.glob("*.pdf")))

def get_chunks() -> list:
    global cached_chunks, cached_signature
    signature = pdf_signature()
    with chunks_lock:
        if cached_chunks is None or signature != cached_signature:
            cached_chunks = load_pdf_chunks(PDF_DIR)
            cached_signature = signature
        return cached_chunks

@lru_cache(maxsize=2)
def load_image_pipeline(model_id: str, device: str):
    from diffusers import DiffusionPipeline
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.to(device)
    return pipe

def run_generation(job_id: str, prompt: str, steps: int, guidance: float, seed: int) -> None:
    with jobs_lock:
        jobs[job_id].update(status="loading", message="Preparing the image model…")
    try:
        with generation_lock:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = load_image_pipeline(MODEL_ID, device)
            with jobs_lock:
                jobs[job_id].update(status="generating", message="Creating your image…")
            generator = torch.Generator(device=device).manual_seed(seed)
            with torch.inference_mode():
                result = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance, generator=generator)
            filename = job_id + ".png"
            result.images[0].save(GENERATED_DIR / filename)
        with jobs_lock:
            jobs[job_id].update(status="complete", message="Your image is ready.",
                                image_url="/generated/" + filename,
                                download_url="/generated/" + filename + "?download=1")
    except Exception as error:
        app.logger.exception("Image generation failed")
        with jobs_lock:
            jobs[job_id].update(status="error", message=str(error) or error.__class__.__name__)

@app.get("/")
def index():
    return render_template("index.html", model_name=MODEL_ID)

@app.get("/api/status")
def status():
    files = sorted(PDF_DIR.glob("*.pdf"))
    return jsonify(pdf_count=len(files), pdf_names=[p.name for p in files],
                   model=MODEL_ID, device="GPU" if torch.cuda.is_available() else "CPU")

@app.post("/api/upload")
def upload_pdf():
    uploads = request.files.getlist("files")
    if not uploads or all(not f.filename for f in uploads):
        return jsonify(error="Choose one or more PDF files first."), 400
    saved = []
    for item in uploads:
        name = secure_filename(item.filename or "")
        if not name.lower().endswith(".pdf"):
            return jsonify(error="Only PDF files are supported."), 400
        item.save(PDF_DIR / name)
        saved.append(name)
    global cached_chunks, cached_signature
    with chunks_lock:
        cached_chunks = cached_signature = None
    return jsonify(saved=saved, pdf_count=len(list(PDF_DIR.glob("*.pdf"))))

@app.post("/api/chat")
def chat():
    body = request.get_json(silent=True) or {}
    question = str(body.get("question", "")).strip()
    if not question:
        return jsonify(error="Type a question to get started."), 400
    if len(question) > 2000:
        return jsonify(error="Keep your question under 2,000 characters."), 400
    try:
        answer, retrieved = answer_question(question, get_chunks())
    except FileNotFoundError:
        return jsonify(error="Upload a PDF first so I have something to reference."), 400
    except Exception as error:
        app.logger.exception("PDF question failed")
        return jsonify(error=str(error)), 500
    sources, seen = [], set()
    for score, chunk in retrieved:
        key = (chunk.document_name, chunk.page_number)
        if key not in seen:
            seen.add(key)
            sources.append({"document": chunk.document_name, "page": chunk.page_number, "score": round(score, 2)})
    return jsonify(answer=answer, sources=sources)

@app.post("/api/generate")
def generate():
    body = request.get_json(silent=True) or {}
    prompt = str(body.get("prompt", "")).strip()
    if not prompt:
        return jsonify(error="Describe the image you want to create."), 400
    if len(prompt) > 1000:
        return jsonify(error="Keep your prompt under 1,000 characters."), 400
    try:
        steps = max(5, min(50, int(body.get("steps", 25))))
        guidance = max(1.0, min(15.0, float(body.get("guidance", 7.5))))
        seed = int(body.get("seed", 42))
    except (TypeError, ValueError):
        return jsonify(error="Steps, guidance, and seed must be valid numbers."), 400
    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {"status": "queued", "message": "Waiting to start…", "created": time.time()}
    threading.Thread(target=run_generation, args=(job_id, prompt, steps, guidance, seed), daemon=True).start()
    return jsonify(job_id=job_id), 202

@app.get("/api/jobs/<job_id>")
def generation_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return jsonify(error="This generation could not be found."), 404
        return jsonify({key: value for key, value in job.items() if key != "created"})

@app.get("/generated/<path:filename>")
def generated_file(filename: str):
    return send_from_directory(GENERATED_DIR, filename, as_attachment=request.args.get("download") == "1",
                               download_name=filename)

@app.errorhandler(413)
def file_too_large(_error):
    return jsonify(error="Each upload must be smaller than 25 MB."), 413

if __name__ == "__main__":
    app.run(host=os.environ.get("FLASK_HOST", "127.0.0.1"),
            port=int(os.environ.get("FLASK_PORT", "5000")), debug=False, threaded=True)

