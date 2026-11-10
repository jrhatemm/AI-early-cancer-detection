"""FastAPI server so the website can call the real models."""
import base64

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from config import settings as cfg
from modules import history, image_loader, pipeline

app = FastAPI(title="Cancer Detection API")
app.add_middleware(CORSMiddleware, allow_origins=cfg.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/cancers")
def cancers():
    return cfg.CANCER_TYPES


@app.get("/api/history")
def get_history(limit: int = 20):
    return history.recent(min(limit, 200))


@app.get("/api/stats")
def get_stats():
    return history.stats()


# Plain `def` (not async): inference is CPU/GPU-heavy, so FastAPI runs it in a worker thread
# instead of freezing the event loop for every other request.
@app.post("/api/analyze")
def analyze(file: UploadFile = File(...), cancer: str = Form(cfg.DEFAULT_CANCER)):
    if cancer not in cfg.CANCER_TYPES:
        raise HTTPException(400, f"Unsupported cancer type: {cancer}")
    limit = cfg.MAX_UPLOAD_MB * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File larger than {cfg.MAX_UPLOAD_MB} MB")
    try:
        bgr = image_loader.decode_bytes(data)
        result, overlay = pipeline.analyze(bgr, cancer, source=file.filename)
    except pipeline.ImageQualityError as e:
        raise HTTPException(422, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    ok, buf = cv2.imencode(".png", overlay)
    result["overlay_png_b64"] = base64.b64encode(buf.tobytes()).decode()
    return result
