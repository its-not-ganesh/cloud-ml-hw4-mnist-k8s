import hashlib
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from mnist_service.model import ARCHITECTURE, MNISTNet, normalize
from mnist_service.preprocess import uploaded_pixels

MAX_BODY = 2 * 1024 * 1024


class BodyLimit:
    """Bound the entire body, including chunked multipart requests, before parsing."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > MAX_BODY:
                return await JSONResponse({"detail": "Request exceeds 2 MiB"}, 413)(scope, receive, send)
            if not message.get("more_body", False):
                break
        async def replay():
            return {"type": "http.request", "body": bytes(body), "more_body": False}
        await self.app(scope, replay, send)


class Drawing(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pixels: list[float] = Field(min_length=784, max_length=784)


def create_app(data_dir=None):
    data = Path(data_dir or os.environ.get("MODEL_DIR", "/data"))
    state = dict(model=None, metadata=None, metrics=None, examples=[])
    lock = Lock()

    @asynccontextmanager
    async def lifespan(app):
        torch.set_num_threads(2)
        try:
            metadata = json.loads((data / "metadata.json").read_text())
            checkpoint = (data / "model.pt").read_bytes()
            if metadata["architecture"] != ARCHITECTURE or hashlib.sha256(checkpoint).hexdigest() != metadata["checkpoint_sha256"]:
                raise ValueError("Checkpoint provenance mismatch")
            model = MNISTNet()
            model.load_state_dict(torch.load(io.BytesIO(checkpoint), map_location="cpu", weights_only=True))
            model.eval()
            metrics = json.loads((data / "metrics.json").read_text())
            examples = json.loads((data / "examples/index.json").read_text())
            state.update(model=model, metadata=metadata, metrics=metrics, examples=examples)
            logging.getLogger("uvicorn.error").info("Loaded checkpoint SHA256 %s", metadata["checkpoint_sha256"])
        except (OSError, ValueError, KeyError, RuntimeError):
            logging.getLogger("uvicorn.error").exception("Model unavailable; readiness remains false")
        yield

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(BodyLimit)

    def require_model():
        if state["model"] is None:
            raise HTTPException(503, "Model unavailable")

    def predict(pixels):
        require_model()
        with lock, torch.inference_mode():
            start = time.perf_counter()
            probabilities = state["model"](normalize(pixels)).softmax(1)[0].tolist()
            latency = (time.perf_counter() - start) * 1000
        digit = int(np.argmax(probabilities))
        return dict(digit=digit, confidence=probabilities[digit], probabilities=probabilities,
                    inference_ms=latency, checkpoint_sha256=state["metadata"]["checkpoint_sha256"])

    @app.get("/")
    def home():
        return FileResponse(Path(__file__).parent / "static/index.html")

    @app.get("/healthz")
    def health():
        return {"status": "alive"}

    @app.get("/readyz")
    def ready():
        require_model()
        return {"status": "ready", "checkpoint_sha256": state["metadata"]["checkpoint_sha256"]}

    @app.get("/api/metadata")
    def metadata():
        require_model()
        return dict(model=state["metadata"], metrics=state["metrics"])

    @app.get("/api/examples")
    def examples():
        require_model()
        return state["examples"]

    def example_path(identifier):
        item = next((x for x in state["examples"] if x["id"] == identifier), None)
        if item is None or not identifier.isdigit():
            raise HTTPException(404, "Unknown held-out example")
        return item, data / "examples" / f"{identifier}.png"

    @app.get("/api/examples/{identifier}/image")
    def example_image(identifier: str):
        require_model()
        return FileResponse(example_path(identifier)[1], media_type="image/png")

    @app.post("/api/predict/example/{identifier}")
    def example(identifier: str):
        require_model()
        item, path = example_path(identifier)
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("L"), dtype=np.float32) / 255
        return dict(**predict(pixels), actual_label=item["label"], example_id=identifier)

    @app.post("/api/predict/drawn")
    def drawn(payload: Drawing):
        pixels = np.asarray(payload.pixels)
        if not np.isfinite(pixels).all() or (pixels < 0).any() or (pixels > 1).any():
            raise HTTPException(422, "Pixels must be finite and within [0,1]")
        return predict(pixels)

    @app.post("/api/predict")
    async def upload(file: UploadFile = File(...)):
        if file.content_type not in {"image/png", "image/jpeg", "image/webp"}:
            raise HTTPException(415, "Use PNG, JPEG, or WebP")
        try:
            pixels = uploaded_pixels(await file.read(MAX_BODY + 1))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return predict(pixels)

    return app


app = create_app()
