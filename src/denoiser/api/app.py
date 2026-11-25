import io
from threading import Lock
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from denoiser import Denoiser
from denoiser.api.request_types import Iterations
from denoiser.utils.errors import ApiMessages
from denoiser.utils.common import img_to_tensor, tensor_to_img
from denoiser.training.weights import load_config, get_device, load_weights

BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = Path(__file__).resolve().parents[3] / "static"
MAX_IMAGE_SIDE = 4096
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(_app: FastAPI):
    cfg = load_config()
    device = get_device()
    model = Denoiser(cfg["denoise"]).to(device)
    load_weights(model, cfg, device)
    model.eval()

    _app.state.device = device
    _app.state.model = model
    _app.state.inference_lock = Lock()
    yield

    del _app.state.model
    torch.cuda.empty_cache()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(
    directory=FRONTEND_DIR),
    name="static"
)


@app.get("/")
async def root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/predict")
def predict(file: UploadFile, iterations: Iterations):
    lock = app.state.inference_lock
    acquired = lock.acquire(blocking=False)

    if not acquired:
        file.file.close()
        raise HTTPException(status_code=503, detail=ApiMessages.BUSY, headers={"Retry-After": "1"})

    try:
        content = file.file.read(MAX_UPLOAD_BYTES + 1)

        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=ApiMessages.FILE_TOO_LARGE.format(limit=10))

        return restore_image(content, iterations)
    finally:
        if acquired:
            lock.release()

        file.file.close()


def restore_image(content, iterations):
    try:
        with Image.open(io.BytesIO(content)) as source:
            if source.format not in {"JPEG", "PNG", "WEBP", "BMP"} or getattr(source, "is_animated", False):
                raise HTTPException(status_code=415, detail=ApiMessages.UNSUPPORTED_IMAGE)

            if max(source.size) > MAX_IMAGE_SIDE:
                raise HTTPException(status_code=413, detail=ApiMessages.IMAGE_TOO_LARGE.format(limit=MAX_IMAGE_SIDE))

            image = ImageOps.exif_transpose(source).convert("RGB")
    except Image.DecompressionBombError as exc:
        raise HTTPException(status_code=413, detail=ApiMessages.IMAGE_TOO_LARGE.format(limit=MAX_IMAGE_SIDE)) from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise HTTPException(status_code=422, detail=ApiMessages.INVALID_IMAGE) from exc

    transform = img_to_tensor(normalize=True)
    model = app.state.model
    image = transform(image).to(app.state.device)

    with torch.inference_mode():
        for i in range(iterations):
            image = (image - model(image)).clamp(-1, 1)

    image = tensor_to_img(image)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True, app_dir=str(BASE_DIR / "src"))