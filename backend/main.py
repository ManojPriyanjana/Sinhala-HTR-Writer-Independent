from pathlib import Path
import io
import os
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI(title="Sinhala HTR API", version="0.1")

# Vite React dev server default URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_ENV_KEY = "MODEL_PATH"
MODEL_EXTENSIONS = {".onnx", ".pt", ".pth", ".ckpt", ".bin"}


def resolve_model_path() -> Path | None:
    """
    Resolve model file in this order:
    1) MODEL_PATH environment variable
    2) backend/models/model.onnx or backend/models/model.pt
    3) first supported model file inside backend/models
    """
    model_path_env = os.getenv(MODEL_ENV_KEY, "").strip()
    if model_path_env:
        env_path = Path(model_path_env).expanduser()
        if not env_path.is_absolute():
            env_path = (BASE_DIR / env_path).resolve()
        if env_path.exists() and env_path.is_file():
            return env_path

    for filename in ("model.onnx", "model.pt"):
        candidate = MODEL_DIR / filename
        if candidate.exists() and candidate.is_file():
            return candidate

    if MODEL_DIR.exists():
        for candidate in sorted(MODEL_DIR.iterdir()):
            if candidate.is_file() and candidate.suffix.lower() in MODEL_EXTENSIONS:
                return candidate

    return None


def get_model_status() -> dict:
    model_path = resolve_model_path()
    if model_path is None:
        return {
            "model_connected": False,
            "model_path": None,
            "message": "Please connect with model.",
        }
    return {
        "model_connected": True,
        "model_path": str(model_path),
        "message": "Model connected.",
    }


@app.get("/health")
def health():
    return {"status": "ok", **get_model_status()}


@app.post("/predict")
async def predict(files: list[UploadFile] = File(...)):
    """
    Upload multiple line images.
    If model is not connected, return clear instruction.
    Later: replace mock with real model inference.
    """
    status = get_model_status()
    if not status["model_connected"]:
        raise HTTPException(status_code=503, detail=status["message"])

    lines = []
    for upload in files:
        content = await upload.read()

        try:
            img = Image.open(io.BytesIO(content)).convert("RGB")
            width, height = img.size
        except Exception:
            lines.append(
                {
                    "line_id": upload.filename,
                    "text": "",
                    "confidence": 0.0,
                    "error": "Invalid image file",
                }
            )
            continue

        # TODO: replace this mock output with real model inference.
        lines.append(
            {
                "line_id": upload.filename or str(uuid.uuid4()),
                "text": "Mock output: model file detected. Replace with real inference.",
                "confidence": 0.75,
                "meta": {"width": width, "height": height},
            }
        )

    return {"lines": lines, **status}
