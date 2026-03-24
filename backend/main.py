from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import io
import json
import os
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

app = FastAPI(title="Sinhala HTR API", version="0.2")

# Vite React dev server default URL.
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
CHARSET_ENV_KEY = "CHARSET_PATH"
TRAIN_LABELS_ENV_KEY = "TRAIN_LABELS_PATH"
MODEL_EXTENSIONS = {".onnx", ".pt", ".pth", ".ckpt", ".bin"}
SUPPORTED_PYTORCH_EXTENSIONS = {".pt", ".pth", ".ckpt", ".bin"}
MODEL_IMAGE_HEIGHT = 64
MODEL_IMAGE_WIDTH = 512
TEXT_COLUMN_CANDIDATES = ("text", "transcription", "label", "sentence", "gt")
SEGMENT_MIN_LINE_HEIGHT = 18
SEGMENT_GAP = 6
SEGMENT_PADDING = 8
SEGMENT_HEIGHT_TRIGGER = 140


@dataclass
class ModelRuntime:
    model_path: Path
    signature: tuple[str, int]
    model: Any
    torch: Any
    np: Any
    cv2: Any
    device: str
    idx2char: dict[int, str]
    charset_source: str


_RUNTIME: ModelRuntime | None = None
_RUNTIME_SIGNATURE: tuple[str, int] | None = None
_RUNTIME_ERROR: str | None = None


def _clear_runtime_cache() -> None:
    global _RUNTIME, _RUNTIME_SIGNATURE, _RUNTIME_ERROR
    _RUNTIME = None
    _RUNTIME_SIGNATURE = None
    _RUNTIME_ERROR = None


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    return path


def resolve_model_path() -> Path | None:
    """
    Resolve model file in this order:
    1) MODEL_PATH environment variable
    2) backend/models/model.onnx or backend/models/model.pt
    3) first supported model file inside backend/models
    """
    model_path_env = os.getenv(MODEL_ENV_KEY, "").strip()
    if model_path_env:
        env_path = _resolve_path(model_path_env)
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


def _model_signature(model_path: Path) -> tuple[str, int]:
    return str(model_path.resolve()), model_path.stat().st_mtime_ns


def _load_charset_from_json(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        charset = [str(item) for item in raw]
    elif isinstance(raw, dict) and isinstance(raw.get("charset"), list):
        charset = [str(item) for item in raw["charset"]]
    elif isinstance(raw, dict):
        try:
            ordered = sorted(((int(k), str(v)) for k, v in raw.items()), key=lambda item: item[0])
        except Exception as exc:
            raise ValueError(f"Unsupported charset JSON format in {path}") from exc
        charset = [value for _, value in ordered]
    else:
        raise ValueError(f"Unsupported charset JSON format in {path}")

    if len(charset) < 2:
        raise ValueError(f"Charset in {path} is too small.")
    return charset


def _infer_text_column(fieldnames: list[str] | None) -> str | None:
    if not fieldnames:
        return None
    normalized = {name.lower(): name for name in fieldnames}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _load_charset_from_train_labels(path: Path) -> list[str]:
    chars: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        text_column = _infer_text_column(reader.fieldnames)
        if text_column is None:
            raise ValueError(
                f"Could not find a text column in {path}. "
                f"Expected one of: {', '.join(TEXT_COLUMN_CANDIDATES)}."
            )

        for row in reader:
            text = row.get(text_column, "")
            if not isinstance(text, str):
                continue
            chars.update(text)

    charset = ["[blank]"] + sorted(chars)
    if len(charset) < 2:
        raise ValueError(f"No text labels found in {path}.")
    return charset


def _find_charset(model_path: Path) -> tuple[list[str], str]:
    charset_candidates: list[Path] = []
    label_candidates: list[Path] = []

    charset_env = os.getenv(CHARSET_ENV_KEY, "").strip()
    if charset_env:
        charset_candidates.append(_resolve_path(charset_env))

    labels_env = os.getenv(TRAIN_LABELS_ENV_KEY, "").strip()
    if labels_env:
        label_candidates.append(_resolve_path(labels_env))

    charset_candidates.extend(
        [
            model_path.parent / "charset.json",
            model_path.parent / f"{model_path.stem}_charset.json",
            model_path.parent / "idx2char.json",
        ]
    )
    label_candidates.extend(
        [
            model_path.parent / "train_labels.csv",
            model_path.parent / f"{model_path.stem}_labels.csv",
            BASE_DIR.parent / "data" / "train_labels.csv",
        ]
    )

    for candidate in charset_candidates:
        if candidate.exists() and candidate.is_file():
            return _load_charset_from_json(candidate), str(candidate)

    for candidate in label_candidates:
        if candidate.exists() and candidate.is_file():
            return _load_charset_from_train_labels(candidate), str(candidate)

    raise RuntimeError(
        "Missing charset metadata. Add one of these files: "
        "backend/models/charset.json, backend/models/train_labels.csv, "
        "or set CHARSET_PATH / TRAIN_LABELS_PATH environment variables."
    )


def _import_runtime_stack() -> tuple[Any, Any, Any, Any]:
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install dependencies with: "
            "pip install torch torchvision numpy opencv-python-headless"
        ) from exc

    try:
        from torchvision import models  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "torchvision is not installed. Install dependencies with: "
            "pip install torchvision"
        ) from exc

    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "numpy is not installed. Install dependencies with: "
            "pip install numpy"
        ) from exc

    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "OpenCV is not installed. Install dependencies with: "
            "pip install opencv-python-headless"
        ) from exc

    return torch, models, np, cv2


def _build_densenet_crnn(torch: Any, models: Any, num_classes: int) -> Any:
    nn = torch.nn

    class DenseNetCRNN(nn.Module):
        def __init__(self, classes: int):
            super().__init__()
            densenet = models.densenet121(weights=None)

            # Match training notebook: grayscale input + width-preserving transitions.
            densenet.features.conv0 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            densenet.features.transition1.pool = nn.AvgPool2d(kernel_size=2, stride=(2, 1))
            densenet.features.transition2.pool = nn.AvgPool2d(kernel_size=2, stride=(2, 1))
            densenet.features.transition3.pool = nn.AvgPool2d(kernel_size=2, stride=(2, 1))

            self.features = densenet.features
            self.rnn = nn.LSTM(
                input_size=2048,
                hidden_size=256,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
                dropout=0.5,
            )
            self.fc = nn.Linear(512, classes)

        def forward(self, x: Any) -> Any:
            feat = self.features(x)
            feat = torch.relu(feat)
            b, c, h, w = feat.size()
            feat = feat.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)
            rnn_out, _ = self.rnn(feat)
            return self.fc(rnn_out)

    return DenseNetCRNN(num_classes)


def _strip_prefix(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            out[key[len(prefix) :]] = value
        else:
            out[key] = value
    return out


def _extract_state_dict(checkpoint: Any, torch: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Unsupported checkpoint format. Expected a dict-based PyTorch checkpoint.")

    for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            return candidate

    tensor_map = {
        key: value
        for key, value in checkpoint.items()
        if isinstance(value, (torch.Tensor, torch.nn.Parameter))
    }
    if tensor_map:
        return tensor_map

    raise RuntimeError(
        "Could not find model weights inside checkpoint. "
        "Expected keys like 'state_dict' or raw tensor parameter map."
    )


def _load_state_dict(model: Any, state_dict: dict[str, Any]) -> None:
    candidates = [
        state_dict,
        _strip_prefix(state_dict, "module."),
        _strip_prefix(state_dict, "model."),
        _strip_prefix(state_dict, "model.module."),
    ]

    first_error: Exception | None = None
    for candidate in candidates:
        try:
            model.load_state_dict(candidate, strict=True)
            return
        except Exception as exc:
            if first_error is None:
                first_error = exc

    if first_error is None:
        raise RuntimeError("Failed to load checkpoint state dict.")
    raise RuntimeError(str(first_error)) from first_error


def _load_runtime(model_path: Path) -> ModelRuntime:
    suffix = model_path.suffix.lower()
    if suffix not in SUPPORTED_PYTORCH_EXTENSIONS:
        raise RuntimeError(
            f"Model extension '{suffix}' is not supported for inference yet. "
            "Use a PyTorch checkpoint (.pt/.pth/.ckpt/.bin)."
        )

    charset, charset_source = _find_charset(model_path)
    torch, models, np, cv2 = _import_runtime_stack()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_densenet_crnn(torch=torch, models=models, num_classes=len(charset)).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = _extract_state_dict(checkpoint, torch=torch)
    _load_state_dict(model, state_dict)
    model.eval()

    return ModelRuntime(
        model_path=model_path,
        signature=_model_signature(model_path),
        model=model,
        torch=torch,
        np=np,
        cv2=cv2,
        device=device,
        idx2char={index: str(char) for index, char in enumerate(charset)},
        charset_source=charset_source,
    )


def _get_runtime(model_path: Path) -> tuple[ModelRuntime | None, str | None]:
    global _RUNTIME, _RUNTIME_SIGNATURE, _RUNTIME_ERROR
    signature = _model_signature(model_path)
    if _RUNTIME_SIGNATURE == signature:
        return _RUNTIME, _RUNTIME_ERROR

    try:
        _RUNTIME = _load_runtime(model_path)
        _RUNTIME_SIGNATURE = signature
        _RUNTIME_ERROR = None
    except Exception as exc:
        _RUNTIME = None
        _RUNTIME_SIGNATURE = signature
        _RUNTIME_ERROR = str(exc)

    return _RUNTIME, _RUNTIME_ERROR


def _greedy_decode(logits: Any, idx2char: dict[int, str]) -> list[str]:
    # logits: (B, T, C)
    pred = logits.argmax(dim=-1).detach().cpu().numpy()
    decoded: list[str] = []
    for seq in pred:
        last = None
        chars: list[str] = []
        for token_id in seq:
            token = int(token_id)
            if token != last and token != 0:
                chars.append(idx2char.get(token, ""))
            last = token
        decoded.append("".join(chars))
    return decoded


def _decode_image_to_bgr(content: bytes, runtime: ModelRuntime) -> tuple[Any, int, int]:
    raw = runtime.np.frombuffer(content, dtype=runtime.np.uint8)
    bgr = runtime.cv2.imdecode(raw, runtime.cv2.IMREAD_COLOR)
    if bgr is not None:
        height, width = bgr.shape[:2]
        return bgr, width, height

    try:
        with Image.open(io.BytesIO(content)) as img:
            rgb = img.convert("RGB")
            width, height = rgb.size
            rgb_np = runtime.np.array(rgb, dtype=runtime.np.uint8)
            bgr = runtime.cv2.cvtColor(rgb_np, runtime.cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise ValueError("Invalid image file") from exc

    return bgr, width, height


def _remove_ruled_lines_horizontal_only(bgr: Any, runtime: ModelRuntime) -> Any:
    hsv = runtime.cv2.cvtColor(bgr, runtime.cv2.COLOR_BGR2HSV)
    lower = runtime.np.array([85, 10, 90], dtype=runtime.np.uint8)
    upper = runtime.np.array([140, 220, 255], dtype=runtime.np.uint8)
    blue_mask = runtime.cv2.inRange(hsv, lower, upper)

    width = bgr.shape[1]
    kernel_width = max(25, width // 15)
    horiz_kernel = runtime.cv2.getStructuringElement(runtime.cv2.MORPH_RECT, (kernel_width, 1))
    horiz = runtime.cv2.morphologyEx(
        blue_mask, runtime.cv2.MORPH_OPEN, horiz_kernel, iterations=1
    )
    horiz = runtime.cv2.dilate(
        horiz,
        runtime.cv2.getStructuringElement(runtime.cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    return runtime.cv2.inpaint(bgr, horiz, 3, runtime.cv2.INPAINT_TELEA)


def _segment_lines_by_projection(
    bw: Any,
    runtime: ModelRuntime,
    min_line_height: int = SEGMENT_MIN_LINE_HEIGHT,
    gap: int = SEGMENT_GAP,
) -> list[tuple[int, int]]:
    row_sum = (bw > 0).sum(axis=1).astype(runtime.np.float32)
    kernel = runtime.np.ones(15, dtype=runtime.np.float32) / 15.0
    smooth = runtime.np.convolve(row_sum, kernel, mode="same")

    peak = float(runtime.np.max(smooth)) if smooth.size > 0 else 0.0
    threshold = max(5, int(0.15 * peak))
    is_text = smooth > threshold

    raw_lines: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for y, value in enumerate(is_text):
        if value and not in_run:
            in_run = True
            start = y
        elif not value and in_run:
            end = y
            in_run = False
            if (end - start) >= min_line_height:
                raw_lines.append((start, end))

    if in_run:
        end = len(is_text)
        if (end - start) >= min_line_height:
            raw_lines.append((start, end))

    merged: list[tuple[int, int]] = []
    for y1, y2 in raw_lines:
        if not merged:
            merged.append((y1, y2))
            continue
        prev_y1, prev_y2 = merged[-1]
        if y1 - prev_y2 <= gap:
            merged[-1] = (prev_y1, y2)
        else:
            merged.append((y1, y2))
    return merged


def _extract_line_crops(content: bytes, runtime: ModelRuntime) -> tuple[list[Any], int, int]:
    bgr, width, height = _decode_image_to_bgr(content, runtime)
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions")

    cleaned_bgr = _remove_ruled_lines_horizontal_only(bgr, runtime)
    recognition_gray = runtime.cv2.cvtColor(cleaned_bgr, runtime.cv2.COLOR_BGR2GRAY)

    # Line image uploads from your UI are usually short in height; do not over-segment those.
    if recognition_gray.shape[0] <= SEGMENT_HEIGHT_TRIGGER:
        return [recognition_gray], width, height

    clahe = runtime.cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    segmentation_gray = clahe.apply(recognition_gray)
    segmentation_gray = runtime.cv2.convertScaleAbs(segmentation_gray, alpha=1.3, beta=0)

    bw = runtime.cv2.adaptiveThreshold(
        segmentation_gray,
        255,
        runtime.cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        runtime.cv2.THRESH_BINARY_INV,
        31,
        12,
    )
    bw = runtime.cv2.morphologyEx(
        bw,
        runtime.cv2.MORPH_CLOSE,
        runtime.cv2.getStructuringElement(runtime.cv2.MORPH_RECT, (25, 3)),
        iterations=1,
    )

    line_coords = _segment_lines_by_projection(bw, runtime=runtime)
    if not line_coords:
        return [recognition_gray], width, height

    crops: list[Any] = []
    image_height = recognition_gray.shape[0]
    for y1, y2 in line_coords:
        yy1 = max(0, y1 - SEGMENT_PADDING)
        yy2 = min(image_height, y2 + SEGMENT_PADDING)
        crop = recognition_gray[yy1:yy2, :]
        if crop.size == 0:
            continue
        crops.append(crop)

    if not crops:
        return [recognition_gray], width, height
    return crops, width, height


def _preprocess_line_image(line_gray: Any, runtime: ModelRuntime) -> Any:
    height, width = line_gray.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions")

    new_width = int(width * (MODEL_IMAGE_HEIGHT / float(height)))
    new_width = max(1, min(new_width, MODEL_IMAGE_WIDTH))
    resized = runtime.cv2.resize(
        line_gray, (new_width, MODEL_IMAGE_HEIGHT), interpolation=runtime.cv2.INTER_AREA
    )

    canvas = runtime.np.full((MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH), 255, dtype=runtime.np.uint8)
    canvas[:, :new_width] = resized
    x = runtime.torch.from_numpy(canvas).float().unsqueeze(0).unsqueeze(0) / 255.0
    return x.to(runtime.device)


def _predict_text(runtime: ModelRuntime, content: bytes) -> tuple[str, float, dict[str, int]]:
    crops, width, height = _extract_line_crops(content, runtime)
    lines: list[str] = []
    confidences: list[float] = []

    runtime.model.eval()
    with runtime.torch.no_grad():
        for crop in crops:
            x = _preprocess_line_image(crop, runtime)
            logits = runtime.model(x)
            decoded = _greedy_decode(logits, runtime.idx2char)
            text = decoded[0] if decoded else ""
            lines.append(text)

            # Match Colab confidence behavior: average max-prob over non-blank timesteps.
            probs = runtime.torch.softmax(logits, dim=-1)
            max_probs, pred_ids = runtime.torch.max(probs, dim=-1)
            p_vals = max_probs.squeeze(0)
            ids = pred_ids.squeeze(0)
            non_blank = p_vals[ids != 0]
            if int(non_blank.numel()) > 0:
                conf = float(non_blank.mean().item())
            else:
                conf = 1.0
            confidences.append(max(0.0, min(1.0, conf)))

    text = "\n".join(lines).strip()
    avg_conf = (sum(confidences) / len(confidences)) if confidences else 0.0
    avg_conf = max(0.0, min(1.0, avg_conf))
    return text, avg_conf, {"width": width, "height": height, "detected_lines": len(crops)}


def get_model_status() -> dict[str, Any]:
    model_path = resolve_model_path()
    if model_path is None:
        _clear_runtime_cache()
        return {
            "model_connected": False,
            "model_path": None,
            "message": "Please connect with model.",
        }

    runtime, runtime_error = _get_runtime(model_path)
    if runtime is None:
        detail = runtime_error or "Model found but failed to load."
        return {
            "model_connected": False,
            "model_path": str(model_path),
            "message": f"Model found, but not ready: {detail}",
        }

    return {
        "model_connected": True,
        "model_path": str(model_path),
        "message": "Model connected.",
        "charset_source": runtime.charset_source,
        "charset_size": len(runtime.idx2char),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", **get_model_status()}


@app.post("/predict")
async def predict(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    status = get_model_status()
    if not status["model_connected"]:
        raise HTTPException(status_code=503, detail=status["message"])

    model_path = Path(str(status["model_path"]))
    runtime, runtime_error = _get_runtime(model_path)
    if runtime is None:
        detail = runtime_error or "Model runtime is unavailable."
        raise HTTPException(status_code=503, detail=detail)

    lines: list[dict[str, Any]] = []
    for upload in files:
        content = await upload.read()
        line_id = upload.filename or str(uuid.uuid4())
        if not content:
            lines.append(
                {
                    "line_id": line_id,
                    "text": "",
                    "confidence": 0.0,
                    "error": "Empty file",
                }
            )
            continue

        try:
            text, confidence, meta = _predict_text(runtime, content)
            lines.append(
                {
                    "line_id": line_id,
                    "text": text,
                    "confidence": confidence,
                    "meta": meta,
                }
            )
        except ValueError as exc:
            lines.append(
                {
                    "line_id": line_id,
                    "text": "",
                    "confidence": 0.0,
                    "error": str(exc),
                }
            )
        except Exception as exc:
            lines.append(
                {
                    "line_id": line_id,
                    "text": "",
                    "confidence": 0.0,
                    "error": f"Inference failed: {exc}",
                }
            )

    return {"lines": lines, **status}
