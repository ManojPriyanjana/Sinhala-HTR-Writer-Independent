from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import uuid

app = FastAPI(title="Sinhala HTR API", version="0.1")

# Vite React dev server default URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(files: list[UploadFile] = File(...)):
    """
    Upload multiple line images. Returns mock Sinhala text now.
    Later: replace mock with real model inference.
    """
    lines = []
    for f in files:
        content = await f.read()

        # validate image
        try:
            img = Image.open(io.BytesIO(content)).convert("RGB")
            w, h = img.size
        except Exception:
            lines.append({
                "line_id": f.filename,
                "text": "",
                "confidence": 0.0,
                "error": "Invalid image file"
            })
            continue

        # mock result
        lines.append({
            "line_id": f.filename or str(uuid.uuid4()),
            "text": "මෙය demo output එකයි",
            "confidence": 0.75,
            "meta": {"width": w, "height": h}
        })

    return {"lines": lines}
