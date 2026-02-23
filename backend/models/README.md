# Model Directory

Put your trained OCR model file here.

Supported filenames/extensions for auto-detection:
- `model.onnx`
- `model.pt`
- any of: `.onnx`, `.pt`, `.pth`, `.ckpt`, `.bin`

## Option 1: Default path
Place your file in this folder (for example `backend/models/model.onnx`).

## Option 2: Custom path
Set environment variable `MODEL_PATH` before running backend.

PowerShell example:
```powershell
$env:MODEL_PATH = "C:\\path\\to\\your\\model.onnx"
uvicorn main:app --reload --port 8000
```

If no model is detected, the app shows: "Please connect with model." and prediction is blocked.
