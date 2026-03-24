# Model Directory

Put your trained OCR model file here.

Supported model filenames/extensions for auto-detection:
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

## Charset requirement for CTC decode
For `.pt/.pth` models, backend also needs character mapping (`idx -> Sinhala char`).

Provide one of these:
- `backend/models/charset.json`
- `backend/models/train_labels.csv` (must have a text column like `text`, `transcription`, `label`, `sentence`, or `gt`)
- set env var `CHARSET_PATH` to a charset JSON file
- set env var `TRAIN_LABELS_PATH` to a labels CSV file

Example `charset.json`:
```json
["[blank]", "අ", "ආ", "ඇ"]
```

If model or charset is missing/invalid, `/health` returns `model_connected: false` and prediction is blocked.

## Input behavior
- If upload image is a **single line**, backend predicts that line directly.
- If upload image is a **full paragraph/page**, backend auto-segments lines (Colab-style) and returns joined text.
