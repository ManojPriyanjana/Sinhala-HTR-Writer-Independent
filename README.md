# Sinhala HTR Writer-Independent

Writer-independent Sinhala handwritten text recognition (HTR) system with:
- FastAPI backend for health checks and OCR inference
- React + Vite frontend for file upload and result viewing

## Project Structure

```
backend/      FastAPI app, model loading, OCR inference
frontend/     React Vite UI
data/         Dataset and split artifacts
src/          Data preparation and segmentation scripts
report/       Report generation assets
notebooks/    Experiment notebooks
```

## Prerequisites

- Python 3.10+ (recommended 3.11)
- Node.js 18+ and npm
- Windows PowerShell (commands below use PowerShell syntax)

## Backend Setup and Run

From repository root:

```powershell
cd backend
..\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Backend URL:
- http://localhost:8000

Health endpoint:
- http://localhost:8000/health

## Frontend Setup and Run

Open a second terminal from repository root:

```powershell
cd frontend
npm install
npm run dev
```

Frontend URL:
- http://localhost:5173

The frontend uses `VITE_API_BASE` if provided, otherwise defaults to `http://localhost:8000`.

## API Endpoints

Custom endpoints:
- `GET /health`
- `POST /predict`

FastAPI auto docs (enabled by default):
- `GET /docs`
- `GET /redoc`
- `GET /openapi.json`

## Model Requirements

Place model files in `backend/models/`.

Auto-detected filenames/extensions include:
- `model.onnx`
- `model.pt`
- Any of: `.onnx`, `.pt`, `.pth`, `.ckpt`, `.bin`

For PyTorch models (`.pt/.pth/.ckpt/.bin`), charset metadata is required for decoding.
Provide one of:
- `backend/models/charset.json`
- `backend/models/train_labels.csv`
- `CHARSET_PATH` environment variable
- `TRAIN_LABELS_PATH` environment variable

Optional model path override:

```powershell
$env:MODEL_PATH = "C:\path\to\model.pth"
uvicorn main:app --reload --port 8000
```

## Quick API Checks

Health:

```powershell
curl http://localhost:8000/health
```

Predict (single or multiple files):

```powershell
curl -X POST "http://localhost:8000/predict" `
	-F "files=@C:\path\to\line_or_paragraph_image.png"
```

## Troubleshooting

- If `/health` returns `model_connected: false`:
	- Check model exists in `backend/models/` or set `MODEL_PATH`
	- Ensure charset metadata exists (`charset.json` or labels CSV)
- If frontend cannot reach backend:
	- Confirm backend is running on port `8000`
	- Confirm frontend is running on port `5173`
	- Check browser console/network for CORS or connection errors
- If dependencies fail to install:
	- Activate the correct virtual environment
	- Re-run `pip install -r backend/requirements.txt` and `npm install` in `frontend`

## Notes

- Backend predicts both line images and full paragraph/page images (auto line segmentation for paragraph-like inputs).
- Keep backend and frontend terminals running during usage.
