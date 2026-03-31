# Software Testing Package (Thesis Artifact)

This document defines a lightweight and formal testing strategy for the Sinhala handwriting recognition demo system.

## 1) Scope and Main Testable Features

### Backend (FastAPI)
- `GET /health` returns API status and model connection readiness.
- `POST /predict` accepts uploaded handwriting image files and returns recognition results.
- Error handling for missing files, empty files, and invalid image content.
- Response payload structure (`lines`, confidence, text, metadata) and service-level status handling.

### Frontend (React + Vite)
- API connectivity status indicator (backend reachable/unreachable).
- Model connection indicator and message display.
- Upload interaction and result rendering.
- Handling error messages returned by backend.

## 2) Testing Plan

### Functional Testing
- Verify endpoint correctness (`/health`, `/predict`) with valid and invalid inputs.
- Validate backend error handling for edge cases.
- Confirm that frontend displays connectivity and prediction responses correctly.

### System Testing
- End-to-end flow:
  1. Start backend and frontend.
  2. Check health status.
  3. Upload one or more handwriting images.
  4. Validate recognition output display and confidence values.

### Usability Testing
- Evaluate whether upload, status indicators, and result view are understandable.
- Observe user ability to complete the main task with minimal guidance.
- Record subjective feedback from at least 5 users (students/supervisor peers).

### Performance Testing
- Measure prediction response time using repeated requests to `/predict`.
- Report per-request timing and aggregate statistics (min/max/avg/median/stdev).

### Compatibility Testing
- Validate on Windows (primary environment).
- Optional checks on Linux/macOS for backend startup and API behavior.
- Browser checks: Chrome, Edge (and optionally Firefox) for frontend behavior.

### User Acceptance Testing (UAT)
- Academic acceptance criteria:
  - System starts without errors.
  - Health status is visible.
  - Prediction requests return results for valid images.
  - Invalid inputs are handled gracefully.
- Conduct UAT with supervisor/demo audience checklist and sign-off summary.

## 3) Automated Tests (Backend)

Location:
- `backend/tests/test_api.py`

Coverage:
- Health endpoint success response.
- Prediction endpoint success response shape.
- Missing file validation (`422`).
- Empty file handling (line-level error message).
- Invalid image handling (line-level error message).
- Model disconnected behavior (`503`).
- Runtime unavailable behavior (`503`).

Run commands:

```bash
cd backend
pip install -r requirements.txt
pytest -q
```

## 4) Lightweight Frontend Testing

To keep complexity low and avoid changes to core behavior, this package focuses on backend automation plus structured manual frontend checks in `manual_test_cases.csv`.

## 5) Manual Testing Evidence

Use:
- `manual_test_cases.csv`

Fill the `actual_result` and `status` columns during test execution for thesis appendix evidence.

## 6) Performance Evaluation

Script:
- `backend/performance_eval.py`

Example usage:

```bash
cd backend
python performance_eval.py --image "path/to/sample_line.png" --runs 10
```

Output includes:
- response time per prediction request
- min/max/average/median/standard deviation
- success rate

## 7) Thesis Reporting Notes

For your final report, include:
- Automated test pass summary (e.g., `7 passed`).
- Filled manual test case table with pass/fail outcomes.
- Performance summary from at least one test session (hardware + environment noted).
- Brief discussion of known limitations (demo scope, model dependency, environment variance).
