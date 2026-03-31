# Performance Test Notes (Thesis Use)

## Objective
Measure backend prediction response time for the Sinhala HTR demo API.

## Method
- Endpoint under test: `POST /predict`
- Tool: `backend/performance_eval.py`
- Metric unit: milliseconds (ms)
- Repeated calls using same sample image

## Test Environment Template
- Date:
- Machine/CPU:
- RAM:
- OS:
- Python version:
- Backend run command: `uvicorn main:app --reload --port 8000`

## Execution Command

```bash
cd backend
python performance_eval.py --image "path/to/sample_line.png" --runs 10
```

## Capture These Outputs
- Per-run latency values
- Min latency
- Max latency
- Average latency
- Median latency
- Standard deviation
- Success rate (`200` responses / total)

## Result Table Template

| Session | Runs | Min (ms) | Max (ms) | Avg (ms) | Median (ms) | Std Dev (ms) | Success Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Session 1 | 10 |  |  |  |  |  |  |
| Session 2 (optional) | 10 |  |  |  |  |  |  |

## Interpretation Guidance
- Lower average latency indicates faster prediction handling.
- High standard deviation suggests unstable runtime or system load variance.
- If failures occur, include error details and probable cause (e.g., model not connected, invalid input, server overload).

## Thesis Reporting Tip
In the report, combine this quantitative latency data with functional test pass results to demonstrate both correctness and operational efficiency of the software artifact.
