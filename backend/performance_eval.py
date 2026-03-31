import argparse
import statistics
import time
from pathlib import Path

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple performance check for the /predict endpoint."
    )
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:8000",
        help="FastAPI base URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to a sample handwriting image file",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of prediction requests (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds (default: 120)",
    )
    return parser.parse_args()


def run_once(client: httpx.Client, api_base: str, image_path: Path) -> tuple[float, int]:
    with image_path.open("rb") as f:
        files = {"files": (image_path.name, f, "image/png")}
        start = time.perf_counter()
        response = client.post(f"{api_base}/predict", files=files)
        elapsed = (time.perf_counter() - start) * 1000.0
    return elapsed, response.status_code


def summarize(times_ms: list[float]) -> dict[str, float]:
    return {
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "avg_ms": statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "stdev_ms": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
    }


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)

    if not image_path.exists() or not image_path.is_file():
        print(f"ERROR: Image file not found: {image_path}")
        return 1

    if args.runs <= 0:
        print("ERROR: --runs must be greater than 0")
        return 1

    api_base = args.api_base.rstrip("/")

    with httpx.Client(timeout=args.timeout) as client:
        health = client.get(f"{api_base}/health")
        if health.status_code != 200:
            print(f"ERROR: Health check failed with status {health.status_code}")
            return 1

        print("Health check OK.")
        print(f"Running {args.runs} prediction requests on {image_path.name} ...")

        times_ms: list[float] = []
        status_codes: list[int] = []

        for i in range(args.runs):
            elapsed, status = run_once(client, api_base, image_path)
            times_ms.append(elapsed)
            status_codes.append(status)
            print(f"Run {i + 1:02d}: {elapsed:.2f} ms (status={status})")

    stats = summarize(times_ms)
    successful = sum(1 for code in status_codes if code == 200)

    print("\nSummary")
    print(f"Success rate: {successful}/{len(status_codes)}")
    print(f"Min:    {stats['min_ms']:.2f} ms")
    print(f"Max:    {stats['max_ms']:.2f} ms")
    print(f"Average:{stats['avg_ms']:.2f} ms")
    print(f"Median: {stats['median_ms']:.2f} ms")
    print(f"Std dev:{stats['stdev_ms']:.2f} ms")

    return 0 if successful == len(status_codes) else 2


if __name__ == "__main__":
    raise SystemExit(main())
