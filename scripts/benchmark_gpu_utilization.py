"""Sample NVIDIA device utilization during a benchmark interval."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path


FIELDS = (
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "power.draw",
    "clocks.sm",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    samples: list[list[float]] = []
    started = time.perf_counter()
    while time.perf_counter() - started < args.seconds:
        line = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(FIELDS)}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip().splitlines()[0]
        samples.append([float(value.strip()) for value in line.split(",")])
        remaining = args.interval - (time.perf_counter() - started) % args.interval
        time.sleep(min(remaining, args.interval))
    columns = list(zip(*samples))
    summary = {
        "seconds": time.perf_counter() - started,
        "sample_count": len(samples),
        "fields": {
            field: {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "minimum": min(values),
                "maximum": max(values),
            }
            for field, values in zip(FIELDS, columns)
        },
    }
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
