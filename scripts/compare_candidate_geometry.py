#!/usr/bin/env python
"""Compare overlapping candidate-geometry rows before/after batching."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path


IDENTITY = ("checkpoint", "panel_index", "view", "candidate", "role", "layer")


def rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def numeric_leaves(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from numeric_leaves(child, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from numeric_leaves(child, f"{prefix}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, float(value)


def run(reference: Path, optimized: Path, output: Path):
    baseline = {
        tuple(row[key] for key in IDENTITY): row
        for row in rows(reference) if row["checkpoint"] == "native_r8_s533"
    }
    compared = 0
    leaves = 0
    maximum = 0.0
    squared = 0.0
    above = {"1e-7": 0, "1e-6": 0, "1e-5": 0, "1e-4": 0}
    missing = []
    for row in rows(optimized):
        identity = tuple(row[key] for key in IDENTITY)
        old = baseline.get(identity)
        if old is None:
            missing.append(identity)
            continue
        old_values = dict(numeric_leaves(old))
        new_values = dict(numeric_leaves(row))
        for key in old_values.keys() & new_values.keys():
            # This diagnostic was intentionally corrected from a product-only
            # proxy to the executable released source+product semantic path.
            if key in {"released_loss", "paper_loss"}:
                continue
            first, second = old_values[key], new_values[key]
            if not (math.isfinite(first) and math.isfinite(second)):
                continue
            difference = abs(first - second)
            maximum = max(maximum, difference)
            squared += difference * difference
            leaves += 1
            for threshold in above:
                above[threshold] += difference > float(threshold)
        compared += 1
    payload = {
        "reference": str(reference), "optimized": str(optimized),
        "overlapping_rows": compared, "optimized_rows_missing_reference": len(missing),
        "numeric_leaves": leaves, "maximum_absolute_difference": maximum,
        "rms_absolute_difference": math.sqrt(squared / max(1, leaves)),
        "counts_above_tolerance": above,
        "intentional_exclusions": [
            "released_loss (corrected semantic path and bounded fixed spans)",
            "paper_loss (bounded fixed spans)",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("optimized", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.reference, args.optimized, args.output)
