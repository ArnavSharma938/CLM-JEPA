from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .chemfm_native import canonicalize


def prediction_records(predictions: Iterable[str], targets: Iterable[str]) -> list[dict]:
    records = []
    for index, (prediction, target) in enumerate(zip(predictions, targets)):
        canonical_prediction = canonicalize(prediction)
        canonical_target = canonicalize(target)
        records.append(
            {
                "index": index,
                "prediction": prediction,
                "target": target,
                "canonical_prediction": canonical_prediction,
                "canonical_target": canonical_target,
                "valid": bool(canonical_prediction),
                "exact": bool(canonical_prediction) and canonical_prediction == canonical_target,
            }
        )
    return records


def metrics(records: Iterable[dict]) -> dict[str, int]:
    rows = list(records)
    return {
        "count": len(rows),
        "valid_products": sum(bool(row["valid"]) for row in rows),
        "exact_products": sum(bool(row["exact"]) for row in rows),
    }


def save_records(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
