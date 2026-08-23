"""Build an ordered, matched endpoint panel from resumable worker shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def subset_panel(
    manifest: Path,
    prediction_inputs: list[Path],
    limit: int,
) -> tuple[list[dict], list[dict], int]:
    if limit < 1:
        raise ValueError("limit must be positive")
    manifest_rows = read_jsonl(manifest)
    if len(manifest_rows) < limit:
        raise ValueError(f"manifest has {len(manifest_rows)} rows, fewer than limit {limit}")
    panel = manifest_rows[:limit]
    expected = [row["reaction_identity"] for row in panel]
    if len(set(expected)) != len(expected):
        raise ValueError("panel reaction identities must be unique")

    by_identity: dict[str, dict] = {}
    input_rows = 0
    for path in prediction_inputs:
        for row in read_jsonl(path):
            input_rows += 1
            identity = row["reaction_identity"]
            if identity in by_identity:
                raise ValueError(f"duplicate prediction identity: {identity}")
            by_identity[identity] = row
    missing = [identity for identity in expected if identity not in by_identity]
    if missing:
        raise ValueError(f"prediction inputs miss {len(missing)} panel identities")
    return panel, [by_identity[identity] for identity in expected], input_rows - limit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter and order prediction shards against a frozen manifest prefix",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prediction-input", type=Path, action="append", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-predictions", type=Path, required=True)
    args = parser.parse_args()
    panel, predictions, ignored = subset_panel(
        args.manifest, args.prediction_input, args.limit,
    )
    write_jsonl(args.output_manifest, panel)
    write_jsonl(args.output_predictions, predictions)
    print(json.dumps({
        "manifest": str(args.output_manifest),
        "predictions": str(args.output_predictions),
        "reactions": len(panel),
        "ignored_out_of_panel_predictions": ignored,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
