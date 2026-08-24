"""Create an identity-checked prefix panel and aligned prediction slices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument(
        "--predictions", type=Path, nargs=2, action="append", default=[],
        metavar=("INPUT", "OUTPUT"),
    )
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("limit must be positive")

    full_manifest = read_jsonl(args.manifest)
    if len(full_manifest) < args.limit:
        raise ValueError("manifest is smaller than the requested prefix")
    manifest = full_manifest[: args.limit]
    identities = [row["reaction_identity"] for row in manifest]
    if len(set(identities)) != len(identities):
        raise ValueError("prefix manifest contains duplicate reaction identities")
    if [row.get("panel_index") for row in manifest] != list(range(args.limit)):
        raise ValueError("prefix is not the frozen panel-index 0..limit-1 sequence")
    write_jsonl_atomic(args.output_manifest, manifest)

    slices = []
    for input_path, output_path in args.predictions:
        rows = read_jsonl(input_path)
        by_identity = {}
        for row in rows:
            identity = row["reaction_identity"]
            if identity in by_identity:
                raise ValueError(f"duplicate prediction identity in {input_path}: {identity}")
            by_identity[identity] = row
        missing = [identity for identity in identities if identity not in by_identity]
        if missing:
            raise ValueError(f"{input_path} is missing {len(missing)} prefix identities")
        selected = [by_identity[identity] for identity in identities]
        write_jsonl_atomic(output_path, selected)
        slices.append({
            "input": str(input_path.resolve()),
            "input_rows": len(rows),
            "output": str(output_path.resolve()),
            "output_rows": len(selected),
            "output_sha256": sha256(output_path),
        })

    metadata = {
        "schema_version": 1,
        "selection": "first 256 rows of the pre-existing frozen prespecified_stage1_1280 manifest",
        "limit": args.limit,
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256(args.manifest),
        "output_manifest": str(args.output_manifest.resolve()),
        "output_manifest_sha256": sha256(args.output_manifest),
        "reaction_identities": identities,
        "prediction_slices": slices,
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest_rows": len(manifest), "prediction_slices": len(slices)}))


if __name__ == "__main__":
    main()
