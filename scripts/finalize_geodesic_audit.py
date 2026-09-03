#!/usr/bin/env python
"""Write a compact integrity/provenance manifest for the geodesic audit."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(root: Path) -> None:
    root = root.resolve()
    raw = []
    for path in sorted((root / "raw").glob("*.jsonl.gz")):
        # Reading through EOF verifies every concatenated gzip member.
        with gzip.open(path, "rb") as handle:
            while handle.read(8 << 20):
                pass
        raw.append({
            "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    derived = []
    for folder in (root / "analysis", root / "plots"):
        if not folder.exists():
            continue
        for path in sorted(file for file in folder.rglob("*") if file.is_file()):
            derived.append({
                "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    sources = []
    for relative in (
        "src/geodesic_audit.py", "scripts/run_geodesic_audit.py",
        "scripts/summarize_geodesic_audit.py",
        "docs/preregistrations/GEODESIC_MECHANISM_AUDIT_PROTOCOL.md",
    ):
        path = ROOT / relative
        sources.append({"path": relative, "sha256": sha256(path)})
    payload = {
        "type": "geodesic_mechanism_audit_manifest",
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
        ).strip(),
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "performance_validation": {
            "selected_capture_reference_seconds": 0.2130007,
            "selected_capture_optimized_seconds": 0.1209637,
            "selected_capture_speedup": 1.7609,
            "selected_capture_reference_peak_bytes": 2336975360,
            "selected_capture_optimized_peak_bytes": 2185980416,
            "candidate_reference_observed_seconds_approx": 5582.0,
            "candidate_reference_rows": 15270,
            "candidate_batched_seconds": 855.221,
            "candidate_batched_rows": 11475,
            "candidate_combined_speedup_approx": 6.526,
            "candidate_per_record_speedup_approx": 4.906,
        },
        "raw": raw, "derived": derived, "sources": sources,
        "excluded_from_compact_archive": [
            "BF16 gold-state caches (reconstructable)",
            "superseded/interrupted raw streams",
            "base model and existing checkpoints",
        ],
    }
    destination = root / "artifact_manifest.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(destination), "raw_files": len(raw), "derived_files": len(derived)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "runs" / "geodesic_mechanism_audit")
    run(parser.parse_args().root)
