#!/usr/bin/env python
"""Resumable A6000 orchestration for the preregistered frozen audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIRMATION = (
    ROOT / "data/clm_jepa_uspto_mit_stp_confirmation/untouched_1280.jsonl"
)


def lines(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "runs/latent_predictability_audit")
    parser.add_argument("--confirmation-manifest", type=Path, default=DEFAULT_CONFIRMATION)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = args.output.resolve()
    common = [
        "--confirmation-manifest", str(args.confirmation_manifest.resolve()),
        "--output", str(output), "--device", args.device,
    ]
    runner = [sys.executable, "-u", "scripts/run_latent_predictability_audit.py"]
    stages = [
        ("extract", runner + ["extract", *common, "--batch-size", "8"],
         lambda: len(list((output / "cache/canonical").glob("*.pt"))) == 8),
        ("fit_probes", runner + ["fit-probes", *common, "--resume"],
         lambda: lines(output / "raw/probe_metrics.jsonl") == 512),
        ("decoder", runner + [
            "score-decoder", *common,
            "--decoder-reactions", "64",
            "--decoder-positions", "96",
            "--decoder-rare-positions", "32",
        ],
         lambda: lines(output / "raw/decoder_metrics.jsonl") == 1536),
        ("extract_views", runner + ["extract-views", *common, "--batch-size", "8"],
         lambda: len(list((output / "cache/views").glob("*.pt"))) == 8),
        ("analyze_views", runner + ["analyze-views", *common],
         lambda: lines(output / "raw/invariance.jsonl") == 384),
        ("extract_candidates", runner + ["extract-candidates", *common, "--batch-size", "8"],
         lambda: len(list((output / "cache/candidates").glob("*.pt"))) == 8),
        ("score_candidates", runner + ["score-candidates", *common],
         lambda: (output / "raw/candidate_predictability.jsonl").exists()),
        ("development_views", runner + ["extract-development-views", *common, "--batch-size", "8"],
         lambda: len(list((output / "raw/development_invariance").glob("*.jsonl"))) == 8),
        ("summarize", runner + ["summarize", *common],
         lambda: False),
        ("analyze", [sys.executable, "-u", "scripts/analyze_latent_predictability_audit.py", "--input", str(output)],
         lambda: False),
    ]
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "execution.json"
    metadata = {"type": "latent_audit_execution", "stages": []}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for name, command, complete in stages:
        if complete():
            print(json.dumps({"event": "stage_reused", "stage": name}), flush=True)
            continue
        start = time.perf_counter()
        print(json.dumps({"event": "stage_start", "stage": name, "command": command}), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        record = {"stage": name, "wall_seconds": time.perf_counter() - start}
        metadata["stages"].append(record)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"event": "stage_complete", **record}), flush=True)
    print(json.dumps({"event": "audit_complete"}), flush=True)


if __name__ == "__main__":
    main()
