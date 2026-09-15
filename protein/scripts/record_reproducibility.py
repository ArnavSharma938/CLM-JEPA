#!/usr/bin/env python
"""Record the environment and immutable inputs for the RITA diagnostic."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    block = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            block.update(chunk)
    return block.hexdigest()


def main() -> None:
    packages = {}
    for name in ("torch", "transformers", "numpy", "scipy", "scikit-learn", "pandas", "pytest"):
        packages[name] = importlib.metadata.version(name)
    paths = [
        ROOT / "data/uniref50_candidates.jsonl",
        ROOT / "data/mmseqs30_cluster.tsv",
        ROOT / "data/diagnostic_pool.jsonl",
        ROOT / "data/tape_structural_manifest.jsonl",
        ROOT / "data/ProteinGym_DMS_substitutions_reference.csv",
        ROOT / "data/DMS_ProteinGym_substitutions.zip",
        ROOT / "data/proteingym_panel.json",
        ROOT / "data/proteingym_panel/manifest.json",
    ]
    record = {
        "created_date": "2026-09-14",
        "model": "lightonai/RITA_m",
        "model_revision": "d819157a3a96d278500232b2cbb2a02d9646bcf6",
        "tokenizer_revision": "d819157a3a96d278500232b2cbb2a02d9646bcf6",
        "random_seed": 20260914,
        "repository_commit_at_start": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT.parent, text=True
        ).strip(),
        "repository_worktree_note": "protein diagnostic is an uncommitted addition at report generation",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "hardware": {
            "local_gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "local_cuda": torch.version.cuda,
            "thunder": "A6000 unavailable; one temporary L40, 6 vCPU, 100 GB, used only for MMseqs2 and deleted",
        },
        "mmseqs": {
            "version": "13-45111+ds-2",
            "command": "easy-cluster --min-seq-id 0.3 -c 0.8 --cov-mode 0 --cluster-mode 2 --threads 6",
        },
        "input_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in paths},
        "cache_performance": {
            "uniref_3000": {"seconds": 84.5293833000469, "peak_vram_bytes": 746777600},
            "tape_684": {"seconds": 23.45763249997981, "peak_vram_bytes": 746617344},
            "reverse_128": {"seconds": 6.327405099989846, "peak_vram_bytes": 746599936},
        },
    }
    destination = ROOT / "runs/reproducibility.json"
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
