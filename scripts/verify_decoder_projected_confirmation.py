#!/usr/bin/env python3
"""Verify the outcome-blind second 640 confirmation panel and prerequisites.

This check is intentionally read-only.  It prevents a confirmation launch when
the required same-seed Native checkpoints or archived controls are unavailable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "data/clm_jepa_uspto_mit_stp_confirmation/untouched_1280.jsonl"
USED = ROOT / "data/clm_jepa_uspto_mit_stp_confirmation/untouched_640.jsonl"
AUDIT = ROOT / "data/clm_jepa_uspto_mit_latent_audit/splits.json"
OUT = ROOT / "runs/decoder_projected/confirmation_prerequisites.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    lines = PARENT.read_bytes().splitlines(keepends=True)
    parent = [json.loads(line) for line in lines if line.strip()]
    used = load_lines(USED)
    second = parent[640:1280]
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))["records"]
    second_pairs = {row["chemical_pair_id"] for row in second}
    used_pairs = {row["chemical_pair_id"] for row in used}
    audit_pairs = {row["chemical_pair_id"] for row in audit}
    native_checkpoints = {}
    for seed in (2027, 3163):
        candidates = list((ROOT / "runs").rglob(f"seed_{seed}"))
        native_checkpoints[str(seed)] = [str(path) for path in candidates if "native" in str(path).lower()]
    payload = {
        "parent_manifest": str(PARENT),
        "parent_manifest_sha256": sha256_bytes(PARENT.read_bytes()),
        "panel_definition": "exact byte-preserving lines 641-1280 (zero-based slice 640:1280)",
        "panel_reactions": len(second),
        "panel_indices": [second[0]["panel_index"], second[-1]["panel_index"]],
        "panel_sha256": sha256_bytes(b"".join(lines[640:1280])),
        "panel_pair_count": len(second_pairs),
        "overlap_with_used_first_640": len(second_pairs & used_pairs),
        "overlap_with_report03_latent_audit": len(second_pairs & audit_pairs),
        "native_checkpoints_for_confirmation_seeds": native_checkpoints,
        "native_checkpoint_prerequisite_satisfied": all(native_checkpoints[str(seed)] for seed in (2027, 3163)),
        "outcome_generation_started": False,
        "status": "blocked_missing_same_seed_native_checkpoints" if not all(native_checkpoints[str(seed)] for seed in (2027, 3163)) else "ready_for_review",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
