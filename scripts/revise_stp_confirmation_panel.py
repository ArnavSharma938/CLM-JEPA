#!/usr/bin/env python
"""Apply the outcome-blind 1,280 -> 640 confirmation-panel amendment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PANEL_ROOT = ROOT / "data/clm_jepa_uspto_mit_stp_confirmation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    source = PANEL_ROOT / "untouched_1280.jsonl"
    destination = PANEL_ROOT / "untouched_640.jsonl"
    source_rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    if len(source_rows) != 1280 or [row["panel_index"] for row in source_rows] != list(range(1280)):
        raise ValueError("original locked 1,280 panel changed")
    if [row["selection_key"] for row in source_rows] != sorted(row["selection_key"] for row in source_rows):
        raise ValueError("original panel is not in outcome-blind selection-key order")
    selected = source_rows[:640]
    destination.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8", newline="\n",
    )
    old_meta = json.loads((PANEL_ROOT / "untouched_1280.metadata.json").read_text(encoding="utf-8"))
    metadata = dict(old_meta)
    metadata.update({
        "schema_version": 2,
        "manifest": "data/clm_jepa_uspto_mit_stp_confirmation/untouched_640.jsonl",
        "manifest_sha256": sha256(destination),
        "requested_reactions": 640,
        "parent_manifest": "data/clm_jepa_uspto_mit_stp_confirmation/untouched_1280.jsonl",
        "parent_manifest_sha256": sha256(source),
        "amendment": "first 640 rows in the already-locked outcome-blind salted selection order",
        "amended_after_training_before_any_confirmation_inference": True,
        "integrity_checks": {
            "reactions": 640, "unique_official_groups": 640,
            "unique_chemical_pairs": 640, "example_ids": 3200,
            "unique_example_ids": 3200, "five_views_each": True,
            "panel_indices_exact": True, "historical_group_overlap": 0,
            "excluded_chemical_pair_overlap": 0,
        },
    })
    # Store repository-relative provenance; the original metadata retains the
    # machine-specific creation paths for the full selection.
    metadata["exclusion_ledger"] = "data/clm_jepa_uspto_mit_stp_confirmation/exclusion_ledger.json"
    write_json(PANEL_ROOT / "untouched_640.metadata.json", metadata)
    prereg_path = PANEL_ROOT / "preregistration.json"
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    prereg.update({
        "schema_version": 2,
        "panel": "data/clm_jepa_uspto_mit_stp_confirmation/untouched_640.jsonl",
        "panel_sha256": sha256(destination),
        "panel_reactions": 640,
        "original_panel": "data/clm_jepa_uspto_mit_stp_confirmation/untouched_1280.jsonl",
        "original_panel_sha256": sha256(source),
        "amendment_timing": "after six primary trajectories trained, before any confirmation-panel inference or outcome inspection",
        "amendment_reason": "explicit user instruction to halve endpoint cost",
        "amendment_selection": "first 640 reactions in the original outcome-blind salted selection order; no model output used",
    })
    write_json(prereg_path, prereg)
    print(json.dumps({"panel": str(destination), "sha256": sha256(destination), "reactions": 640}, sort_keys=True))


if __name__ == "__main__":
    main()
