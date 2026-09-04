#!/usr/bin/env python3
"""Freeze or validate the untouched official-test STP confirmation panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from confirmation_design import build_untouched_panel, validate_panel  # noqa: E402


DEFAULT_ROOT = ROOT / "data/clm_jepa_uspto_mit_stp_confirmation"


def paths(root: Path) -> tuple[Path, Path, Path]:
    return root / "untouched_1280.jsonl", root / "untouched_1280.metadata.json", root / "exclusion_ledger.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--processes", type=int, default=6)
    args = parser.parse_args()
    panel, metadata, exclusion = paths(args.output_root)
    if args.command == "build":
        result = build_untouched_panel(
            official_test=ROOT / "data/uspto_mit_synthesis/test_r_smiles.csv",
            historical_manifests=[
                ROOT / "data/clm_jepa_uspto_mit_official_endpoint/prespecified_3300.jsonl",
                ROOT / "data/clm_jepa_uspto_mit_official_endpoint/equivalence_24.jsonl",
            ],
            excluded_csvs=[
                ROOT / "data/clm_jepa_uspto_mit_pilot_1280/uspto_mit_train.csv",
                ROOT / "data/clm_jepa_uspto_mit_validation_1024/uspto_mit_validation_1024.csv",
                ROOT / "data/clm_jepa_uspto_mit_validation_256/uspto_mit_validation_length_stratified_256.csv",
            ],
            output=panel,
            metadata_output=metadata,
            exclusion_output=exclusion,
            processes=args.processes,
        )
    else:
        result = validate_panel(panel, metadata, exclusion)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
