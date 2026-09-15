#!/usr/bin/env python
"""Lock and materialize the preregistered 14-assay ProteinGym v1.3 panel."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd

AA = set("ACDEFGHIKLMNPQRSTVWY")
SALT = "rita-panel-v1"


def lock(reference: Path, output: Path, size: int = 14):
    table = pd.read_csv(reference)
    eligible = table[
        table.seq_len.between(64, 512)
        & table.target_seq.map(lambda sequence: set(str(sequence)) <= AA)
        & (table.DMS_number_single_mutants >= 100)
    ].copy()
    eligible["deterministic_rank"] = eligible.DMS_id.map(
        lambda identifier: hashlib.sha256(f"{SALT}\0{identifier}".encode()).hexdigest()
    )
    eligible = eligible.sort_values("deterministic_rank").drop_duplicates("UniProt_ID")
    categories = sorted(eligible.coarse_selection_type.unique())
    chosen, index = [], 0
    while len(chosen) < size:
        category = categories[index % len(categories)]
        candidates = eligible[(eligible.coarse_selection_type == category)
                              & ~eligible.DMS_id.isin([row.DMS_id for row in chosen])]
        if len(candidates): chosen.append(next(candidates.itertuples()))
        index += 1
    keys = (
        "DMS_id", "DMS_filename", "UniProt_ID", "target_seq", "seq_len",
        "DMS_number_single_mutants", "coarse_selection_type", "selection_type",
        "taxon", "ProteinGym_version", "deterministic_rank",
    )
    rows = []
    for row in chosen:
        clean = {}
        for key in keys:
            value = getattr(row, key)
            clean[key] = None if pd.isna(value) else value.item() if hasattr(value, "item") else value
        rows.append(clean)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"selection_salt": SALT, "selected_before_surrogate_analysis": True,
        "eligibility": "canonical target; 64<=length<=512; >=100 single substitutions; distinct UniProt",
        "round_robin_categories": categories, "assays": rows}, indent=2) + "\n", encoding="utf-8")


def materialize(panel: Path, archive: Path, output: Path, variants: int = 256):
    specification = json.loads(panel.read_text())
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        for assay in specification["assays"]:
            filename = assay["DMS_filename"]
            matches = [name for name in names if name.endswith("/" + filename) or name == filename]
            if len(matches) != 1: raise RuntimeError(f"archive match failure for {filename}: {matches}")
            with bundle.open(matches[0]) as handle: table = pd.read_csv(handle)
            table = table[table.mutant.astype(str).str.count(":") == 0]
            table = table[table.mutated_sequence.map(lambda sequence: set(str(sequence)) <= AA)]
            # Stable variant rank, independent of archive ordering and phenotype.
            table["_rank"] = table.mutant.map(lambda mutant: hashlib.sha256(
                f"rita-variants-v1\0{assay['DMS_id']}\0{mutant}".encode()).hexdigest())
            table.sort_values("_rank").head(variants).drop(columns="_rank").to_csv(
                output / filename, index=False)
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.glob("*.csv"))}
    (output / "manifest.json").write_text(json.dumps({"panel": str(panel), "variants_per_assay_cap": variants,
        "subsampling": "SHA256(rita-variants-v1, assay, mutant), before scores are inspected",
        "file_sha256": hashes}, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("lock"); p.add_argument("reference", type=Path); p.add_argument("output", type=Path)
    p = sub.add_parser("materialize"); p.add_argument("panel", type=Path); p.add_argument("archive", type=Path); p.add_argument("output", type=Path)
    args = parser.parse_args()
    lock(args.reference, args.output) if args.stage == "lock" else materialize(args.panel, args.archive, args.output)


if __name__ == "__main__": main()
