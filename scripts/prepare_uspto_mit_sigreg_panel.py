"""Create the fixed length-stratified USPTO-MIT SIGReg panel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemfm import END, PRODUCT_START, REACTANT_START, TOKENIZER_DIR, load_reaction_tokenizer
from train import file_sha256

SEED = 533
PANEL_SIZE = 256
STRATA = 8


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_panel(source: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_csv = output_dir / "uspto_mit_validation_length_stratified_256.csv"
    panel_jsonl = output_dir / "uspto_mit_validation_length_stratified_256.jsonl"
    manifest_path = output_dir / "manifest.json"
    if any(path.exists() for path in (panel_csv, panel_jsonl, manifest_path)):
        if not all(path.exists() for path in (panel_csv, panel_jsonl, manifest_path)):
            raise RuntimeError("partial panel artifacts exist; refusing to overwrite")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            file_sha256(panel_csv) != manifest["panel_csv_sha256"]
            or file_sha256(panel_jsonl) != manifest["panel_jsonl_sha256"]
        ):
            raise RuntimeError("existing panel does not match its frozen manifest")
        return manifest

    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < PANEL_SIZE:
        raise ValueError("source panel is smaller than the requested panel")
    identities = [row["reaction_identity"] for row in rows]
    targets = [row["target_identity"] for row in rows]
    if len(identities) != len(set(identities)) or len(targets) != len(set(targets)):
        raise ValueError("source panel must contain unique reaction and target identities")

    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    source_texts = [
        f"{REACTANT_START}{row['source']}{END}{PRODUCT_START}" for row in rows
    ]
    target_texts = [f"{PRODUCT_START}{row['target']}{END}" for row in rows]
    source_lengths = np.asarray([
        len(ids) for ids in tokenizer(source_texts, add_special_tokens=False)["input_ids"]
    ])
    target_lengths = np.asarray([
        len(ids) for ids in tokenizer(target_texts, add_special_tokens=False)["input_ids"]
    ])

    # Equal-frequency source-length strata ensure short through long reactions are
    # represented. Sampling is seeded and depends only on inputs/lengths, never outputs.
    order = np.lexsort((np.asarray(identities), source_lengths))
    bins = np.array_split(order, STRATA)
    per_stratum = PANEL_SIZE // STRATA
    rng = random.Random(SEED)
    selected_indices: list[int] = []
    audits = []
    for stratum, indices in enumerate(bins):
        candidates = list(map(int, indices))
        chosen = rng.sample(candidates, per_stratum)
        selected_indices.extend(chosen)
        audits.append({
            "stratum": stratum,
            "candidate_count": len(candidates),
            "selected_count": len(chosen),
            "source_token_min": int(source_lengths[indices].min()),
            "source_token_max": int(source_lengths[indices].max()),
        })
    rng.shuffle(selected_indices)

    selected = []
    reference = []
    for panel_index, source_index in enumerate(selected_indices):
        row = dict(rows[source_index])
        row["pilot_panel_index"] = str(panel_index)
        row["source_token_count"] = str(int(source_lengths[source_index]))
        row["target_token_count"] = str(int(target_lengths[source_index]))
        selected.append(row)
        reference.append({
            "panel_index": panel_index,
            "reaction_identity": row["reaction_identity"],
        })
    _write_csv(panel_csv, selected)
    panel_jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in reference),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "created_before_checkpoint_evaluation": True,
        "dataset": "uspto_mit_synthesis",
        "split": "validation",
        "seed": SEED,
        "selection": (
            "seeded random sample of 32 identities from each of eight equal-frequency "
            "source-prompt token-length strata in the frozen 1,024-identity validation panel"
        ),
        "source_panel": str(source.resolve()),
        "source_panel_sha256": file_sha256(source),
        "source_identities": len(rows),
        "selected_identities": len(selected),
        "source_length_strata": audits,
        "selected_source_token_summary": {
            "min": int(source_lengths[selected_indices].min()),
            "median": float(np.median(source_lengths[selected_indices])),
            "max": int(source_lengths[selected_indices].max()),
        },
        "selected_target_token_summary": {
            "min": int(target_lengths[selected_indices].min()),
            "median": float(np.median(target_lengths[selected_indices])),
            "max": int(target_lengths[selected_indices].max()),
        },
        "panel_csv": str(panel_csv.resolve()),
        "panel_csv_sha256": file_sha256(panel_csv),
        "panel_jsonl": str(panel_jsonl.resolve()),
        "panel_jsonl_sha256": file_sha256(panel_jsonl),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the SIGReg batch-16 pilot panel")
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data/clm_jepa_uspto_mit_validation_1024/uspto_mit_validation_1024.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/clm_jepa_uspto_mit_validation_256",
    )
    args = parser.parse_args()
    print(json.dumps(select_panel(args.source.resolve(), args.output_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
