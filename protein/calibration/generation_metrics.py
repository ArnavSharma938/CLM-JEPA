from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


EXTERNAL_COLUMNS = {
    "sample_id",
    "rosetta_generated",
    "rosetta_native",
    "esmfold_plddt",
    "tm_to_conditioning_backbone",
    "nearest_training_sequence_identity",
    "sequence_cluster_id",
}


def prepare(generation_root: Path, output_dir: Path) -> None:
    rows = []
    for path in sorted(generation_root.rglob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise ValueError("No generation JSONL files found")
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate generated sample_id")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "generated.fasta").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f">{row['sample_id']}\n{row['generated_sequence']}\n")
    pd.DataFrame(rows).to_parquet(output_dir / "generation_manifest.parquet", index=False)
    (output_dir / "README.txt").write_text(
        "Return external_metrics.csv with: sample_id, rosetta_generated, rosetta_native, "
        "esmfold_plddt, tm_to_conditioning_backbone, nearest_training_sequence_identity, "
        "sequence_cluster_id. Rosetta scores must use the same preregistered protocol for native "
        "and generated sequences. The cluster definition/threshold must be recorded separately.\n",
        encoding="utf-8",
    )


def _mean_pairwise_identity(sequences: list[str]) -> float:
    values = []
    for left, right in combinations(sequences, 2):
        if len(left) != len(right):
            raise ValueError("Generated sequence length changed despite fixed-backbone generation")
        values.append(sum(a == b for a, b in zip(left, right)) / len(left))
    return float(np.mean(values)) if values else np.nan


def ingest(manifest_path: Path, external_path: Path, output_dir: Path) -> None:
    manifest = pd.read_parquet(manifest_path)
    external = pd.read_csv(external_path)
    missing = EXTERNAL_COLUMNS - set(external.columns)
    if missing:
        raise ValueError(f"External generation results are missing {sorted(missing)}")
    if external["sample_id"].duplicated().any() or set(external["sample_id"]) != set(manifest["sample_id"]):
        raise ValueError("External sample IDs are not a one-to-one match to generation manifest")
    frame = manifest.merge(external, on="sample_id", validate="one_to_one")
    frame["delta_energy"] = frame["rosetta_generated"] - frame["rosetta_native"]
    group_columns = ["model", "training_seed", "checkpoint", "domain"]
    summaries = []
    for keys, group in frame.groupby(group_columns, sort=True):
        summaries.append(
            dict(
                zip(group_columns, keys),
                n=len(group),
                median_delta_energy=float(group["delta_energy"].median()),
                fraction_delta_energy_below_zero=float((group["delta_energy"] < 0).mean()),
                mean_plddt=float(group["esmfold_plddt"].mean()),
                mean_tm_score=float(group["tm_to_conditioning_backbone"].mean()),
                mean_nearest_training_identity=float(group["nearest_training_sequence_identity"].mean()),
                mean_pairwise_identity=_mean_pairwise_identity(group["generated_sequence"].tolist()),
                unique_fraction=float(group["generated_sequence"].nunique() / len(group)),
                sequence_cluster_count=int(group["sequence_cluster_id"].nunique()),
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_dir / "generation_metrics.parquet", index=False)
    pd.DataFrame(summaries).to_parquet(output_dir / "backbone_summaries.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("--generation-root", required=True, type=Path)
    prep.add_argument("--output-dir", required=True, type=Path)
    merge = subparsers.add_parser("ingest")
    merge.add_argument("--manifest", required=True, type=Path)
    merge.add_argument("--external", required=True, type=Path)
    merge.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.generation_root, args.output_dir)
    else:
        ingest(args.manifest, args.external, args.output_dir)


if __name__ == "__main__":
    main()
