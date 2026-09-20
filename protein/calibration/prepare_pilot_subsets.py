from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

FRACTION = 0.25


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_order(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    keys = frame["sequence_id"].map(lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
    return frame.assign(_priority=keys).sort_values(["_priority", "sequence_id"])


def _sample_domain(domain: pd.DataFrame, seed: int) -> pd.DataFrame:
    target = max(domain["backbone_path"].nunique(), round(len(domain) * FRACTION))
    ranked = domain["ddg"].rank(method="first")
    work = domain.assign(
        _ddg_bin=pd.qcut(ranked, q=min(4, len(domain)), labels=False, duplicates="drop").astype(int),
        _stabilizing=domain["ddg"] < 0,
        _mutation_count=domain["mutation_count"].astype(int),
        _length=domain["target_sequence"].str.len().astype(int),
    )
    strata = ["_stabilizing", "_mutation_count", "_ddg_bin", "_length"]
    counts = work.groupby(strata, sort=True).size().rename("count").reset_index()
    counts["ideal"] = counts["count"] * target / len(work)
    counts["quota"] = np.floor(counts["ideal"]).astype(int)
    remaining = target - int(counts["quota"].sum())
    counts["fractional"] = counts["ideal"] - counts["quota"]
    counts = counts.sort_values(strata).sort_values("fractional", ascending=False, kind="stable")
    counts.iloc[:remaining, counts.columns.get_loc("quota")] += 1
    selected = []
    for _, spec in counts.iterrows():
        mask = np.ones(len(work), dtype=bool)
        for column in strata:
            mask &= work[column].to_numpy() == spec[column]
        selected.append(_stable_order(work.loc[mask], seed).head(int(spec["quota"])))
    sample = pd.concat(selected) if selected else work.iloc[:0]
    missing = set(work["backbone_path"]) - set(sample["backbone_path"])
    for backbone in sorted(missing):
        addition = _stable_order(work[work["backbone_path"] == backbone], seed).iloc[[0]]
        sample = pd.concat([sample, addition]).drop_duplicates("sequence_id")
    if len(sample) > target:
        protected = set(sample.groupby("backbone_path", sort=False).head(1)["sequence_id"])
        removable = _stable_order(sample[~sample["sequence_id"].isin(protected)], seed)
        sample = sample.drop(removable.head(len(sample) - target).index)
    return sample.drop(columns=[column for column in sample if column.startswith("_")])


def build_proportional_subset(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    train, heldout = frame[frame["split"] == "train"], frame[frame["split"] != "train"]
    sampled = pd.concat(
        [_sample_domain(group, seed) for _, group in train.groupby("domain_id", sort=True)]
    ).sort_index()
    if set(sampled["domain_id"]) != set(train["domain_id"]):
        raise RuntimeError("Pilot lost training domains")
    if set(sampled["backbone_path"]) != set(train["backbone_path"]):
        raise RuntimeError("Pilot lost training structures")
    result = pd.concat([sampled, heldout]).sort_index().reset_index(drop=True)
    if not heldout.reset_index(drop=True).equals(result[result["split"] != "train"].reset_index(drop=True)):
        raise RuntimeError("Validation/test rows changed")
    return result


def write_subset(manifest: Path, output_dir: Path, seed: int = 42) -> dict:
    frame = pd.read_parquet(manifest)
    subset = build_proportional_subset(frame, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    path_map = {
        value: os.path.relpath(manifest.parent / Path(str(value).replace("\\", "/")), output_dir).replace(os.sep, "/")
        for value in frame["backbone_path"].astype(str).unique()
    }
    subset["backbone_path"] = subset["backbone_path"].astype(str).map(path_map)
    output = output_dir / "megascale_seed42_train_25pct.parquet"
    subset.to_parquet(output, index=False)
    ids_path = output_dir / "megascale_seed42_train_25pct.sequence_ids.json"
    train = subset[subset["split"] == "train"]
    ids_path.write_bytes(json.dumps(sorted(train["sequence_id"].astype(str)), indent=2).encode("utf-8"))
    metadata = json.loads(manifest.with_suffix(manifest.suffix + ".metadata.json").read_text(encoding="utf-8"))
    metadata.update({
        "manifest_sha256": _sha256(output), "pilot_parent_manifest_sha256": _sha256(manifest),
        "pilot_training_fraction_target": FRACTION, "pilot_seed": seed,
        "pilot_selection": "proportional_within_domain_stratified_variant_subsampling",
        "sampled_sequence_ids_sha256": _sha256(ids_path),
    })
    output.with_suffix(output.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    full_train = frame[frame["split"] == "train"]
    report = {
        "parent_manifest_sha256": _sha256(manifest), "manifest_sha256": _sha256(output),
        "sampled_sequence_ids_sha256": _sha256(ids_path), "seed": seed,
        "training_records": len(train), "training_fraction_observed": len(train) / len(full_train),
        "training_domains": train["domain_id"].nunique(), "full_training_domains": full_train["domain_id"].nunique(),
        "training_structures": train["backbone_path"].nunique(), "full_training_structures": full_train["backbone_path"].nunique(),
        "stabilizing_fraction": float((train["ddg"] < 0).mean()),
        "full_stabilizing_fraction": float((full_train["ddg"] < 0).mean()),
    }
    (output_dir / "pilot_25pct_provenance.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(write_subset(args.manifest, args.output_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()
