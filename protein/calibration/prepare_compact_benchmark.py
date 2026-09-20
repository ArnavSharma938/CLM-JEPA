from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePath

import numpy as np
import pandas as pd

TRAIN_DOMAINS = 36
TRAIN_PER_DOMAIN = 306
VALIDATION_DOMAINS = 6
TEST_DOMAINS = 9
HELDOUT_PER_DOMAIN = 160


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _bins(series: pd.Series, count: int = 4) -> pd.Series:
    return pd.qcut(series.rank(method="first"), min(count, len(series)), labels=False, duplicates="drop").astype(int)


def _stats(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain, group in frame.groupby("domain_id", sort=True):
        rows.append({
            "domain_id": domain, "cluster": group["structural_cluster"].iloc[0],
            "length": int(group["target_sequence"].str.len().median()),
            "double_fraction": float((group["mutation_count"] == 2).mean()),
            "ddg_range": float(group["ddg"].max() - group["ddg"].min()),
            "stabilizing": int((group["ddg"] < 0).sum()),
        })
    result = pd.DataFrame(rows)
    for column in ("length", "double_fraction", "ddg_range"):
        result[f"{column}_bin"] = _bins(result[column])
    return result


def _balanced_train_domains(stats: pd.DataFrame, seed: int) -> list[str]:
    stats = stats[stats["stabilizing"] >= TRAIN_PER_DOMAIN].copy()
    strata_cols = ["length_bin", "double_fraction_bin", "ddg_range_bin"]
    groups = {
        key: sorted(group.to_dict("records"), key=lambda row: _stable(row["domain_id"], seed))
        for key, group in stats.groupby(strata_cols, sort=True)
    }
    selected, clusters = [], set()
    while len(selected) < TRAIN_DOMAINS:
        progress = False
        for key in sorted(groups):
            while groups[key] and groups[key][0]["cluster"] in clusters:
                groups[key].pop(0)
            if groups[key] and len(selected) < TRAIN_DOMAINS:
                row = groups[key].pop(0)
                selected.append(row["domain_id"]); clusters.add(row["cluster"]); progress = True
        if not progress:
            break
    if len(selected) != TRAIN_DOMAINS:
        raise RuntimeError(f"Could select only {len(selected)} unique-cluster training domains")
    return selected


def _balanced_heldout_domains(stats: pd.DataFrame, count: int, seed: int) -> list[str]:
    by_cluster = {
        cluster: sorted(group.to_dict("records"), key=lambda row: (
            row["length_bin"], row["double_fraction_bin"], row["ddg_range_bin"],
            _stable(row["domain_id"], seed),
        ))
        for cluster, group in stats.groupby("cluster", sort=True)
    }
    selected = []
    while len(selected) < count:
        progress = False
        for cluster in sorted(by_cluster):
            if by_cluster[cluster] and len(selected) < count:
                selected.append(by_cluster[cluster].pop(0)["domain_id"]); progress = True
        if not progress:
            raise RuntimeError(f"Could select only {len(selected)} held-out domains")
    return selected


def _sample_variants(group: pd.DataFrame, count: int, seed: int, stabilizing_only: bool) -> pd.DataFrame:
    work = group[group["ddg"] < 0].copy() if stabilizing_only else group.copy()
    count = min(count, len(work))
    work["_ddg_bin"] = _bins(work["ddg"])
    strata = ["mutation_count", "_ddg_bin"]
    sizes = work.groupby(strata, sort=True).size().rename("n").reset_index()
    sizes["ideal"] = sizes["n"] * count / len(work)
    sizes["quota"] = np.floor(sizes["ideal"]).astype(int)
    remaining = count - int(sizes["quota"].sum())
    sizes["fraction"] = sizes["ideal"] - sizes["quota"]
    sizes = sizes.sort_values(strata).sort_values("fraction", ascending=False, kind="stable")
    sizes.iloc[:remaining, sizes.columns.get_loc("quota")] += 1
    pieces = []
    for _, row in sizes.iterrows():
        mask = np.ones(len(work), dtype=bool)
        for column in strata:
            mask &= work[column].to_numpy() == row[column]
        part = work.loc[mask].copy()
        part["_key"] = part["sequence_id"].map(lambda value: _stable(value, seed))
        pieces.append(part.sort_values(["ddg", "_key"]).iloc[
            np.linspace(0, len(part) - 1, int(row["quota"])).round().astype(int)
        ] if row["quota"] else part.iloc[:0])
    return pd.concat(pieces).drop(columns=["_ddg_bin", "_key"]).sort_index()


def build_compact(frame: pd.DataFrame, clusters: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    cluster_map = dict(zip(clusters["member"].str.removesuffix(".pdb"), clusters["representative"]))
    structure = frame["backbone_path"].astype(str).map(lambda value: PurePath(value.replace("\\", "/")).stem)
    frame = frame.assign(structural_cluster=structure.map(cluster_map), _row=np.arange(len(frame)))
    if frame["structural_cluster"].isna().any():
        raise RuntimeError("Cluster mapping did not cover the manifest")
    train_ids = _balanced_train_domains(_stats(frame[frame.split == "train"]), seed)
    validation_ids = _balanced_heldout_domains(_stats(frame[frame.split == "validation"]), VALIDATION_DOMAINS, seed)
    test_ids = _balanced_heldout_domains(_stats(frame[frame.split == "test"]), TEST_DOMAINS, seed)
    pieces = []
    for split, identifiers, cap, stabilizing in (
        ("train", train_ids, TRAIN_PER_DOMAIN, True),
        ("validation", validation_ids, HELDOUT_PER_DOMAIN, False),
        ("test", test_ids, HELDOUT_PER_DOMAIN, False),
    ):
        source = frame[(frame.split == split) & frame.domain_id.isin(identifiers)]
        pieces.extend(_sample_variants(group, cap, seed, stabilizing) for _, group in source.groupby("domain_id", sort=True))
    compact = pd.concat(pieces).sort_values("_row").drop(columns=["structural_cluster", "_row"]).reset_index(drop=True)
    selected_clusters = {
        split: set(frame[frame.domain_id.isin(ids)]["structural_cluster"])
        for split, ids in (("train", train_ids), ("validation", validation_ids), ("test", test_ids))
    }
    if any(selected_clusters[a] & selected_clusters[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise RuntimeError("Structural cluster leakage in compact benchmark")
    return compact, {
        "domains": {"train": train_ids, "validation": validation_ids, "test": test_ids},
        "cluster_counts": {key: len(value) for key, value in selected_clusters.items()},
    }


def write_compact(manifest: Path, cluster_tsv: Path, output_dir: Path, seed: int = 42) -> dict:
    frame = pd.read_parquet(manifest)
    clusters = pd.read_csv(cluster_tsv, sep="\t", header=None, names=["representative", "member"])
    compact, selection = build_compact(frame, clusters, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    path_map = {value: os.path.relpath(manifest.parent / Path(str(value).replace("\\", "/")), output_dir).replace(os.sep, "/") for value in compact.backbone_path.astype(str).unique()}
    compact["backbone_path"] = compact.backbone_path.astype(str).map(path_map)
    output = output_dir / "megascale_gate1_compact_seed42.parquet"
    compact.to_parquet(output, index=False)
    ids_path = output_dir / "megascale_gate1_compact_seed42.sequence_ids.json"
    ids = {split: sorted(compact[compact.split == split].sequence_id.astype(str)) for split in ("train", "validation", "test")}
    ids_path.write_bytes(json.dumps(ids, indent=2, sort_keys=True).encode())
    metadata = json.loads(manifest.with_suffix(manifest.suffix + ".metadata.json").read_text(encoding="utf-8"))
    counts = {split: int((compact.split == split).sum()) for split in ("train", "validation", "test")}
    metadata.update({
        "manifest_sha256": _sha(output), "compact_parent_manifest_sha256": _sha(manifest),
        "compact_cluster_tsv_sha256": _sha(cluster_tsv), "compact_seed": seed,
        "compact_selection": selection, "compact_counts": counts,
        "compact_sequence_ids_sha256": _sha(ids_path),
    })
    output.with_suffix(output.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    balance = {}
    for split in ("train", "validation", "test"):
        group = compact[compact.split == split]
        lengths = group.target_sequence.str.len()
        balance[split] = {
            "domains": int(group.domain_id.nunique()), "structures": int(group.backbone_path.nunique()),
            "length_min_median_max": [int(lengths.min()), float(lengths.median()), int(lengths.max())],
            "single_mutations": int((group.mutation_count == 1).sum()),
            "double_mutations": int((group.mutation_count == 2).sum()),
            "ddg_min_median_max": [float(group.ddg.min()), float(group.ddg.median()), float(group.ddg.max())],
        }
    report = {"manifest_sha256": _sha(output), "sequence_ids_sha256": _sha(ids_path), "counts": counts, "balance": balance, **selection}
    (output_dir / "compact_gate1_provenance.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--cluster-tsv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(write_compact(args.manifest, args.cluster_tsv, args.output_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()
