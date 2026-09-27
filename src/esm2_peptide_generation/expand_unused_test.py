from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .freeze_evaluation import make_fixed_decoys
from .prepare_data import sha256_file
from .split_data import assign_clusters, _read_jsonl, _write_jsonl


def expand_test_with_unused_clusters(
    *, pairs_path: Path, clusters_path: Path, manifest_dir: Path, output_dir: Path,
    seed: int = 20260926, minimum_clusters: int = 1,
) -> dict:
    pairs, clusters = _read_jsonl(pairs_path), _read_jsonl(clusters_path)
    train = _read_jsonl(manifest_dir / "train.jsonl")
    validation = _read_jsonl(manifest_dir / "validation.jsonl")
    test = _read_jsonl(manifest_dir / "test.jsonl")
    summary = json.loads((manifest_dir / "frozen_split_summary.json").read_text(encoding="utf-8"))
    assignments = assign_clusters(pairs, clusters, seed=int(summary["seed"]))
    cluster_for_receptor = {row["receptor_id"]: row["receptor_cluster_id"] for row in clusters}
    existing_test_clusters = {row["receptor_cluster_id"] for row in test}
    used_clusters = {
        row["receptor_cluster_id"] for row in train + validation + test
    }
    train_peptides = {row["peptide_sequence"] for row in train}
    train_ids = {row["pair_id"] for row in train}
    test_ids = {row["pair_id"] for row in test}
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in pairs:
        cluster_id = cluster_for_receptor[row["receptor_id"]]
        if (
            assignments[row["receptor_id"]] == "train"
            and row["pair_id"] not in train_ids
            and cluster_id not in used_clusters
            and cluster_id not in existing_test_clusters
            and row["peptide_sequence"] not in train_peptides
            and row["pair_id"] not in test_ids
        ):
            groups[cluster_id].append({**row, "receptor_cluster_id": cluster_id, "split": "test"})
    added = [row for cluster_id in sorted(groups) for row in groups[cluster_id]]
    if len(groups) < minimum_clusters:
        return {
            "added_pair_count": 0,
            "added_receptor_cluster_count": len(groups),
            "insufficient_unused_clusters": True,
            "candidate_cluster_pair_counts": {key: len(value) for key, value in groups.items()},
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    expanded_test = sorted(test + added, key=lambda row: row["pair_id"])
    expanded_path = output_dir / "test_expanded_unused_clusters.jsonl"
    digest = _write_jsonl(expanded_path, expanded_test)
    decoy_report = make_fixed_decoys(
        expanded_path, output_dir / "test_expanded_unused_clusters_decoys.jsonl", count=4, seed=seed
    )
    report = {
        "source_test_sha256": sha256_file(manifest_dir / "test.jsonl"),
        "source_train_sha256": sha256_file(manifest_dir / "train.jsonl"),
        "source_pair_manifest_sha256": sha256_file(pairs_path),
        "source_cluster_manifest_sha256": sha256_file(clusters_path),
        "expanded_test_sha256": digest,
        "added_pair_count": len(added),
        "added_receptor_cluster_count": len(groups),
        "expanded_test_pair_count": len(expanded_test),
        "expanded_test_receptor_cluster_count": len({row["receptor_cluster_id"] for row in expanded_test}),
        "added_clusters_have_no_pairs_in_train_validation_or_test_manifests": True,
        "train_test_peptide_overlap": len(
            train_peptides & {row["peptide_sequence"] for row in expanded_test}
        ),
        "fixed_decoys": decoy_report,
        "manifest": expanded_path.name,
    }
    if report["train_test_peptide_overlap"] != 0:
        raise RuntimeError("expanded test introduced a train/test exact-peptide collision")
    (output_dir / "expanded_test_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand Gate-1 detectability test with unused receptor clusters")
    parser.add_argument("--pairs", type=Path, default=Path("data/protein/propedia26/unique_pairs.jsonl"))
    parser.add_argument("--clusters", type=Path, default=Path("data/protein/propedia26/clustering/receptor_clusters.jsonl"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/protein/propedia26/manifests"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-clusters", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(expand_test_with_unused_clusters(
        pairs_path=args.pairs, clusters_path=args.clusters, manifest_dir=args.manifest_dir,
        output_dir=args.output_dir, minimum_clusters=args.minimum_clusters,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
