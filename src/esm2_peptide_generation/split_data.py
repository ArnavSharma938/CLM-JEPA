from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from .config import CLUSTER_COVERAGE, CLUSTER_IDENTITY
from .prepare_data import sha256_file


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            stream.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def assign_clusters(pairs: list[dict], clusters: list[dict], seed: int = 20260926) -> dict[str, str]:
    pair_counts: dict[str, int] = defaultdict(int)
    for pair in pairs:
        pair_counts[pair["receptor_id"]] += 1
    receptor_to_cluster = {row["receptor_id"]: row["receptor_cluster_id"] for row in clusters}
    if set(pair_counts) != set(receptor_to_cluster):
        raise RuntimeError("receptor sequences and DIAMOND cluster membership differ")
    cluster_counts: dict[str, int] = defaultdict(int)
    for receptor_id, count in pair_counts.items():
        cluster_counts[receptor_to_cluster[receptor_id]] += count
    total = sum(cluster_counts.values())
    targets = {"train": total * 0.8, "validation": total * 0.1, "test": total * 0.1}
    current = dict.fromkeys(targets, 0)
    ordered = sorted(
        cluster_counts,
        key=lambda cluster_id: hashlib.sha256(f"{seed}\0{cluster_id}".encode()).hexdigest(),
    )
    split_for_cluster: dict[str, str] = {}
    for cluster_id in ordered:
        size = cluster_counts[cluster_id]
        def projected_error(split: str) -> tuple[float, str]:
            error = 0.0
            for name, target in targets.items():
                amount = current[name] + (size if name == split else 0)
                error += ((amount - target) / max(target, 1.0)) ** 2
            return error, split
        selected = min(targets, key=lambda split: projected_error(split))
        split_for_cluster[cluster_id] = selected
        current[selected] += size
    return {receptor_id: split_for_cluster[cluster_id] for receptor_id, cluster_id in receptor_to_cluster.items()}


def _fasta(path: Path, pairs: list[dict], split: str) -> None:
    receptors = sorted({
        pair["receptor_id"]: pair["receptor_sequence"]
        for pair in pairs if pair["split"] == split
    }.items())
    with path.open("w", encoding="ascii", newline="\n") as stream:
        for receptor_id, sequence in receptors:
            stream.write(f">r{receptor_id}\n{sequence}\n")


def nearest_cross_split_similarity(
    train_fasta: Path, test_fasta: Path, output_dir: Path, threads: int = 6
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="propedia_nearest_", dir=output_dir) as temporary:
        temp = Path(temporary)
        database, result = temp / "train.dmnd", temp / "cross.tsv"
        subprocess.run(
            ["diamond", "makedb", "--in", str(train_fasta), "--db", str(database)],
            check=True, capture_output=True, text=True,
        )
        subprocess.run([
            "diamond", "blastp", "--query", str(test_fasta), "--db", str(database),
            "--out", str(result), "--outfmt", "6", "qseqid", "sseqid", "pident",
            "qcovhsp", "scovhsp", "--very-sensitive", "--evalue", "1000",
            "--max-target-seqs", "0", "--threads", str(threads), "--tmpdir", str(temp),
        ], check=True, capture_output=True, text=True)
        best: tuple[float, float, float] | None = None
        hit_count = 0
        for line in result.read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            identity, qcov, scov = map(float, fields[2:5])
            if qcov >= CLUSTER_COVERAGE and scov >= CLUSTER_COVERAGE:
                hit_count += 1
                if best is None or identity > best[0]:
                    best = (identity, qcov, scov)
    return {
        "search": "test receptors against train receptors",
        "sensitivity": "very-sensitive",
        "evalue": 1000,
        "max_target_sequences": 0,
        "coverage_filter_percent_both_sequences": CLUSTER_COVERAGE,
        "qualifying_alignments": hit_count,
        "nearest_train_test_identity_percent": best[0] if best else None,
        "nearest_query_coverage_percent": best[1] if best else None,
        "nearest_subject_coverage_percent": best[2] if best else None,
        "identity_cluster_cutoff_percent": CLUSTER_IDENTITY,
    }


def freeze_split(
    pairs_path: Path,
    clusters_path: Path,
    output_dir: Path,
    *,
    seed: int = 20260926,
    threads: int = 6,
) -> dict:
    pairs, clusters = _read_jsonl(pairs_path), _read_jsonl(clusters_path)
    assignments = assign_clusters(pairs, clusters, seed)
    cluster_for_receptor = {
        cluster["receptor_id"]: cluster["receptor_cluster_id"] for cluster in clusters
    }
    pair_rows = []
    for pair in pairs:
        row = dict(pair)
        row["receptor_cluster_id"] = cluster_for_receptor[pair["receptor_id"]]
        row["split"] = assignments[pair["receptor_id"]]
        pair_rows.append(row)

    heldout_peptides = {
        pair["peptide_sequence"] for pair in pair_rows
        if pair["split"] in {"validation", "test"}
    }
    filtered = [
        pair for pair in pair_rows
        if not (pair["split"] == "train" and pair["peptide_sequence"] in heldout_peptides)
    ]
    split_rows = {
        split: sorted((row for row in filtered if row["split"] == split), key=lambda row: row["pair_id"])
        for split in ("train", "validation", "test")
    }
    train_peptides = {row["peptide_sequence"] for row in split_rows["train"]}
    test_peptides = {row["peptide_sequence"] for row in split_rows["test"]}
    validation_peptides = {row["peptide_sequence"] for row in split_rows["validation"]}
    if train_peptides & test_peptides or train_peptides & validation_peptides:
        raise RuntimeError("held-out peptide sequence crossed into training")

    output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split, rows in split_rows.items():
        path = output_dir / f"{split}.jsonl"
        hashes[split] = _write_jsonl(path, rows)
        _fasta(output_dir / f"{split}_receptors.fasta", rows, split)
    cluster_ids = {row["receptor_cluster_id"] for row in pair_rows}
    used_cluster_ids = {
        split: {row["receptor_cluster_id"] for row in rows}
        for split, rows in split_rows.items()
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if used_cluster_ids[left] & used_cluster_ids[right]:
            raise RuntimeError(f"receptor cluster leakage across {left}/{right}")

    nearest = nearest_cross_split_similarity(
        output_dir / "train_receptors.fasta", output_dir / "test_receptors.fasta",
        output_dir / "nearest_similarity", threads,
    )
    nearest_identity = nearest["nearest_train_test_identity_percent"]
    if nearest_identity is not None and nearest_identity >= CLUSTER_IDENTITY:
        raise RuntimeError(
            "split fails its independent nearest-similarity audit: a reciprocal-coverage "
            f"train/test hit has {nearest_identity:.1f}% identity"
        )
    summary = {
        "seed": seed,
        "pair_source_sha256": sha256_file(pairs_path),
        "cluster_source_sha256": sha256_file(clusters_path),
        "usable_unique_pair_count_before_split": len(pairs),
        "receptor_cluster_count": len(cluster_ids),
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "cluster_counts": {split: len(values) for split, values in used_cluster_ids.items()},
        "train_heldout_peptide_collisions_removed": len(pair_rows) - len(filtered),
        "manifest_sha256": hashes,
        "manifests_frozen": True,
        "cluster_overlap_checks_passed": True,
        "train_validation_peptide_overlap": len(train_peptides & validation_peptides),
        "train_test_peptide_overlap": len(train_peptides & test_peptides),
        "nearest_train_test_receptor_similarity": nearest,
    }
    (output_dir / "frozen_split_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze receptor-cluster-safe Propedia manifests")
    parser.add_argument("--pairs", type=Path, default=Path("data/protein/propedia26/unique_pairs.jsonl"))
    parser.add_argument("--clusters", type=Path, default=Path("data/protein/propedia26/clustering/receptor_clusters.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/protein/propedia26/manifests"))
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(freeze_split(args.pairs, args.clusters, args.output_dir, seed=args.seed, threads=args.threads), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
