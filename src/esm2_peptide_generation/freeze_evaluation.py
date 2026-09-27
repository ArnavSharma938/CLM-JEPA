from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .prepare_data import sha256_file


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write(path: Path, rows: list[dict]) -> str:
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            stream.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def make_hpo_subset(train_manifest: Path, output_path: Path, count: int = 4096, seed: int = 7) -> dict:
    rows = _read(train_manifest)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["receptor_cluster_id"]].append(row)
    for cluster_id, members in groups.items():
        members.sort(key=lambda item: hashlib.sha256(f"{seed}\0{item['pair_id']}".encode()).hexdigest())
    cluster_ids = sorted(
        groups,
        key=lambda value: hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest(),
    )
    target = min(count, len(rows))
    selected: list[dict] = []
    cursor = 0
    while len(selected) < target:
        added = False
        for cluster_id in cluster_ids:
            if cursor < len(groups[cluster_id]):
                selected.append(groups[cluster_id][cursor])
                added = True
                if len(selected) == target:
                    break
        if not added:
            break
        cursor += 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_hash = _write(output_path, selected)
    return {
        "seed": seed,
        "requested_pairs": count,
        "selected_pairs": len(selected),
        "source_train_sha256": sha256_file(train_manifest),
        "hpo_manifest_sha256": manifest_hash,
        "receptor_clusters_represented": len({row["receptor_cluster_id"] for row in selected}),
        "manifest": output_path.name,
    }


def make_fixed_decoys(
    test_manifest: Path, output_path: Path, *, count: int = 4, seed: int = 20260926
) -> dict:
    rows = _read(test_manifest)
    receptors: dict[str, dict] = {}
    for row in rows:
        receptors.setdefault(row["receptor_id"], {
            "receptor_id": row["receptor_id"],
            "receptor_sequence": row["receptor_sequence"],
            "receptor_cluster_id": row["receptor_cluster_id"],
        })
    choices: dict[str, list[str]] = {}
    for receptor_id, receptor in receptors.items():
        target_len = len(receptor["receptor_sequence"])
        alternatives = [
            item["receptor_id"] for item in receptors.values()
            if item["receptor_cluster_id"] != receptor["receptor_cluster_id"]
        ]
        if len(alternatives) < count:
            raise RuntimeError("test set has too few receptor clusters for fixed decoys")
        in_band = [
            candidate for candidate in alternatives
            if abs(len(receptors[candidate]["receptor_sequence"]) - target_len) / target_len <= 0.2
        ]
        pool = in_band if len(in_band) >= count else sorted(
            alternatives,
            key=lambda candidate: (
                abs(len(receptors[candidate]["receptor_sequence"]) - target_len), candidate
            ),
        )
        pool = sorted(pool, key=lambda candidate: hashlib.sha256(
            f"{seed}\0{receptor_id}\0{candidate}".encode()
        ).hexdigest())
        choices[receptor_id] = pool[:count]
    decoys = []
    for row in rows:
        for decoy_id in choices[row["receptor_id"]]:
            decoy = receptors[decoy_id]
            decoys.append({
                "pair_id": row["pair_id"],
                "true_receptor_id": row["receptor_id"],
                "true_receptor_cluster_id": row["receptor_cluster_id"],
                "decoy_receptor_id": decoy_id,
                "decoy_receptor_cluster_id": decoy["receptor_cluster_id"],
                "decoy_receptor_sequence": decoy["receptor_sequence"],
                "relative_length_difference": abs(
                    len(decoy["receptor_sequence"]) - len(row["receptor_sequence"])
                ) / len(row["receptor_sequence"]),
            })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    digest = _write(output_path, decoys)
    return {
        "seed": seed,
        "decoys_per_pair": count,
        "source_test_sha256": sha256_file(test_manifest),
        "fixed_decoy_manifest_sha256": digest,
        "pair_count": len(rows),
        "decoy_evaluations": len(decoys),
        "mean_relative_length_difference": sum(x["relative_length_difference"] for x in decoys) / len(decoys),
        "manifest": output_path.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze cluster-balanced HPO and fixed-decoy manifests")
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/protein/propedia26/manifests"))
    parser.add_argument("--hpo-count", type=int, default=4096)
    parser.add_argument("--decoys", type=int, default=4)
    args = parser.parse_args()
    hpo = make_hpo_subset(args.manifest_dir / "train.jsonl", args.manifest_dir / "hpo_4096.jsonl", args.hpo_count)
    decoys = make_fixed_decoys(args.manifest_dir / "test.jsonl", args.manifest_dir / "test_decoys.jsonl", count=args.decoys)
    summary = {"hpo_subset": hpo, "fixed_decoys": decoys}
    (args.manifest_dir / "frozen_evaluation_manifests.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
