"""Deterministic manifests and homology-split validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from .tokenization import is_canonical


def stable_hash(value: str, salt: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{salt}\0{value}".encode()).digest()[:8], "big")


def manifest_sha256(rows: Iterable[dict]) -> str:
    payload = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def filter_sequence(row: dict, min_length: int = 64, max_length: int = 512) -> bool:
    sequence = str(row["sequence"]).upper()
    return min_length <= len(sequence) <= max_length and is_canonical(sequence)


def assign_clusters(
    rows: list[dict], cluster_by_id: dict[str, str], *, salt: str = "rita-protein-v1"
) -> list[dict]:
    """Assign complete MMseqs clusters to approximately 70/15/15 splits."""
    split_names = ("train", "validation", "test")
    output = []
    cluster_splits: dict[str, str] = {}
    for row in rows:
        identifier = str(row["id"])
        cluster = cluster_by_id[identifier]
        bucket = stable_hash(cluster, salt) % 100
        split = split_names[0] if bucket < 70 else split_names[1] if bucket < 85 else split_names[2]
        cluster_splits.setdefault(cluster, split)
        if cluster_splits[cluster] != split:
            raise AssertionError("one homology cluster crossed split boundaries")
        output.append({**row, "cluster_id": cluster, "split": split})
    return output


def validate_split_integrity(rows: Iterable[dict]) -> None:
    seen_ids: set[str] = set()
    cluster_split: dict[str, str] = {}
    for row in rows:
        identifier, cluster, split = str(row["id"]), str(row["cluster_id"]), str(row["split"])
        if identifier in seen_ids:
            raise AssertionError(f"duplicate protein id: {identifier}")
        seen_ids.add(identifier)
        previous = cluster_split.setdefault(cluster, split)
        if previous != split:
            raise AssertionError(f"homology cluster {cluster} crosses {previous}/{split}")


def write_jsonl(path: Path, rows: Iterable[dict]) -> str:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8", newline="\n")
    return hashlib.sha256(text.encode()).hexdigest()

