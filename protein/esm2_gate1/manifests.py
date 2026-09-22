from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from .config import CANONICAL_AA, DATASET_ID, DATASET_REVISION


@dataclass(frozen=True)
class BackboneSequence:
    backbone_id: str
    sequence_id: str
    sequence: str
    length: int
    brn_member: bool
    brq_member: bool
    sequence_sha256: str
    description: str
    structural_similarity_10k: float | None = None
    structural_similarity_50k: float | None = None
    sequence_identity_10k: float | None = None
    sequence_coverage_10k: float | None = None
    sequence_identity_50k: float | None = None
    sequence_coverage_50k: float | None = None


def stable_hash(*values: object, seed: int = 0) -> str:
    joined = "\0".join(str(value) for value in (seed, *values))
    return hashlib.sha256(joined.encode()).hexdigest()


def normalize_sequence(value: object) -> str | None:
    sequence = str(value).upper()
    if not sequence or not set(sequence) <= set(CANONICAL_AA):
        return None
    return sequence


def sequence_metrics(sequence: str) -> dict[str, float]:
    counts = Counter(sequence)
    probabilities = np.asarray([count / len(sequence) for count in counts.values()])
    longest = 1
    run = 1
    for previous, current in zip(sequence, sequence[1:]):
        run = run + 1 if previous == current else 1
        longest = max(longest, run)
    result = {
        "entropy_bits": float(-(probabilities * np.log2(probabilities)).sum()),
        "maximum_residue_fraction": max(counts.values()) / len(sequence),
        "maximum_homopolymer_fraction": longest / len(sequence),
        "unique_residue_fraction": len(counts) / 20,
    }
    result.update({f"aa_{aa}": counts.get(aa, 0) / len(sequence) for aa in CANONICAL_AA})
    return result


def population_summary(records: Sequence[BackboneSequence]) -> dict:
    result = sequence_collection_summary([record.sequence for record in records])
    result.update({
        "backbones": len({record.backbone_id for record in records}),
        "sequences": len(records),
    })
    return result


def sequence_collection_summary(sequences: Sequence[str]) -> dict:
    if not sequences:
        raise ValueError("cannot summarize an empty sequence population")
    lengths = np.asarray([len(sequence) for sequence in sequences], dtype=np.int64)
    metrics = [sequence_metrics(sequence) for sequence in sequences]
    residues = Counter("".join(sequences))
    return {
        "sequences": len(sequences),
        "residues": int(lengths.sum()),
        "length": _distribution(lengths),
        "entropy_bits": _distribution(np.asarray([value["entropy_bits"] for value in metrics])),
        "maximum_residue_fraction": _distribution(
            np.asarray([value["maximum_residue_fraction"] for value in metrics])
        ),
        "maximum_homopolymer_fraction": _distribution(
            np.asarray([value["maximum_homopolymer_fraction"] for value in metrics])
        ),
        "unique_residue_fraction": _distribution(
            np.asarray([value["unique_residue_fraction"] for value in metrics])
        ),
        "amino_acid_frequencies": {
            aa: residues[aa] / int(lengths.sum()) for aa in CANONICAL_AA
        },
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(values.min()),
        "q01": float(np.quantile(values, 0.01)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.5)),
        "q75": float(np.quantile(values, 0.75)),
        "q99": float(np.quantile(values, 0.99)),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
    }


def choose_representatives(rows: Iterable[dict], allowed: set[str] | None = None) -> tuple[dict[str, BackboneSequence], dict]:
    representatives: dict[str, tuple[str, BackboneSequence]] = {}
    row_counts: Counter[str] = Counter()
    malformed = 0
    invalid_length = 0
    total = 0
    for row in rows:
        total += 1
        backbone_id = str(row["accession"])
        if allowed is not None and backbone_id not in allowed:
            continue
        row_counts[backbone_id] += 1
        sequence = normalize_sequence(row["sequence"])
        if sequence is None:
            malformed += 1
            continue
        if not 40 <= len(sequence) <= 512:
            invalid_length += 1
            continue
        description = str(row.get("description", ""))
        sequence_id = f"{backbone_id}|{description.lstrip('| ').replace(' ', '_')}"
        key = stable_hash(backbone_id, description, sequence, seed=17)
        record = BackboneSequence(
            backbone_id=backbone_id,
            sequence_id=sequence_id,
            sequence=sequence,
            length=len(sequence),
            brn_member=True,
            brq_member=allowed is not None,
            sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
            description=description,
        )
        if backbone_id not in representatives or key < representatives[backbone_id][0]:
            representatives[backbone_id] = (key, record)
    counts = np.asarray(list(row_counts.values()), dtype=np.int64)
    return (
        {key: value[1] for key, value in representatives.items()},
        {
            "rows_scanned": total,
            "unique_accessions": len(row_counts),
            "malformed_rows": malformed,
            "invalid_length_rows": invalid_length,
            "rows_per_accession": _distribution(counts) if counts.size else {},
        },
    )


def remove_cross_backbone_duplicates(records: Sequence[BackboneSequence]) -> tuple[list[BackboneSequence], int]:
    by_sequence: dict[str, list[BackboneSequence]] = defaultdict(list)
    for record in records:
        by_sequence[record.sequence_sha256].append(record)
    retained: list[BackboneSequence] = []
    removed = 0
    for group in by_sequence.values():
        # Preserve the preferred BRn∩BRq population when a duplicate crosses filters.
        group.sort(key=lambda record: (not record.brq_member, stable_hash(record.backbone_id, seed=23)))
        retained.append(group[0])
        removed += len(group) - 1
    retained.sort(key=lambda record: record.backbone_id)
    return retained, removed


def derive_low_complexity_thresholds(natural_sequences: Sequence[str]) -> dict:
    """Preregistered extreme empirical tails; called before any model result exists."""
    values = [sequence_metrics(sequence) for sequence in natural_sequences]
    return {
        "rule": "exclude if entropy<q0.001 OR max-residue>q0.999 OR homopolymer>q0.999",
        "entropy_bits_min": float(np.quantile([x["entropy_bits"] for x in values], 0.001)),
        "maximum_residue_fraction_max": float(
            np.quantile([x["maximum_residue_fraction"] for x in values], 0.999)
        ),
        "maximum_homopolymer_fraction_max": float(
            np.quantile([x["maximum_homopolymer_fraction"] for x in values], 0.999)
        ),
    }


def passes_low_complexity(sequence: str, thresholds: dict) -> bool:
    metrics = sequence_metrics(sequence)
    return (
        metrics["entropy_bits"] >= thresholds["entropy_bits_min"]
        and metrics["maximum_residue_fraction"] <= thresholds["maximum_residue_fraction_max"]
        and metrics["maximum_homopolymer_fraction"] <= thresholds["maximum_homopolymer_fraction_max"]
    )


def stratified_nested_sample(
    records: Sequence[BackboneSequence], small: int = 10_000, large: int = 50_000, seed: int = 11
) -> tuple[list[BackboneSequence], list[BackboneSequence], list[BackboneSequence]]:
    if len(records) < large:
        raise ValueError(f"need at least {large} eligible backbones, found {len(records)}")
    bins: dict[int, list[BackboneSequence]] = defaultdict(list)
    for record in records:
        bins[(record.length - 1) // 32].append(record)
    for group in bins.values():
        group.sort(key=lambda record: stable_hash(record.backbone_id, seed=seed))
    large_counts = proportional_counts({key: len(value) for key, value in bins.items()}, large)
    selected_large = [record for key in sorted(bins) for record in bins[key][: large_counts[key]]]
    selected_by_bin: dict[int, list[BackboneSequence]] = defaultdict(list)
    for record in selected_large:
        selected_by_bin[(record.length - 1) // 32].append(record)
    small_counts = proportional_counts({key: len(value) for key, value in selected_by_bin.items()}, small)
    selected_small = [
        record for key in sorted(selected_by_bin) for record in selected_by_bin[key][: small_counts[key]]
    ]
    large_ids = {record.backbone_id for record in selected_large}
    heldout = [record for record in records if record.backbone_id not in large_ids]
    heldout.sort(key=lambda record: stable_hash(record.backbone_id, seed=seed + 1))
    return selected_small, selected_large, heldout


def proportional_counts(sizes: dict[int, int], total: int) -> dict[int, int]:
    population = sum(sizes.values())
    raw = {key: total * size / population for key, size in sizes.items()}
    counts = {key: min(sizes[key], math.floor(value)) for key, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(sizes, key=lambda key: (-(raw[key] - counts[key]), key))
    while remaining:
        changed = False
        for key in order:
            if counts[key] < sizes[key] and remaining:
                counts[key] += 1
                remaining -= 1
                changed = True
        if not changed:
            raise RuntimeError("could not allocate stratified sample")
    return counts


def write_manifest(path: Path, records: Sequence[BackboneSequence], *, split: str, extra: dict | None = None) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({**asdict(record), "split": split}, sort_keys=True) for record in records]
    payload = ("\n".join(lines) + "\n").encode()
    path.write_bytes(payload)
    metadata = {
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "split": split,
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "statistics": population_summary(records),
        **(extra or {}),
    }
    path.with_suffix(path.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata


def load_manifest(path: Path) -> list[dict]:
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["dataset_revision"] != DATASET_REVISION:
        raise RuntimeError("manifest uses a different Dayhoff revision")
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["manifest_sha256"]:
        raise RuntimeError("manifest hash mismatch")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    ids = [row["backbone_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("backbone identity is not unique within manifest")
    return rows
