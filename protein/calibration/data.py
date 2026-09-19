from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
REQUIRED_COLUMNS = {
    "domain_id",
    "sequence_id",
    "split",
    "backbone_path",
    "chain_id",
    "native_sequence",
    "ddg",
    "foldseek_qtm_max_to_train",
}
ALLOWED_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class SequenceRecord:
    domain_id: str
    sequence_id: str
    split: str
    backbone_path: Path
    chain_id: str
    native_sequence: str
    target_sequence: str
    ddg: float
    substitution_count: int
    foldseek_qtm_max_to_train: float

    @property
    def variant_sequence(self) -> str:
        return self.target_sequence

    @property
    def mutation_count(self) -> int:
        return self.substitution_count

    @property
    def sequence_length(self) -> int:
        return len(self.target_sequence)

    @property
    def d_struct(self) -> float:
        return 1.0 - self.foldseek_qtm_max_to_train


# Backward-compatible alias for existing imports and tests
VariantRecord = SequenceRecord


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError("Calibration manifest must be .jsonl or .parquet")


def load_manifest(path: Path, *, require_protein_dpo_provenance: bool = True) -> list[SequenceRecord]:
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing provenance sidecar: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if require_protein_dpo_provenance:
        required = {
            "dataset": "ProteinDPO curated Megascale v2 2023-04-20",
            "foldseek_metric": "query-normalized-tm-score",
        }
        bad = {key: (metadata.get(key), value) for key, value in required.items() if metadata.get(key) != value}
        provenance = metadata.get("split_provenance")
        reconstructed = {"reconstructed-proteindpo-methods", "reconstructed-tmalign-fallback"}
        if provenance not in {"authors-released", *reconstructed}:
            bad["split_provenance"] = (provenance, "authors-released or a disclosed reconstruction")
        if provenance in reconstructed and metadata.get("split_seed") != 42:
            bad["split_seed"] = (metadata.get("split_seed"), 42)
        split_algorithm = (
            "lexicographic-cluster-ids; python-random-v1-seed-42-shuffle; "
            "floor-90%-floor-5%-remainder"
        )
        if provenance in reconstructed and metadata.get("split_algorithm") != split_algorithm:
            bad["split_algorithm"] = (metadata.get("split_algorithm"), split_algorithm)
        if bad:
            raise RuntimeError(
                "ProteinDPO provenance contract failed: " f"metadata mismatches={bad}"
            )
    rows = _read_rows(path)
    if not rows:
        raise ValueError("Manifest is empty")
    row_keys = set(rows[0])
    missing = REQUIRED_COLUMNS - row_keys
    has_target = "target_sequence" in row_keys or "variant_sequence" in row_keys
    has_substitutions = "substitution_count" in row_keys or "mutation_count" in row_keys
    if missing or not has_target or not has_substitutions:
        unmet = set(missing)
        if not has_target:
            unmet.add("target_sequence|variant_sequence")
        if not has_substitutions:
            unmet.add("substitution_count|mutation_count")
        raise ValueError(f"Manifest is missing columns: {sorted(unmet)}")
    root = path.parent
    records = [
        SequenceRecord(
            domain_id=str(row["domain_id"]),
            sequence_id=str(row["sequence_id"]),
            split=str(row["split"]),
            backbone_path=(root / str(row["backbone_path"])).resolve(),
            chain_id=str(row["chain_id"]),
            native_sequence=str(row["native_sequence"]),
            target_sequence=str(row.get("target_sequence", row.get("variant_sequence"))),
            ddg=float(row["ddg"]),
            substitution_count=int(row.get("substitution_count", row.get("mutation_count"))),
            foldseek_qtm_max_to_train=float(row["foldseek_qtm_max_to_train"]),
        )
        for row in rows
    ]
    validate_records(records)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if metadata.get("manifest_sha256") != digest:
        raise RuntimeError("Manifest SHA-256 does not match its provenance sidecar")
    return records


def validate_records(records: Sequence[SequenceRecord]) -> None:
    domains_by_split: dict[str, set[str]] = {split: set() for split in ALLOWED_SPLITS}
    backbones_by_split: dict[str, set[tuple[Path, str]]] = {
        split: set() for split in ALLOWED_SPLITS
    }
    seen_ids: set[str] = set()
    for record in records:
        if record.sequence_id in seen_ids:
            raise ValueError(f"Duplicate sequence_id: {record.sequence_id}")
        seen_ids.add(record.sequence_id)
        if record.split not in ALLOWED_SPLITS:
            raise ValueError(f"Invalid split {record.split!r}")
        domains_by_split[record.split].add(record.domain_id)
        backbones_by_split[record.split].add((record.backbone_path, record.chain_id))
        if len(record.native_sequence) != len(record.target_sequence):
            raise ValueError(f"Indel is forbidden: {record.sequence_id}")
        if not set(record.native_sequence + record.target_sequence) <= CANONICAL_AA:
            raise ValueError(f"Noncanonical residue: {record.sequence_id}")
        observed = sum(a != b for a, b in zip(record.native_sequence, record.target_sequence))
        if observed != record.substitution_count or observed not in (1, 2):
            raise ValueError(f"Only valid single/double substitutions are allowed: {record.sequence_id}")
        if not 0.0 <= record.foldseek_qtm_max_to_train <= 1.0:
            raise ValueError(f"Invalid query-normalized TM-score: {record.sequence_id}")
        if not record.backbone_path.is_file():
            raise FileNotFoundError(record.backbone_path)
    for left in ALLOWED_SPLITS:
        for right in ALLOWED_SPLITS:
            if left < right and domains_by_split[left] & domains_by_split[right]:
                raise ValueError(f"Domain leakage between {left} and {right}")
            if left < right and backbones_by_split[left] & backbones_by_split[right]:
                raise ValueError(f"Backbone leakage between {left} and {right}")


def sft_training_records(records: Sequence[SequenceRecord]) -> list[SequenceRecord]:
    selected = [record for record in records if record.split == "train" and record.ddg < 0]
    if not selected:
        raise ValueError("No stabilizing training records (ddG < 0)")
    return selected


class EpochRandomSampler(Sampler[int]):
    """Identical epoch-specific ordering across adaptation conditions."""

    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.size, generator=generator).tolist())

    def __len__(self) -> int:
        return self.size


class BackboneBatchSampler(Sampler[list[int]]):
    """Paired, epoch-deterministic batches with maximal backbone reuse.

    Every example occurs exactly once per epoch. Full batches contain one
    backbone whenever possible; per-backbone remainders are packed together.
    This changes neither the examples nor the effective batch size and is used
    identically for every adaptation condition.
    """

    def __init__(self, records: Sequence[SequenceRecord], batch_size: int, seed: int, *, shuffle: bool) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.records = list(records)
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        grouped: dict[tuple[Path, str], list[int]] = {}
        for index, record in enumerate(self.records):
            grouped.setdefault((record.backbone_path, record.chain_id), []).append(index)
        self._groups = list(grouped.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        full: list[list[int]] = []
        tails: list[int] = []
        for original in self._groups:
            indices = list(original)
            if self.shuffle and len(indices) > 1:
                order = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[i] for i in order]
            cutoff = len(indices) - len(indices) % self.batch_size
            full.extend(indices[i : i + self.batch_size] for i in range(0, cutoff, self.batch_size))
            tails.extend(indices[cutoff:])
        if self.shuffle and len(tails) > 1:
            order = torch.randperm(len(tails), generator=generator).tolist()
            tails = [tails[i] for i in order]
        full.extend(tails[i : i + self.batch_size] for i in range(0, len(tails), self.batch_size))
        if self.shuffle and len(full) > 1:
            order = torch.randperm(len(full), generator=generator).tolist()
            full = [full[i] for i in order]
        return full

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._batches())

    def __len__(self) -> int:
        return (len(self.records) + self.batch_size - 1) // self.batch_size


class CoordinateDataset(Dataset[SequenceRecord]):
    def __init__(self, records: Sequence[SequenceRecord]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> SequenceRecord:
        return self.records[index]


def load_record_coords(record: SequenceRecord) -> np.ndarray:
    if record.backbone_path.suffix == ".npy":
        coords = np.load(record.backbone_path, allow_pickle=False)
    else:
        from esm.inverse_folding.util import extract_coords_from_structure, load_structure

        structure = load_structure(str(record.backbone_path), record.chain_id)
        coords, _ = extract_coords_from_structure(structure)
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 3 or coords.shape[1:] != (3, 3):
        raise ValueError(f"Backbone must have shape [L,3,3], got {coords.shape} for {record.backbone_path}")
    return coords


class IF1Collator:
    def __init__(
        self,
        alphabet: Any,
        *,
        training: bool,
        coordinate_noise: float = 0.0,
        pad_to_sequence_length: int | None = 72,
        coordinate_cache: dict[tuple[Path, str], np.ndarray] | None = None,
    ) -> None:
        from esm.inverse_folding.util import CoordBatchConverter

        self.converter = CoordBatchConverter(alphabet)
        self.training = training
        self.coordinate_noise = coordinate_noise
        self.pad_to_sequence_length = pad_to_sequence_length
        self._cache = coordinate_cache if coordinate_cache is not None else {}

    def _coords(self, record: SequenceRecord) -> np.ndarray:
        key = (record.backbone_path, record.chain_id)
        cached = self._cache.get(key)
        if cached is None:
            cached = load_record_coords(record)
            if cached.ndim != 3 or cached.shape[1:] != (3, 3):
                raise ValueError(
                    f"Backbone must have shape [L,3,3], got {cached.shape} for {record.backbone_path}"
                )
            self._cache[key] = cached
        return cached.copy()

    def __call__(self, records: Sequence[SequenceRecord]) -> dict[str, Any]:
        raw = []
        noisy_by_backbone: dict[tuple[Path, str], np.ndarray] = {}
        for record in records:
            key = (record.backbone_path, record.chain_id)
            coords = noisy_by_backbone.get(key)
            if coords is None:
                coords = self._coords(record)
                if self.training and self.coordinate_noise:
                    finite = np.isfinite(coords)
                    noise = np.random.normal(0.0, self.coordinate_noise, size=coords.shape).astype(np.float32)
                    coords[finite] += noise[finite]
                noisy_by_backbone[key] = coords
            if len(coords) != record.sequence_length:
                raise ValueError(f"Backbone/sequence length mismatch: {record.sequence_id}")
            raw.append((coords, None, record.target_sequence))
        add_sentinel = self.pad_to_sequence_length is not None
        if add_sentinel:
            if any(len(record.target_sequence) > self.pad_to_sequence_length for record in records):
                raise ValueError(
                    f"Sequence exceeds fixed compiled length {self.pad_to_sequence_length}; no truncation is allowed"
                )
            sentinel = np.full((self.pad_to_sequence_length, 3, 3), np.inf, dtype=np.float32)
            raw.append((sentinel, None, "A" * self.pad_to_sequence_length))
        coords, confidence, _, tokens, padding_mask = self.converter(raw)
        if add_sentinel:
            coords, confidence, tokens, padding_mask = (
                coords[:-1], confidence[:-1], tokens[:-1], padding_mask[:-1]
            )
        first: dict[tuple[Path, str], int] = {}
        unique: list[int] = []
        inverse: list[int] = []
        for row, record in enumerate(records):
            key = (record.backbone_path, record.chain_id)
            if key not in first:
                first[key] = len(unique)
                unique.append(row)
            inverse.append(first[key])
        return {
            "coords": coords,
            "confidence": confidence,
            "tokens": tokens,
            "padding_mask": padding_mask,
            "records": list(records),
            "backbone_unique": torch.tensor(unique, dtype=torch.long),
            "backbone_inverse": torch.tensor(inverse, dtype=torch.long),
        }


def preload_coordinate_cache(records: Sequence[SequenceRecord]) -> dict[tuple[Path, str], np.ndarray]:
    """Parse every referenced backbone once before persistent workers spawn."""
    cache: dict[tuple[Path, str], np.ndarray] = {}
    for record in records:
        key = (record.backbone_path, record.chain_id)
        if key not in cache:
            cache[key] = load_record_coords(record)
    return cache
