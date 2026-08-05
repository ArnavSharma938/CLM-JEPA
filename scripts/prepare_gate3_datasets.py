from __future__ import annotations

import csv
import hashlib
import heapq
import json
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "data" / "official"
OUTPUT = ROOT / "data" / "manifests" / "gate3_multi"
SAMPLE_SIZE = 1024
SEED = 533


def compact(text: object) -> str:
    return "" if text is None or pd.isna(text) else str(text).replace(" ", "").strip()


def released_identity(smiles: str, *, unordered: bool = False) -> str:
    components = smiles.split(".")
    if unordered:
        components.sort()
    return ".".join(components)


def csv_pairs(path: Path, *, reverse: bool = False) -> Iterator[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            src, tgt = compact(row["src"]), compact(row["tgt"])
            yield (tgt, src) if reverse else (src, tgt)


def parallel_pairs(source: Path, target: Path) -> Iterator[tuple[str, str]]:
    with source.open(encoding="utf-8") as sources, target.open(encoding="utf-8") as targets:
        for src, tgt in zip(sources, targets, strict=True):
            yield compact(src), compact(tgt)


def orderly_pairs(path: Path, *, reverse: bool = False) -> Iterator[tuple[str, str]]:
    frame = pd.read_parquet(path)
    source_columns = [
        column for column in frame.columns
        if column.startswith(("reactant_", "solvent_", "agent_"))
    ]
    product_columns = [column for column in frame.columns if column.startswith("product_")]
    for row in frame.itertuples(index=False):
        values = row._asdict()
        products = [compact(values[column]) for column in product_columns if compact(values[column])]
        if len(products) != 1:
            continue
        inputs = [compact(values[column]) for column in source_columns if compact(values[column])]
        if not inputs:
            continue
        source, product = ".".join(inputs), products[0]
        if reverse:
            reactants = [
                compact(values[column]) for column in source_columns
                if column.startswith("reactant_") and compact(values[column])
            ]
            if reactants:
                yield product, ".".join(reactants)
        else:
            yield source, product


def retro_extrapolation_pairs(path: Path) -> Iterator[tuple[str, str]]:
    with path.open(encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2:
                yield compact(row[1]), compact(row[0])


def datasets() -> dict[str, tuple[Iterable[tuple[str, str]], bool, str]]:
    stereo = OFFICIAL / "uspto_stereo" / "data" / "STEREO_mixed"
    meta = OFFICIAL / "metatrans" / "repo" / "datasets"
    orderly = OFFICIAL / "orderly"
    return {
        "uspto_mit_synthesis": (
            csv_pairs(OFFICIAL / "uspto_mit_synthesis" / "train.csv"), False,
            "ChemFM Box train.csv (official R-SMILES synthesis training split)",
        ),
        "uspto_stereo_mixed": (
            parallel_pairs(stereo / "src-train.txt", stereo / "tgt-train.txt"), False,
            "MolecularTransformer Box STEREO_mixed published training split",
        ),
        "orderly_forward": (
            orderly_pairs(orderly / "orderly_forward_train.parquet"), False,
            "ORDerly Figshare forward train, exactly one product",
        ),
        "non_uspto_forward": (
            orderly_pairs(orderly / "non_uspto_data" / "non_uspto_orderly_forward.parquet"), False,
            "ORDerly Figshare non-USPTO forward release",
        ),
        "metatrans_full": (
            parallel_pairs(meta / "train" / "source_train.txt", meta / "train" / "target_train.txt"), False,
            "MetaTrans author-released full training collection",
        ),
        "uspto_50k_retro": (
            csv_pairs(OFFICIAL / "uspto_50k" / "train_single.csv", reverse=True), True,
            "ChemFM Box USPTO-50K single-enumeration training split, reversed for retrosynthesis",
        ),
        "uspto_480k_template_heldout": (
            retro_extrapolation_pairs(OFFICIAL / "retro_extrapolation" / "train_480k_mt.txt"), True,
            "Retrosynthesis-Extrapolation Figshare frequency/template-held-out training split",
        ),
        "non_uspto_retro": (
            orderly_pairs(orderly / "non_uspto_data" / "non_uspto_orderly_retro.parquet", reverse=True), True,
            "ORDerly Figshare non-USPTO retrosynthesis release",
        ),
    }


def sample_dataset(name: str, pairs: Iterable[tuple[str, str]], unordered: bool, provenance: str) -> dict:
    heap: list[tuple[int, str, str, str]] = []
    seen_identities: set[str] = set()
    audit = {
        "dataset": name, "provenance": provenance, "rows_seen": 0, "valid_rows": 0,
        "empty_target_rows": 0, "source_target_component_overlap": 0, "duplicate_target_identities": 0,
    }
    def consume(buffer: list[tuple[str, str]]) -> None:
        for src, tgt in buffer:
            identity = released_identity(tgt, unordered=unordered)
            if not tgt:
                audit["empty_target_rows"] += 1
                continue
            if set(src.split(".")) & set(tgt.split(".")):
                audit["source_target_component_overlap"] += 1
                continue
            if identity in seen_identities:
                audit["duplicate_target_identities"] += 1
                continue
            seen_identities.add(identity)
            audit["valid_rows"] += 1
            priority = int.from_bytes(
                hashlib.sha256(f"{SEED}|{name}|{src}|{tgt}".encode()).digest()[:8], "big"
            )
            item = (-priority, identity, src, tgt)
            if len(heap) < SAMPLE_SIZE:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)

    buffer: list[tuple[str, str]] = []
    for src, tgt in pairs:
        audit["rows_seen"] += 1
        buffer.append((src, tgt))
        if len(buffer) == 65536:
            consume(buffer)
            buffer = []
    consume(buffer)
    selected = sorted(heap, key=lambda item: (-item[0], item[1]))
    if len(selected) < SAMPLE_SIZE:
        raise RuntimeError(f"{name} has only {len(selected)} usable unique targets")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["src", "tgt"])
        writer.writeheader()
        writer.writerows({"src": src, "tgt": tgt} for _, _, src, tgt in selected)
    audit.update({"sample_size": len(selected), "manifest": str(path.relative_to(ROOT)), "seed": SEED})
    print(json.dumps(audit), flush=True)
    return audit


def overlap(left: Iterable[tuple[str, str]], right: Iterable[tuple[str, str]], *, parent_only: bool = False) -> dict:
    left_values = {compact(src) if parent_only else f"{compact(src)}>>{compact(tgt)}" for src, tgt in left}
    right_values = {compact(src) if parent_only else f"{compact(src)}>>{compact(tgt)}" for src, tgt in right}
    return {"left_unique": len(left_values), "right_unique": len(right_values), "overlap": len(left_values & right_values)}


def leakage_checks() -> dict:
    stereo = OFFICIAL / "uspto_stereo" / "data" / "STEREO_mixed"
    meta = OFFICIAL / "metatrans" / "repo" / "datasets"
    orderly = OFFICIAL / "orderly"
    return {
        "uspto_stereo_train_test_pairs": overlap(
            parallel_pairs(stereo / "src-train.txt", stereo / "tgt-train.txt"),
            parallel_pairs(stereo / "src-test.txt", stereo / "tgt-test.txt"),
        ),
        "orderly_forward_train_test_pairs": overlap(
            orderly_pairs(orderly / "orderly_forward_train.parquet"),
            orderly_pairs(orderly / "orderly_forward_test.parquet"),
        ),
        "metatrans_train_valid_parents": overlap(
            parallel_pairs(meta / "train" / "source_train.txt", meta / "train" / "target_train.txt"),
            parallel_pairs(meta / "valid" / "source_valid.txt", meta / "valid" / "target_valid.txt"),
            parent_only=True,
        ),
        "uspto_50k_train_test_pairs": overlap(
            csv_pairs(OFFICIAL / "uspto_50k" / "train_single.csv", reverse=True),
            csv_pairs(OFFICIAL / "uspto_50k" / "test_single.csv", reverse=True),
        ),
        "uspto_480k_train_test_pairs": overlap(
            retro_extrapolation_pairs(OFFICIAL / "retro_extrapolation" / "train_480k_mt.txt"),
            retro_extrapolation_pairs(OFFICIAL / "retro_extrapolation" / "test_480k_mt.txt"),
        ),
    }


def main() -> None:
    audits = [sample_dataset(name, pairs, unordered, source) for name, (pairs, unordered, source) in datasets().items()]
    result = {
        "sample_policy": "1024 smallest SHA-256 priorities over first valid row per canonical target identity",
        "sample_size": SAMPLE_SIZE,
        "seed": SEED,
        "datasets": audits,
        "released_split_leakage": leakage_checks(),
        "validity_boundary": "Gate 3 uses the authors' released canonical strings. Full RDKit sanitization is deferred to task preprocessing because RDKit 2024.9.1 segfaults on malformed released strings; exact source-target component overlap is excluded here.",
        "metatrans_curated_subset": "omitted: released source/target files contain no per-pair provenance",
        "non_uspto_source": "ORDerly public Figshare release; Bradshaw Pistachio splits are proprietary and were not substituted",
    }
    path = ROOT / "artifacts" / "data" / "dataset_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
