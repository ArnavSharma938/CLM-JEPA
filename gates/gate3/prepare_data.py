from __future__ import annotations

import csv
import argparse
import hashlib
import heapq
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUTPUT = DATA / "gate3"
SAMPLE_SIZE = 1024
SEED = 533


def compact(text: object) -> str:
    return "" if text is None else str(text).replace(" ", "").strip()


def released_identity(smiles: str, *, unordered: bool = False) -> str:
    components = smiles.split(".")
    if unordered:
        components.sort()
    return ".".join(components)


def csv_pairs(path: Path, *, reverse: bool = False) -> Iterator[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            src = compact(row.get("source", row.get("src")))
            tgt = compact(row.get("target", row.get("tgt")))
            yield (tgt, src) if reverse else (src, tgt)


def datasets() -> dict[str, tuple[Iterable[tuple[str, str]], bool, str]]:
    return {
        "uspto_mit_synthesis": (
            csv_pairs(DATA / "uspto_mit_synthesis" / "train_r_smiles.csv"), False,
            "ChemFM/R-SMILES official synthesis training split, normalized to paired CSV",
        ),
        "orderly_forward": (
            csv_pairs(DATA / "orderly_forward" / "train.csv"), False,
            "ORDerly Figshare forward train, exactly one product, normalized CSV",
        ),
        "non_uspto_forward": (
            csv_pairs(DATA / "non_uspto_forward" / "test.csv"), False,
            "ORDerly Figshare non-USPTO forward external test, exactly one product",
        ),
        "metatrans_full": (
            csv_pairs(DATA / "metatrans" / "train.csv"), False,
            "MetaTrans author-released full training collection, normalized CSV",
        ),
        "uspto_50k_retro": (
            csv_pairs(DATA / "uspto_50k" / "train_single.csv"), False,
            "ChemFM author-named USPTO-50K train_single release in retrosynthesis direction",
        ),
        "uspto_480k_template_heldout": (
            csv_pairs(DATA / "uspto_480k_template_heldout" / "train.csv"), True,
            "Retrosynthesis Extrapolation official template-held-out training split, normalized CSV",
        ),
        "non_uspto_retro": (
            csv_pairs(DATA / "non_uspto_retro" / "test.csv"), True,
            "ORDerly Figshare non-USPTO retrosynthesis external test, normalized CSV",
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
    return {
        "orderly_forward_train_test_pairs": overlap(
            csv_pairs(DATA / "orderly_forward" / "train.csv"),
            csv_pairs(DATA / "orderly_forward" / "test.csv"),
        ),
        "metatrans_train_valid_parents": overlap(
            csv_pairs(DATA / "metatrans" / "train.csv"),
            csv_pairs(DATA / "metatrans" / "released_validation.csv"),
            parent_only=True,
        ),
        "uspto_50k_train_test_pairs": overlap(
            csv_pairs(DATA / "uspto_50k" / "train_r_smiles.csv"),
            csv_pairs(DATA / "uspto_50k" / "test_r_smiles.csv"),
        ),
        "uspto_480k_train_test_pairs": overlap(
            csv_pairs(DATA / "uspto_480k_template_heldout" / "train.csv"),
            csv_pairs(DATA / "uspto_480k_template_heldout" / "test.csv"),
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", choices=sorted(datasets()))
    parser.add_argument(
        "--skip-leakage-checks", action="store_true",
        help="Permit targeted regeneration when unrelated official split files are absent.",
    )
    parser.add_argument("--audit", type=Path, help="Optional path for a preparation audit JSON.")
    args = parser.parse_args()
    selected = args.dataset or list(datasets())
    definitions = datasets()
    audits = [sample_dataset(name, *definitions[name]) for name in selected]
    result = {
        "sample_policy": "1024 smallest SHA-256 priorities over first valid row per released target identity",
        "sample_size": SAMPLE_SIZE,
        "seed": SEED,
        "datasets": audits,
        "released_split_leakage": (
            {} if args.skip_leakage_checks else leakage_checks()
        ),
        "validity_boundary": "Gate 3 uses the authors' released canonical strings. Full RDKit sanitization is deferred to task preprocessing because RDKit 2024.9.1 segfaults on malformed released strings; exact source-target component overlap is excluded here.",
        "metatrans_curated_subset": "omitted: released source/target files contain no per-pair provenance",
        "non_uspto_source": "ORDerly public Figshare release; Bradshaw Pistachio splits are proprietary and were not substituted",
    }
    if args.audit:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
