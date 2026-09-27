from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from .config import DATASET_ID, DATASET_REVISION
from .manifests import (
    BackboneSequence,
    choose_representatives,
    derive_low_complexity_thresholds,
    normalize_sequence,
    passes_low_complexity,
    population_summary,
    proportional_counts,
    remove_cross_backbone_duplicates,
    sequence_collection_summary,
    stable_hash,
    stratified_nested_sample,
    write_manifest,
)


def _dataset(config: str, split: str):
    from datasets import load_dataset

    return load_dataset(
        DATASET_ID,
        name=config,
        split=split,
        revision=DATASET_REVISION,
        streaming=True,
    )


def _natural_pool() -> list[tuple[str, str]]:
    result = []
    for row in _dataset("uniref50", "test"):
        sequence = normalize_sequence(row["sequence"])
        if sequence is None or not 40 <= len(sequence) <= 512:
            continue
        result.append((str(row.get("accession", stable_hash(sequence)[:20])), sequence))
    return result


def _length_matched_natural(
    pool: list[tuple[str, str]], synthetic, limit: int
) -> list[tuple[str, str]]:
    by_bin: dict[int, list[tuple[str, str]]] = {}
    for accession, sequence in pool:
        by_bin.setdefault((len(sequence) - 1) // 32, []).append((accession, sequence))
    for values in by_bin.values():
        values.sort(key=lambda item: stable_hash(item[0], item[1], seed=5))
    synthetic_counts = Counter((record.length - 1) // 32 for record in synthetic)
    missing_bins = [key for key in synthetic_counts if not by_bin.get(key)]
    if missing_bins:
        raise RuntimeError(f"UniRef50 reference has no sequences in length bins: {missing_bins}")
    synthetic_total = sum(synthetic_counts.values())
    # Find the largest sample whose bin counts can preserve the synthetic proportions.
    capacity = min(
        len(by_bin[key]) * synthetic_total / synthetic_counts[key]
        for key in synthetic_counts
    )
    target = min(limit, int(capacity), sum(len(by_bin[key]) for key in synthetic_counts))
    while target:
        desired = proportional_counts(dict(synthetic_counts), target)
        if all(desired[key] <= len(by_bin[key]) for key in desired):
            break
        target -= 1
    if target == 0:
        raise RuntimeError("could not construct a length-matched UniRef50 population")
    selected = []
    for key in sorted(desired):
        selected.extend(by_bin[key][: desired[key]])
    return selected


def prepare(output_dir: Path, natural_sample_size: int = 100_000) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    brq_ids: set[str] = set()
    brq_rows = 0
    brq_counts: Counter[str] = Counter()
    for row in _dataset("backboneref", "BRq"):
        accession = str(row["accession"])
        brq_ids.add(accession)
        brq_counts[accession] += 1
        brq_rows += 1

    brn_ids: set[str] = set()

    def brn_rows():
        for row in _dataset("backboneref", "BRn"):
            brn_ids.add(str(row["accession"]))
            yield row

    representatives, brn_scan = choose_representatives(brn_rows())
    intersection = brn_ids & brq_ids
    if not intersection <= set(representatives):
        missing = intersection - set(representatives)
        raise RuntimeError(f"intersection accessions without valid representatives: {len(missing)}")

    brn_candidates = [
        replace(record, brq_member=record.backbone_id in brq_ids)
        for record in representatives.values()
    ]
    brn_candidates, duplicate_count = remove_cross_backbone_duplicates(brn_candidates)
    preferred_prefilter = [record for record in brn_candidates if record.brq_member]
    natural_pool = _natural_pool()
    natural = _length_matched_natural(natural_pool, preferred_prefilter, natural_sample_size)
    if len(natural) < 10_000:
        raise RuntimeError("natural reference sample is unexpectedly small")
    thresholds = derive_low_complexity_thresholds([sequence for _, sequence in natural])
    composition_comparison = {
        "natural_length_matched_uniref50": sequence_collection_summary(
            [sequence for _, sequence in natural]
        ),
        "synthetic_before_low_complexity_filter": sequence_collection_summary(
            [record.sequence for record in preferred_prefilter]
        ),
    }
    before_complexity = len(brn_candidates)
    brn_candidates = [
        record for record in brn_candidates if passes_low_complexity(record.sequence, thresholds)
    ]
    removed_complexity = before_complexity - len(brn_candidates)
    candidates = [record for record in brn_candidates if record.brq_member]
    preferred_removed_complexity = len(preferred_prefilter) - len(candidates)
    fallback_added = 0
    minimum_required = 54_000
    if len(candidates) < minimum_required:
        supplement = [record for record in brn_candidates if not record.brq_member]
        supplement.sort(key=lambda record: stable_hash(record.backbone_id, seed=29))
        fallback_added = minimum_required - len(candidates)
        candidates.extend(supplement[:fallback_added])
    if len(candidates) < minimum_required:
        raise RuntimeError(
            f"BRn contains only {len(candidates)} eligible unique backbones after the BRq fallback"
        )
    composition_comparison["synthetic_after_low_complexity_filter"] = (
        sequence_collection_summary([record.sequence for record in candidates])
    )
    natural_evaluation = _length_matched_natural(natural_pool, candidates, 2000)

    train10, train50, heldout = stratified_nested_sample(candidates)
    if not {x.backbone_id for x in train10} <={x.backbone_id for x in train50}:
        raise RuntimeError("10k load is not nested in 50k load")

    manifests = {}
    natural_records = [
        BackboneSequence(
            backbone_id=accession,
            sequence_id=accession,
            sequence=sequence,
            length=len(sequence),
            brn_member=False,
            brq_member=False,
            sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
            description="Dayhoff uniref50 test",
        )
        for accession, sequence in natural_evaluation
    ]
    for name, records, split in (
        ("train_10k", train10, "train"),
        ("train_50k", train50, "train"),
        ("heldout_candidates", heldout, "candidate"),
        ("natural_test", natural_records, "natural_test"),
    ):
        manifests[name] = write_manifest(
            output_dir / f"{name}.jsonl",
            records,
            split=split,
            extra={"selection_seed": 11, "one_sequence_per_backbone": True},
        )

    audit = {
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "brn_unique_backbones": len(brn_ids),
        "brq_unique_backbones": len(brq_ids),
        "brn_brq_intersection_backbones": len(intersection),
        "brq_rows": brq_rows,
        "brq_rows_per_accession_min": min(brq_counts.values()),
        "brq_rows_per_accession_max": max(brq_counts.values()),
        "brn_scan": brn_scan,
        "exact_duplicates_across_backbones_removed": duplicate_count,
        "malformed_records_removed": brn_scan["malformed_rows"],
        "invalid_length_records_removed": brn_scan["invalid_length_rows"],
        "low_complexity_thresholds": thresholds,
        "low_complexity_backbones_removed": removed_complexity,
        "low_complexity_preferred_backbones_removed": preferred_removed_complexity,
        "composition_complexity_comparison": composition_comparison,
        "brn_only_fallback_backbones_added": fallback_added,
        "eligible_population": population_summary(candidates),
        "natural_reference": {
            "sample_size": len(natural),
            "selection": "32-residue-bin length matching, then SHA-256(accession,sequence), seed 5",
        },
        "manifests": manifests,
    }
    (output_dir / "corpus_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    intersection_payload = "\n".join(sorted(intersection)) + "\n"
    (output_dir / "intersection_accessions.txt").write_text(intersection_payload, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--natural-sample-size", type=int, default=100_000)
    args = parser.parse_args()
    prepare(args.output_dir, args.natural_sample_size)


if __name__ == "__main__":
    main()
