"""Outcome-blind construction and validation of an untouched STP panel."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

from eval_uspto_mit_five_view_a6000 import (
    canonical_source,
    canonicalize_official,
    file_sha256,
    read_official_groups,
)


PANEL_SALT = "clm-jepa-stp-confirmation-v1|20260904"
OFFICIAL_TEST_SHA256 = "c2f4a3b731c4ed0a35b1c38fbff9563aee0e61064bcedeca555f335f69964945"


def chemical_pair_id(source: str, target: str) -> str:
    return hashlib.sha256(f"{source}>>{target}".encode("utf-8")).hexdigest()


def stable_selection_key(pair_id: str, salt: str = PANEL_SALT) -> str:
    return hashlib.sha256(f"{salt}|{pair_id}".encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def pair_ids_from_manifest(path: Path) -> tuple[set[int], set[str]]:
    groups, pairs = set(), set()
    for row in read_jsonl(path):
        groups.add(int(row["official_group_index"]))
        source = row.get("canonical_source", "")
        target = row.get("canonical_target", "")
        if source and target:
            pairs.add(chemical_pair_id(source, target))
    return groups, pairs


def pair_ids_from_csv(path: Path) -> set[str]:
    pairs: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            source = row.get("source_identity") or canonical_source(row["source"])
            target = row.get("target_identity") or canonicalize_official(
                row["target"], clear_map=True
            )
            if source and target:
                pairs.add(chemical_pair_id(source, target))
    return pairs


def build_untouched_panel(
    *,
    official_test: Path,
    historical_manifests: list[Path],
    excluded_csvs: list[Path],
    output: Path,
    metadata_output: Path,
    exclusion_output: Path,
    reactions: int = 1280,
    processes: int = 6,
    salt: str = PANEL_SALT,
) -> dict:
    observed_test_hash = file_sha256(official_test)
    if observed_test_hash != OFFICIAL_TEST_SHA256:
        raise ValueError(
            f"official test SHA-256 mismatch: {observed_test_hash}; "
            f"expected {OFFICIAL_TEST_SHA256}"
        )

    excluded_groups: set[int] = set()
    excluded_pairs: set[str] = set()
    inputs = []
    for path in historical_manifests:
        groups, pairs = pair_ids_from_manifest(path)
        excluded_groups.update(groups)
        excluded_pairs.update(pairs)
        inputs.append({
            "kind": "historical_official_manifest",
            "path": str(path),
            "sha256": file_sha256(path),
            "official_groups": len(groups),
            "chemical_pairs": len(pairs),
        })
    for path in excluded_csvs:
        pairs = pair_ids_from_csv(path)
        excluded_pairs.update(pairs)
        inputs.append({
            "kind": "train_or_probe_csv",
            "path": str(path),
            "sha256": file_sha256(path),
            "chemical_pairs": len(pairs),
        })

    candidates = []
    invalid_or_inconsistent = 0
    excluded_by_group = 0
    excluded_by_pair = 0
    duplicate_pair = 0
    seen_candidate_pairs: set[str] = set()
    for group in read_official_groups(official_test, processes):
        if int(group["official_group_index"]) in excluded_groups:
            excluded_by_group += 1
            continue
        source = group["canonical_source"]
        target = group["canonical_target"]
        if not source or not target or not group["source_canonicalization_consistent"]:
            invalid_or_inconsistent += 1
            continue
        pair_id = chemical_pair_id(source, target)
        if pair_id in excluded_pairs:
            excluded_by_pair += 1
            continue
        if pair_id in seen_candidate_pairs:
            duplicate_pair += 1
            continue
        seen_candidate_pairs.add(pair_id)
        item = dict(group)
        item["chemical_pair_id"] = pair_id
        item["selection_key"] = stable_selection_key(pair_id, salt)
        candidates.append(item)

    candidates.sort(
        key=lambda item: (
            item["selection_key"], item["chemical_pair_id"], item["official_group_index"]
        )
    )
    if len(candidates) < reactions:
        raise ValueError(f"only {len(candidates)} untouched unique candidates remain")
    selected = candidates[:reactions]
    for panel_index, row in enumerate(selected):
        row["panel_index"] = panel_index
        row["confirmation_panel_version"] = 1

    selected_groups = {int(row["official_group_index"]) for row in selected}
    selected_pairs = {row["chemical_pair_id"] for row in selected}
    example_ids = [value for row in selected for value in row["example_ids"]]
    checks = {
        "reactions": len(selected),
        "unique_official_groups": len(selected_groups),
        "unique_chemical_pairs": len(selected_pairs),
        "example_ids": len(example_ids),
        "unique_example_ids": len(set(example_ids)),
        "historical_group_overlap": len(selected_groups & excluded_groups),
        "excluded_chemical_pair_overlap": len(selected_pairs & excluded_pairs),
        "panel_indices_exact": [row["panel_index"] for row in selected]
        == list(range(reactions)),
        "five_views_each": all(
            len(row["sources"]) == len(row["targets"]) == len(row["example_ids"]) == 5
            for row in selected
        ),
    }
    required = {
        "reactions": reactions,
        "unique_official_groups": reactions,
        "unique_chemical_pairs": reactions,
        "example_ids": reactions * 5,
        "unique_example_ids": reactions * 5,
        "historical_group_overlap": 0,
        "excluded_chemical_pair_overlap": 0,
        "panel_indices_exact": True,
        "five_views_each": True,
    }
    if checks != required:
        raise ValueError(f"untouched-panel integrity failure: {checks}")

    write_jsonl(output, selected)
    exclusion = {
        "schema_version": 1,
        "official_group_indices": sorted(excluded_groups),
        "chemical_pair_ids": sorted(excluded_pairs),
        "inputs": inputs,
    }
    exclusion_output.parent.mkdir(parents=True, exist_ok=True)
    with exclusion_output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(exclusion, indent=2, sort_keys=True) + "\n")
    metadata = {
        "schema_version": 1,
        "created_before_checkpoint_inference": True,
        "selection_uses_model_outputs": False,
        "official_test": str(official_test),
        "official_test_sha256": observed_test_hash,
        "selection": "lowest outcome-blind salted SHA-256 key after exclusions",
        "selection_salt": salt,
        "requested_reactions": reactions,
        "eligible_unique_reactions": len(candidates),
        "excluded_historical_official_groups": len(excluded_groups),
        "excluded_known_chemical_pairs": len(excluded_pairs),
        "excluded_by_official_group": excluded_by_group,
        "excluded_by_chemical_pair": excluded_by_pair,
        "invalid_or_inconsistent_official_groups": invalid_or_inconsistent,
        "duplicate_eligible_chemical_pairs": duplicate_pair,
        "integrity_checks": checks,
        "inputs": inputs,
        "manifest": str(output),
        "manifest_sha256": file_sha256(output),
        "exclusion_ledger": str(exclusion_output),
        "exclusion_ledger_sha256": file_sha256(exclusion_output),
    }
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    with metadata_output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def validate_panel(
    panel: Path, metadata: Path, exclusion: Path, expected_reactions: int = 1280
) -> dict:
    rows = read_jsonl(panel)
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    ledger = json.loads(exclusion.read_text(encoding="utf-8"))
    groups = {int(row["official_group_index"]) for row in rows}
    pairs = {row["chemical_pair_id"] for row in rows}
    excluded_groups = set(ledger["official_group_indices"])
    excluded_pairs = set(ledger["chemical_pair_ids"])
    checks = {
        "manifest_sha256_matches_metadata": file_sha256(panel)
        == meta["manifest_sha256"],
        "ledger_sha256_matches_metadata": file_sha256(exclusion)
        == meta["exclusion_ledger_sha256"],
        "reactions": len(rows),
        "unique_groups": len(groups),
        "unique_pairs": len(pairs),
        "group_overlap": len(groups & excluded_groups),
        "pair_overlap": len(pairs & excluded_pairs),
        "indices_exact": [row["panel_index"] for row in rows]
        == list(range(expected_reactions)),
        "views_exact": all(len(row["sources"]) == len(row["targets"]) == 5 for row in rows),
    }
    if checks != {
        "manifest_sha256_matches_metadata": True,
        "ledger_sha256_matches_metadata": True,
        "reactions": expected_reactions,
        "unique_groups": expected_reactions,
        "unique_pairs": expected_reactions,
        "group_overlap": 0,
        "pair_overlap": 0,
        "indices_exact": True,
        "views_exact": True,
    }:
        raise ValueError(f"frozen confirmation panel changed: {checks}")
    return checks
