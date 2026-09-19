from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .data import SequenceRecord, VariantRecord, load_manifest


def _domain_records(records: Iterable[SequenceRecord]) -> dict[str, list[SequenceRecord]]:
    grouped: dict[str, list[SequenceRecord]] = defaultdict(list)
    for record in records:
        grouped[record.domain_id].append(record)
    return dict(grouped)


def _evenly_spaced(items: Sequence, count: int) -> list:
    if len(items) < count:
        raise ValueError(f"Need {count} items, only {len(items)} are available")
    indices = np.linspace(0, len(items) - 1, count).round().astype(int)
    if len(set(indices)) != count:
        raise RuntimeError("Panel spacing produced duplicate indices")
    return [items[index] for index in indices]


def _test_tertiles(domains: dict[str, list[SequenceRecord]]) -> dict[str, list[str]]:
    ordered = sorted(domains, key=lambda domain: (domains[domain][0].d_struct, domain))
    chunks = np.array_split(np.asarray(ordered, dtype=object), 3)
    return {label: list(chunk) for label, chunk in zip(("near", "middle", "far"), chunks)}


def _select_sequences(records: Sequence[SequenceRecord]) -> list[str]:
    stabilizing = sorted((r for r in records if r.ddg < 0), key=lambda r: (r.ddg, r.sequence_id))
    destabilizing = sorted((r for r in records if r.ddg > 0), key=lambda r: (r.ddg, r.sequence_id))
    return [r.sequence_id for r in _evenly_spaced(stabilizing, min(8, len(stabilizing)))] + [
        r.sequence_id for r in _evenly_spaced(destabilizing, min(8, len(destabilizing)))
    ]


_select_variants = _select_sequences


def _stratified_train_domains(domains: dict[str, list[SequenceRecord]], count: int = 48) -> list[str]:
    if len(domains) < count:
        raise ValueError(f"Need {count} training domains, only {len(domains)} are available")
    identifiers = sorted(domains)
    lengths = {domain: len(domains[domain][0].native_sequence) for domain in identifiers}
    sizes = {domain: len(domains[domain]) for domain in identifiers}

    def quartiles(values: dict[str, int]) -> dict[str, int]:
        ordered = sorted(identifiers, key=lambda domain: (values[domain], domain))
        return {domain: min(3, (rank * 4) // len(ordered)) for rank, domain in enumerate(ordered)}

    length_bin, size_bin = quartiles(lengths), quartiles(sizes)
    bins: dict[tuple[int, int], list[str]] = defaultdict(list)
    for domain in identifiers:
        bins[(length_bin[domain], size_bin[domain])].append(domain)
    for key in bins:
        bins[key].sort(key=lambda domain: (lengths[domain], sizes[domain], domain))
    selected: list[str] = []
    # Round-robin over the 4x4 grid prevents large central bins from erasing
    # length or domain-size strata. Empty cells simply contribute no domain.
    while len(selected) < count:
        progress = False
        for key in [(i, j) for i in range(4) for j in range(4)]:
            if bins[key] and len(selected) < count:
                selected.append(bins[key].pop(0))
                progress = True
        if not progress:
            raise RuntimeError("Unable to fill stratified training panel")
    return selected


def construct_panels(records: Sequence[SequenceRecord]) -> dict:
    train_domains = _domain_records(r for r in records if r.split == "train")
    test_domains = _domain_records(r for r in records if r.split == "test")
    train_selected = _stratified_train_domains(train_domains, 48)
    tertiles = _test_tertiles(test_domains)
    test_selected: dict[str, list[str]] = {}
    for tier, domain_ids in tertiles.items():
        ordered = sorted(domain_ids, key=lambda d: (len(test_domains[d][0].native_sequence), d))
        test_selected[tier] = _evenly_spaced(ordered, 16)
    generation: dict[str, list[str]] = {}
    for tier, domain_ids in test_selected.items():
        ordered = sorted(domain_ids, key=lambda d: (len(test_domains[d][0].native_sequence), d))
        generation[tier] = _evenly_spaced(ordered, 6)
    return {
        "selection_inputs": {
            "train_diagnostic": ["sequence_length", "domain_size"],
            "test_diagnostic": ["structural_distance", "sequence_length"],
            "generation": ["structural_distance", "sequence_length"],
        },
        "train_diagnostic": {
            domain: _select_sequences(train_domains[domain]) for domain in train_selected
        },
        "test_diagnostic": {
            tier: {domain: _select_sequences(test_domains[domain]) for domain in domains}
            for tier, domains in test_selected.items()
        },
        "generation": generation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    panel = construct_panels(load_manifest(args.manifest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(panel, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
