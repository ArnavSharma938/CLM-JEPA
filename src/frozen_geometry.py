"""Frozen, all-layer ChemFM trajectory-geometry diagnostic.

This module never constructs a loss or optimizer.  It labels chemical events in
the exact maintained forward-reaction serialization, pairs each event with an
ordinary token from the same reaction segment, and compares local curvature and
ordered-span (STP-style) alignment at every hidden layer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import scipy
import torch
import transformers
from rdkit import Chem, rdBase
from rdkit.Chem import rdFMCS
from scipy.optimize import linear_sum_assignment

from chemfm import (
    END,
    MODEL_DIR,
    PRODUCT_START,
    REACTANT_START,
    TOKENIZER_DIR,
    ReactionCollator,
    canonicalize,
    load_lora_model,
    load_reaction_tokenizer,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = (
    ROOT / "data" / "clm_jepa_uspto_mit_official_endpoint"
    / "prespecified_stage1_256.jsonl"
)
DEFAULT_OUTPUT = ROOT / "runs" / "diagnostics" / "frozen_chemfm_stp_geometry"
DEFAULT_STEREO_SUPPLEMENT = ROOT / "data" / "uspto_50k" / "test_r_smiles.csv"
EVENT_TYPES = ("ring_closure", "branch", "stereochemistry", "motif", "reaction_center")
SPAN_BINS = ("adjacent", "short_3_8", "medium_9_24", "long_25_plus")
STP_REFERENCE_COMMIT = "ea0017c654ad917066ff32afc88276bea8ca5f7e"

# A deliberately compact, named chemistry vocabulary.  Matches are collapsed
# to one event at the last atom revealed by the serialized occurrence.
MOTIF_SMARTS = {
    "carbonyl": "[CX3]=[OX1]",
    "carboxyl": "[CX3](=[OX1])[OX2H0-,OX2H1]",
    "ester": "[CX3](=[OX1])[OX2][#6]",
    "amide": "[NX3][CX3](=[OX1])",
    "amine": "[NX3;H0,H1,H2;!$(N-C=O)]",
    "alcohol_or_phenol": "[OX2H][#6]",
    "ether": "[#6][OX2H0][#6]",
    "nitrile": "[CX2]#[NX1]",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "sulfonyl": "[SX4](=[OX1])(=[OX1])",
    "phosphoryl": "[PX4](=[OX1])",
    "carbon_halogen": "[#6][F,Cl,Br,I]",
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#[CX2]",
}
MOTIF_QUERIES = {name: Chem.MolFromSmarts(smarts) for name, smarts in MOTIF_SMARTS.items()}

# Outside brackets, SMILES permits only the organic subset; Br and Cl are its
# two-character elements.  Treating e.g. ``Cn`` as copernicium would wrongly
# merge an aliphatic carbon followed by aromatic nitrogen.
_TWO_LETTER_ATOMS = {"Br", "Cl"}


@dataclass
class TokenInfo:
    index: int
    start: int
    end: int
    text: str
    segment: str
    segment_rank: int
    token_class: str
    branch_depth: int
    component: int
    events: set[str] = field(default_factory=set)
    details: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))


@dataclass
class Example:
    panel_index: int
    reaction_identity: str
    source: str
    target: str
    text: str
    input_ids: list[int]
    tokens: list[TokenInfo]
    reaction_center_metadata: dict
    sample_origin: str = "prespecified_256"
    pairs: list[dict] = field(default_factory=list)


def read_panel(path: Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if limit is not None:
        rows = rows[:limit]
    identities = [row["reaction_identity"] for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("diagnostic panel contains duplicate reaction identities")
    return rows


def read_stereo_supplement(path: Path, count: int, seed: int) -> list[dict]:
    """Select unique stereochemical reactions from the official USPTO-50K test."""
    if count <= 0:
        return []
    unique = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle)):
            # The official ChemFM USPTO-50K release has 20 consecutive
            # R-SMILES views per reaction; use view zero before enrichment so
            # highly augmented reactions cannot receive extra sampling weight.
            if row_index % 20:
                continue
            if not any(symbol in row["source"] or symbol in row["target"] for symbol in ("@", "/", "\\")):
                continue
            source = canonicalize(row["source"])
            target = canonicalize(row["target"])
            if not source or not target or not any(symbol in source or symbol in target for symbol in ("@", "/", "\\")):
                continue
            identity = hashlib.sha256(f"{source}>>{target}".encode("utf-8")).hexdigest()
            unique.setdefault(identity, {
                "reaction_identity": f"uspto50k-stereo-{identity}",
                "canonical_source": source,
                "canonical_target": target,
            })
    candidates = [unique[key] for key in sorted(unique)]
    if len(candidates) < count:
        raise ValueError(f"requested {count} stereochemical reactions but only found {len(candidates)}")
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(candidates), size=count, replace=False)
    return [candidates[index] for index in sorted(selected.tolist())]


def _atom_spans(smiles: str) -> list[tuple[int, int]]:
    """Return SMILES atom lexemes in RDKit atom-index order."""
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(smiles):
        char = smiles[index]
        if char == "[":
            end = smiles.find("]", index + 1)
            if end < 0:
                raise ValueError(f"unterminated bracket atom in {smiles!r}")
            spans.append((index, end + 1))
            index = end + 1
            continue
        two = smiles[index:index + 2]
        if two in {"se", "as"} or two in _TWO_LETTER_ATOMS:
            spans.append((index, index + 2))
            index += 2
            continue
        if char in "BCNOPSFIbcnops*":
            spans.append((index, index + 1))
        index += 1
    return spans


def _component_spans(smiles: str) -> list[tuple[int, int, str]]:
    result = []
    start = 0
    bracket_depth = 0
    for index, char in enumerate(smiles):
        if char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
        elif char == "." and bracket_depth == 0:
            result.append((start, index, smiles[start:index]))
            start = index + 1
    result.append((start, len(smiles), smiles[start:]))
    return result


def _overlap_token(tokens: Sequence[TokenInfo], segment: str, start: int, end: int) -> int | None:
    candidates = [
        token.index for token in tokens
        if token.segment == segment and token.start < end and token.end > start
    ]
    return candidates[-1] if candidates else None


def _infer_reaction_center(source: str, target: str) -> tuple[set[int], set[int], dict]:
    """Infer a transparent scaffold graph-difference center for unmapped data.

    The repository endpoint removed atom maps.  We therefore select the
    source/target component pair with the largest RDKit MCS, then mark unmatched
    atoms and mapped atoms touching an unmatched or bond-order-changed edge.
    Other source components are intentionally not called reaction centers.
    """
    source_components = _component_spans(source)
    target_components = _component_spans(target)
    source_atom_offset = 0
    source_offsets = []
    for _, _, value in source_components:
        source_offsets.append(source_atom_offset)
        source_atom_offset += len(_atom_spans(value))
    target_atom_offset = 0
    target_offsets = []
    for _, _, value in target_components:
        target_offsets.append(target_atom_offset)
        target_atom_offset += len(_atom_spans(value))

    candidates = []
    for source_index, (_, _, source_smiles) in enumerate(source_components):
        source_mol = Chem.MolFromSmiles(source_smiles)
        if source_mol is None:
            continue
        for target_index, (_, _, target_smiles) in enumerate(target_components):
            target_mol = Chem.MolFromSmiles(target_smiles)
            if target_mol is None:
                continue
            result = rdFMCS.FindMCS(
                [source_mol, target_mol],
                atomCompare=rdFMCS.AtomCompare.CompareElements,
                bondCompare=rdFMCS.BondCompare.CompareOrder,
                ringMatchesRingOnly=True,
                completeRingsOnly=False,
                timeout=2,
            )
            candidates.append((
                int(result.numAtoms), int(result.numBonds), target_mol.GetNumHeavyAtoms(),
                source_mol.GetNumHeavyAtoms(), -source_index, -target_index,
                source_index, target_index, source_mol, target_mol, result.smartsString,
            ))
    if not candidates or max(item[0] for item in candidates) < 2:
        return set(), set(), {"status": "no_reliable_mcs"}
    best = max(candidates)
    overlap_atoms, overlap_bonds, _, _, _, _, source_index, target_index, source_mol, target_mol, smarts = best
    query = Chem.MolFromSmarts(smarts)
    source_match = source_mol.GetSubstructMatch(query) if query is not None else ()
    target_match = target_mol.GetSubstructMatch(query) if query is not None else ()
    if not source_match or len(source_match) != len(target_match):
        return set(), set(), {"status": "mcs_match_failed", "overlap_atoms": overlap_atoms}
    source_to_target = dict(zip(source_match, target_match))
    target_to_source = {target_atom: source_atom for source_atom, target_atom in source_to_target.items()}
    mapped_source = set(source_to_target)
    mapped_target = set(target_to_source)
    unmatched_source = set(range(source_mol.GetNumAtoms())) - mapped_source
    unmatched_target = set(range(target_mol.GetNumAtoms())) - mapped_target
    source_center: set[int] = set()
    target_center: set[int] = set()

    for source_atom, target_atom in source_to_target.items():
        source_neighbors = {neighbor.GetIdx() for neighbor in source_mol.GetAtomWithIdx(source_atom).GetNeighbors()}
        target_neighbors = {neighbor.GetIdx() for neighbor in target_mol.GetAtomWithIdx(target_atom).GetNeighbors()}
        source_frontier = source_neighbors & unmatched_source
        target_frontier = target_neighbors & unmatched_target
        if source_frontier or target_frontier:
            source_center.add(source_atom)
            target_center.add(target_atom)
            # Mark only the first graph-difference frontier, not an entire
            # introduced/deleted fragment whose internal atoms are not centers.
            source_center.update(source_frontier)
            target_center.update(target_frontier)
        for source_neighbor in source_neighbors & mapped_source:
            target_neighbor = source_to_target[source_neighbor]
            source_bond = source_mol.GetBondBetweenAtoms(source_atom, source_neighbor)
            target_bond = target_mol.GetBondBetweenAtoms(target_atom, target_neighbor)
            if target_bond is None or source_bond.GetBondType() != target_bond.GetBondType():
                source_center.update((source_atom, source_neighbor))
                target_center.update((target_atom, target_neighbor))

    source_global = {source_offsets[source_index] + atom for atom in source_center}
    target_global = {target_offsets[target_index] + atom for atom in target_center}
    metadata = {
        "status": "mcs_inferred",
        "source_component": source_index,
        "target_component": target_index,
        "overlap_atoms": overlap_atoms,
        "overlap_bonds": overlap_bonds,
        "source_component_atoms": source_mol.GetNumAtoms(),
        "target_component_atoms": target_mol.GetNumAtoms(),
        "source_center_atoms": len(source_center),
        "target_center_atoms": len(target_center),
    }
    return source_global, target_global, metadata


def _mark_smiles_events(
    tokens: list[TokenInfo], segment: str, smiles: str, absolute_start: int,
    reaction_center_atoms: set[int],
) -> None:
    atom_spans = _atom_spans(smiles)
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() != len(atom_spans):
        raise ValueError(f"RDKit/token atom-order mismatch for {smiles!r}")

    # Ring labels alternate open/close; reused labels therefore remain valid.
    open_labels: set[str] = set()
    index = 0
    bracket_depth = 0
    while index < len(smiles):
        char = smiles[index]
        if char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
        elif bracket_depth == 0 and (char.isdigit() or char == "%"):
            end = index + 1
            if char == "%" and index + 2 < len(smiles) and smiles[index + 1:index + 3].isdigit():
                end = index + 3
            label = smiles[index:end]
            if label in open_labels:
                token_index = _overlap_token(tokens, segment, absolute_start + index, absolute_start + end)
                if token_index is not None:
                    tokens[token_index].events.add("ring_closure")
                    tokens[token_index].details["ring_closure"].append(label)
                open_labels.remove(label)
            else:
                open_labels.add(label)
            index = end
            continue
        index += 1

    for relative_index, char in enumerate(smiles):
        if char in "()":
            token_index = _overlap_token(
                tokens, segment, absolute_start + relative_index, absolute_start + relative_index + 1,
            )
            if token_index is not None:
                tokens[token_index].events.add("branch")
                tokens[token_index].details["branch"].append("open" if char == "(" else "close")
        if char in "@/\\":
            token_index = _overlap_token(
                tokens, segment, absolute_start + relative_index, absolute_start + relative_index + 1,
            )
            if token_index is not None:
                tokens[token_index].events.add("stereochemistry")
                tokens[token_index].details["stereochemistry"].append(char)

    motif_labels: dict[int, set[str]] = defaultdict(set)
    for name, query in MOTIF_QUERIES.items():
        if query is None:
            continue
        for match in molecule.GetSubstructMatches(query, uniquify=True):
            completion_atom = max(match)
            motif_labels[completion_atom].add(name)
    for atom_index, labels in motif_labels.items():
        start, end = atom_spans[atom_index]
        token_index = _overlap_token(tokens, segment, absolute_start + start, absolute_start + end)
        if token_index is not None:
            tokens[token_index].events.add("motif")
            tokens[token_index].details["motif"].extend(sorted(labels))

    for atom_index in reaction_center_atoms:
        if atom_index >= len(atom_spans):
            continue
        start, end = atom_spans[atom_index]
        token_index = _overlap_token(tokens, segment, absolute_start + start, absolute_start + end)
        if token_index is not None:
            tokens[token_index].events.add("reaction_center")
            tokens[token_index].details["reaction_center"].append("mcs_graph_difference")


def annotate_example(
    tokenizer, row: dict, panel_index: int, sample_origin: str = "prespecified_256",
    infer_reaction_center: bool = True,
) -> Example:
    source = row["canonical_source"]
    target = row["canonical_target"]
    text = f"{REACTANT_START}{source}{END}{PRODUCT_START}{target}{END}"
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    collated = ReactionCollator(tokenizer)([{"src": source, "tgt": target}])
    expected_ids = collated["input_ids"][0].tolist()
    if encoded["input_ids"] != expected_ids:
        raise ValueError("whole-string offset tokenization differs from maintained ReactionCollator")

    source_start = len(REACTANT_START)
    source_end = source_start + len(source)
    target_start = source_end + len(END) + len(PRODUCT_START)
    target_end = target_start + len(target)
    source_atom_absolute = [(source_start + a, source_start + b) for a, b in _atom_spans(source)]
    target_atom_absolute = [(target_start + a, target_start + b) for a, b in _atom_spans(target)]

    tokens: list[TokenInfo] = []
    segment_counts = defaultdict(int)
    branch_depth = defaultdict(int)
    component = defaultdict(int)
    for token_index, (start, end) in enumerate(encoded["offset_mapping"]):
        if start >= source_start and end <= source_end:
            segment = "source"
            atom_spans = source_atom_absolute
        elif start >= target_start and end <= target_end:
            segment = "target"
            atom_spans = target_atom_absolute
        else:
            segment = "marker"
            atom_spans = []
        value = text[start:end]
        if segment != "marker" and value == ")":
            branch_depth[segment] = max(0, branch_depth[segment] - 1)
        current_depth = branch_depth[segment]
        token_class = "atom" if any(start < b and end > a for a, b in atom_spans) else "syntax"
        if segment == "marker":
            token_class = "marker"
        tokens.append(TokenInfo(
            index=token_index, start=start, end=end, text=value, segment=segment,
            segment_rank=segment_counts[segment], token_class=token_class,
            branch_depth=current_depth, component=component[segment],
        ))
        segment_counts[segment] += 1
        if segment != "marker" and value == "(":
            branch_depth[segment] += 1
        elif segment != "marker" and value == ".":
            component[segment] += 1

    if infer_reaction_center:
        source_center, target_center, center_metadata = _infer_reaction_center(source, target)
    else:
        source_center, target_center, center_metadata = set(), set(), {"status": "not_requested"}
    _mark_smiles_events(tokens, "source", source, source_start, source_center)
    _mark_smiles_events(tokens, "target", target, target_start, target_center)
    return Example(
        panel_index=panel_index,
        reaction_identity=row["reaction_identity"],
        source=source,
        target=target,
        text=text,
        input_ids=encoded["input_ids"],
        tokens=tokens,
        reaction_center_metadata=center_metadata,
        sample_origin=sample_origin,
    )


def _expanded_assignment(events: list[TokenInfo], controls: list[TokenInfo]) -> list[tuple[TokenInfo, TokenInfo]]:
    if not events or not controls:
        return []
    repeats = math.ceil(len(events) / len(controls))
    slots = [(control, reuse) for reuse in range(repeats) for control in controls]
    cost = np.empty((len(events), len(slots)), dtype=np.float64)
    segment_length = max(token.segment_rank for token in events + controls) + 1
    for event_index, event in enumerate(events):
        for slot_index, (control, reuse) in enumerate(slots):
            cost[event_index, slot_index] = (
                100.0 * (event.token_class != control.token_class)
                + 8.0 * abs(event.segment_rank - control.segment_rank) / max(1, segment_length - 1)
                + 2.0 * abs(event.branch_depth - control.branch_depth)
                + 1.0 * (event.component != control.component)
                + 0.25 * abs(len(event.text) - len(control.text))
                + 3.0 * reuse
                + 1e-7 * slot_index
            )
    rows, columns = linear_sum_assignment(cost)
    return [(events[row], slots[column][0]) for row, column in zip(rows, columns)]


def _span_bin(total_span: int) -> int:
    if total_span == 2:
        return 0
    if total_span <= 8:
        return 1
    if total_span <= 24:
        return 2
    return 3


def match_controls_and_anchors(
    example: Example, seed: int, anchors_per_event: int,
    categories: Sequence[str] = EVENT_TYPES,
) -> dict:
    rng = np.random.default_rng(seed + example.panel_index * 1_000_003)
    diagnostics = {category: {"events": 0, "matched": 0, "class_matches": 0} for category in categories}
    pair_counter = 0
    for segment in ("source", "target"):
        segment_tokens = [token for token in example.tokens if token.segment == segment]
        ranks = {token.index: rank for rank, token in enumerate(segment_tokens)}
        ordinary = [
            token for token in segment_tokens if not token.events
        ]
        for category in categories:
            events = [
                token for token in segment_tokens if category in token.events
            ]
            diagnostics[category]["events"] += len(events)
            assignments = []
            for token_class in sorted({event.token_class for event in events}):
                class_events = [event for event in events if event.token_class == token_class]
                class_controls = [control for control in ordinary if control.token_class == token_class]
                assignments.extend(_expanded_assignment(class_events, class_controls or ordinary))
            for event, control in assignments:
                event_rank = ranks[event.index]
                control_rank = ranks[control.index]
                # STP operates on the serialized sequence.  Anchors may cross
                # segment markers; event and control retain identical absolute
                # token offsets, while the centers remain same-segment matched.
                left_room = min(event.index, control.index)
                right_room = min(
                    len(example.tokens) - 1 - event.index,
                    len(example.tokens) - 1 - control.index,
                )
                if left_room < 1 or right_room < 1:
                    continue
                anchors = []
                for _ in range(anchors_per_event):
                    left = int(rng.integers(1, left_room + 1))
                    right = int(rng.integers(1, right_room + 1))
                    anchors.append({
                        "left": left,
                        "right": right,
                        "span": left + right,
                        "span_bin": _span_bin(left + right),
                        "event_s": event.index - left,
                        "event_t": event.index + right,
                        "control_s": control.index - left,
                        "control_t": control.index + right,
                    })
                example.pairs.append({
                    "pair_id": f"{example.panel_index}:{pair_counter}",
                    "panel_index": example.panel_index,
                    "reaction_identity": example.reaction_identity,
                    "sample_origin": example.sample_origin,
                    "category": category,
                    "segment": segment,
                    "event_index": event.index,
                    "control_index": control.index,
                    "event_text": event.text,
                    "control_text": control.text,
                    "event_class": event.token_class,
                    "control_class": control.token_class,
                    "event_details": dict(event.details),
                    "anchors": anchors,
                })
                pair_counter += 1
                diagnostics[category]["matched"] += 1
                diagnostics[category]["class_matches"] += int(event.token_class == control.token_class)
    return diagnostics


def one_minus_cosine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = first.float()
    second = second.float()
    numerator = (first * second).sum(dim=-1)
    denominator = first.square().sum(dim=-1).sqrt() * second.square().sum(dim=-1).sqrt()
    values = 1.0 - numerator / denominator.clamp_min(1e-12)
    return torch.where(denominator > 1e-12, values, torch.full_like(values, torch.nan))


def _compute_example_geometry(
    hidden_states: Sequence[torch.Tensor], example_batch_index: int,
    pairs: list[dict], result_arrays: dict[str, np.ndarray], pair_offset: int,
) -> None:
    if not pairs:
        return
    event_indices = torch.tensor([pair["event_index"] for pair in pairs], device=hidden_states[0].device)
    control_indices = torch.tensor([pair["control_index"] for pair in pairs], device=hidden_states[0].device)
    flat_pair_indices = []
    event_s = []
    event_r = []
    event_t = []
    control_s = []
    control_r = []
    control_t = []
    bins = []
    for local_pair_index, pair in enumerate(pairs):
        for anchor in pair["anchors"]:
            flat_pair_indices.append(local_pair_index)
            event_s.append(anchor["event_s"])
            event_r.append(pair["event_index"])
            event_t.append(anchor["event_t"])
            control_s.append(anchor["control_s"])
            control_r.append(pair["control_index"])
            control_t.append(anchor["control_t"])
            bins.append(anchor["span_bin"])
    device = hidden_states[0].device
    index_tensors = [
        torch.tensor(values, device=device, dtype=torch.long)
        for values in (event_s, event_r, event_t, control_s, control_r, control_t)
    ]
    flat_pair_indices_np = np.asarray(flat_pair_indices, dtype=np.int64)
    bins_np = np.asarray(bins, dtype=np.int64)
    rows = slice(pair_offset, pair_offset + len(pairs))
    for layer, state_batch in enumerate(hidden_states):
        state = state_batch[example_batch_index]
        event_before = state.index_select(0, event_indices - 1)
        event_here = state.index_select(0, event_indices)
        event_after = state.index_select(0, event_indices + 1)
        control_before = state.index_select(0, control_indices - 1)
        control_here = state.index_select(0, control_indices)
        control_after = state.index_select(0, control_indices + 1)
        result_arrays["local_event"][rows, layer] = one_minus_cosine(
            event_here - event_before, event_after - event_here,
        ).cpu().numpy()
        result_arrays["local_control"][rows, layer] = one_minus_cosine(
            control_here - control_before, control_after - control_here,
        ).cpu().numpy()

        es, er, et, cs, cr, ct = [state.index_select(0, indices) for indices in index_tensors]
        event_values = one_minus_cosine(er - es, et - er).cpu().numpy()
        control_values = one_minus_cosine(cr - cs, ct - cr).cpu().numpy()
        for local_pair_index in range(len(pairs)):
            mask = (
                (flat_pair_indices_np == local_pair_index)
                & np.isfinite(event_values) & np.isfinite(control_values)
            )
            global_pair_index = pair_offset + local_pair_index
            result_arrays["semi_valid_counts"][global_pair_index, layer] = int(mask.sum())
            if mask.any():
                result_arrays["semi_event"][global_pair_index, layer] = float(event_values[mask].mean())
                result_arrays["semi_control"][global_pair_index, layer] = float(control_values[mask].mean())
            for bin_index in range(len(SPAN_BINS)):
                bin_mask = mask & (bins_np == bin_index)
                result_arrays["semi_bin_valid_counts"][global_pair_index, layer, bin_index] = int(bin_mask.sum())
                if bin_mask.any():
                    result_arrays["semi_event_bins"][global_pair_index, layer, bin_index] = float(event_values[bin_mask].mean())
                    result_arrays["semi_control_bins"][global_pair_index, layer, bin_index] = float(control_values[bin_mask].mean())


def _compute_batch_geometry(
    hidden_states: Sequence[torch.Tensor], batch_examples: Sequence[Example],
    result_arrays: dict[str, np.ndarray], pair_offsets: dict[int, int],
    anchor_chunk_size: int = 8192,
) -> None:
    """Vectorized batch geometry with bounded-memory anchor chunks."""
    batch_pairs = []
    global_pair_indices = []
    pair_batch_indices = []
    for batch_index, example in enumerate(batch_examples):
        for local_index, pair in enumerate(example.pairs):
            batch_pairs.append(pair)
            global_pair_indices.append(pair_offsets[example.panel_index] + local_index)
            pair_batch_indices.append(batch_index)
    if not batch_pairs:
        return
    device = hidden_states[0].device
    global_pair_indices_np = np.asarray(global_pair_indices, dtype=np.int64)
    pair_batch = torch.tensor(pair_batch_indices, device=device, dtype=torch.long)
    event_indices = torch.tensor([pair["event_index"] for pair in batch_pairs], device=device)
    control_indices = torch.tensor([pair["control_index"] for pair in batch_pairs], device=device)

    anchor_pair = []
    anchor_batch = []
    event_s = []
    event_r = []
    event_t = []
    control_s = []
    control_r = []
    control_t = []
    bins = []
    for local_pair_index, (batch_index, pair) in enumerate(zip(pair_batch_indices, batch_pairs)):
        for anchor in pair["anchors"]:
            anchor_pair.append(local_pair_index)
            anchor_batch.append(batch_index)
            event_s.append(anchor["event_s"])
            event_r.append(pair["event_index"])
            event_t.append(anchor["event_t"])
            control_s.append(anchor["control_s"])
            control_r.append(pair["control_index"])
            control_t.append(anchor["control_t"])
            bins.append(anchor["span_bin"])
    anchor_pair_np = np.asarray(anchor_pair, dtype=np.int64)
    bins_np = np.asarray(bins, dtype=np.int64)
    anchor_batch_tensor = torch.tensor(anchor_batch, device=device, dtype=torch.long)
    position_tensors = [
        torch.tensor(values, device=device, dtype=torch.long)
        for values in (event_s, event_r, event_t, control_s, control_r, control_t)
    ]
    pair_n = len(batch_pairs)
    for layer, state in enumerate(hidden_states):
        local_event = one_minus_cosine(
            state[pair_batch, event_indices] - state[pair_batch, event_indices - 1],
            state[pair_batch, event_indices + 1] - state[pair_batch, event_indices],
        ).cpu().numpy()
        local_control = one_minus_cosine(
            state[pair_batch, control_indices] - state[pair_batch, control_indices - 1],
            state[pair_batch, control_indices + 1] - state[pair_batch, control_indices],
        ).cpu().numpy()
        result_arrays["local_event"][global_pair_indices_np, layer] = local_event
        result_arrays["local_control"][global_pair_indices_np, layer] = local_control

        event_sum = np.zeros(pair_n, dtype=np.float64)
        control_sum = np.zeros(pair_n, dtype=np.float64)
        valid_count = np.zeros(pair_n, dtype=np.int32)
        event_bin_sum = np.zeros((pair_n, len(SPAN_BINS)), dtype=np.float64)
        control_bin_sum = np.zeros((pair_n, len(SPAN_BINS)), dtype=np.float64)
        bin_count = np.zeros((pair_n, len(SPAN_BINS)), dtype=np.int32)
        for chunk_start in range(0, len(anchor_pair), anchor_chunk_size):
            chunk_end = min(len(anchor_pair), chunk_start + anchor_chunk_size)
            batch_chunk = anchor_batch_tensor[chunk_start:chunk_end]
            es, er, et, cs, cr, ct = [
                state[batch_chunk, positions[chunk_start:chunk_end]]
                for positions in position_tensors
            ]
            event_values = one_minus_cosine(er - es, et - er).cpu().numpy()
            control_values = one_minus_cosine(cr - cs, ct - cr).cpu().numpy()
            pair_chunk = anchor_pair_np[chunk_start:chunk_end]
            bin_chunk = bins_np[chunk_start:chunk_end]
            valid = np.isfinite(event_values) & np.isfinite(control_values)
            if not valid.any():
                continue
            valid_pairs = pair_chunk[valid]
            valid_bins = bin_chunk[valid]
            event_sum += np.bincount(valid_pairs, weights=event_values[valid], minlength=pair_n)
            control_sum += np.bincount(valid_pairs, weights=control_values[valid], minlength=pair_n)
            valid_count += np.bincount(valid_pairs, minlength=pair_n)
            flat_bins = valid_pairs * len(SPAN_BINS) + valid_bins
            event_bin_sum += np.bincount(
                flat_bins, weights=event_values[valid], minlength=pair_n * len(SPAN_BINS),
            ).reshape(pair_n, len(SPAN_BINS))
            control_bin_sum += np.bincount(
                flat_bins, weights=control_values[valid], minlength=pair_n * len(SPAN_BINS),
            ).reshape(pair_n, len(SPAN_BINS))
            bin_count += np.bincount(
                flat_bins, minlength=pair_n * len(SPAN_BINS),
            ).reshape(pair_n, len(SPAN_BINS))
        pair_valid = valid_count > 0
        result_arrays["semi_valid_counts"][global_pair_indices_np, layer] = valid_count
        result_arrays["semi_event"][global_pair_indices_np[pair_valid], layer] = event_sum[pair_valid] / valid_count[pair_valid]
        result_arrays["semi_control"][global_pair_indices_np[pair_valid], layer] = control_sum[pair_valid] / valid_count[pair_valid]
        result_arrays["semi_bin_valid_counts"][global_pair_indices_np, layer, :] = bin_count
        for bin_index in range(len(SPAN_BINS)):
            valid_bin = bin_count[:, bin_index] > 0
            result_arrays["semi_event_bins"][global_pair_indices_np[valid_bin], layer, bin_index] = (
                event_bin_sum[valid_bin, bin_index] / bin_count[valid_bin, bin_index]
            )
            result_arrays["semi_control_bins"][global_pair_indices_np[valid_bin], layer, bin_index] = (
                control_bin_sum[valid_bin, bin_index] / bin_count[valid_bin, bin_index]
            )


def _sampled_parameter_fingerprint(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        flat = parameter.detach().view(-1)
        if flat.numel():
            indices = torch.linspace(0, flat.numel() - 1, min(32, flat.numel()), device=flat.device).long()
            digest.update(flat.index_select(0, indices).float().cpu().numpy().tobytes())
    return digest.hexdigest()


def _model_file_metadata(model_dir: Path) -> list[dict]:
    provenance_path = model_dir / "PROVENANCE.json"
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.exists() else {}
    )
    files = []
    for pattern in ("*.safetensors", "*.bin", "config.json"):
        for path in sorted(model_dir.glob(pattern)):
            if (
                path.name == "model.safetensors"
                and provenance.get("model_bytes") == path.stat().st_size
                and provenance.get("model_sha256")
            ):
                digest = provenance["model_sha256"]
                verification = "repository PROVENANCE.json (size checked)"
            else:
                hasher = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        hasher.update(chunk)
                digest = hasher.hexdigest()
                verification = "computed"
            files.append({
                "path": str(path.resolve()), "bytes": path.stat().st_size,
                "sha256": digest, "sha256_source": verification,
            })
    return files


def _cluster_test(values: np.ndarray, seed: int, bootstrap_samples: int, permutations: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return {"reaction_n": 0, "mean_delta": math.nan, "ci95": [math.nan, math.nan], "cohens_dz": math.nan, "p": math.nan}
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else math.nan
    effect = mean / sd if n > 1 and sd > 0 else math.nan
    rng = np.random.default_rng(seed)
    if bootstrap_samples > 0:
        selections = rng.integers(0, n, size=(bootstrap_samples, n))
        boot_means = values[selections].mean(axis=1)
        ci = np.quantile(boot_means, [0.025, 0.975]).tolist()
    else:
        ci = [math.nan, math.nan]
    if permutations <= 0:
        p_value = math.nan
    elif np.all(values == 0):
        p_value = 1.0
    else:
        extreme = 0
        completed = 0
        chunk = 2000
        absolute_observed = abs(mean)
        while completed < permutations:
            size = min(chunk, permutations - completed)
            signs = rng.integers(0, 2, size=(size, n), dtype=np.int8) * 2 - 1
            permuted = (signs * values).mean(axis=1)
            extreme += int((np.abs(permuted) >= absolute_observed - 1e-15).sum())
            completed += size
        p_value = (extreme + 1.0) / (permutations + 1.0)
    return {
        "reaction_n": n,
        "mean_delta": mean,
        "ci95": [float(ci[0]), float(ci[1])],
        "cohens_dz": effect,
        "p": float(p_value),
    }


def _distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "mean": math.nan, "sd": math.nan, "q25": math.nan, "median": math.nan, "q75": math.nan}
    q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    return {
        "n": int(len(values)), "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
        "q25": float(q25), "median": float(median), "q75": float(q75),
    }


def _benjamini_hochberg(records: list[dict]) -> None:
    valid = [(index, record["paired"]["p"]) for index, record in enumerate(records) if math.isfinite(record["paired"]["p"])]
    ordered = sorted(valid, key=lambda item: item[1])
    adjusted = [0.0] * len(ordered)
    running = 1.0
    for reverse_index in range(len(ordered) - 1, -1, -1):
        _, p_value = ordered[reverse_index]
        rank = reverse_index + 1
        running = min(running, p_value * len(ordered) / rank)
        adjusted[reverse_index] = running
    for (record_index, _), q_value in zip(ordered, adjusted):
        records[record_index]["paired"]["q_bh_global"] = float(q_value)


def summarize_geometry(
    pairs: list[dict], arrays: dict[str, np.ndarray], layer_count: int,
    seed: int, bootstrap_samples: int, permutations: int,
) -> dict:
    categories = np.asarray([pair["category"] for pair in pairs])
    reaction_ids = np.asarray([pair["reaction_identity"] for pair in pairs])
    tests = []
    metric_arrays = {
        "local_curvature": (arrays["local_event"], arrays["local_control"]),
        "semi_global_alignment": (arrays["semi_event"], arrays["semi_control"]),
    }
    for metric_index, (metric, (event_values, control_values)) in enumerate(metric_arrays.items()):
        for category_index, category in enumerate(EVENT_TYPES):
            mask = categories == category
            for layer in range(layer_count):
                event_layer = event_values[mask, layer]
                control_layer = control_values[mask, layer]
                deltas = event_layer - control_layer
                clustered = []
                for reaction_id in sorted(set(reaction_ids[mask])):
                    reaction_mask = reaction_ids[mask] == reaction_id
                    reaction_values = deltas[reaction_mask]
                    if np.isfinite(reaction_values).any():
                        clustered.append(float(np.nanmean(reaction_values)))
                test_seed = seed + metric_index * 100_000 + category_index * 1_000 + layer
                tests.append({
                    "metric": metric,
                    "category": category,
                    "layer": layer,
                    "event": _distribution(event_layer),
                    "control": _distribution(control_layer),
                    "paired": _cluster_test(
                        np.asarray(clustered), test_seed, bootstrap_samples, permutations,
                    ),
                })
    _benjamini_hochberg(tests)

    span_effects = []
    for category_index, category in enumerate(EVENT_TYPES):
        mask = categories == category
        for layer in range(layer_count):
            for bin_index, bin_name in enumerate(SPAN_BINS):
                event_layer = arrays["semi_event_bins"][mask, layer, bin_index]
                control_layer = arrays["semi_control_bins"][mask, layer, bin_index]
                deltas = event_layer - control_layer
                clustered = []
                for reaction_id in sorted(set(reaction_ids[mask])):
                    reaction_mask = reaction_ids[mask] == reaction_id
                    if np.isfinite(deltas[reaction_mask]).any():
                        clustered.append(float(np.nanmean(deltas[reaction_mask])))
                span_effects.append({
                    "category": category, "layer": layer, "span_bin": bin_name,
                    "event": _distribution(event_layer), "control": _distribution(control_layer),
                    "paired": _cluster_test(
                        np.asarray(clustered), seed + 500_000 + category_index * 10_000 + layer * 10 + bin_index,
                        min(1000, bootstrap_samples), 0,
                    ),
                })

    # Depth-added effects remove each reaction's layer-0 event/control contrast,
    # exposing contextual geometry beyond static token-embedding differences.
    depth_added = []
    for metric, (event_values, control_values) in metric_arrays.items():
        raw_delta = event_values - control_values
        for category in EVENT_TYPES:
            mask = categories == category
            for layer in range(layer_count):
                delta = raw_delta[mask, layer] - raw_delta[mask, 0]
                clustered = [
                    float(np.nanmean(delta[reaction_ids[mask] == reaction_id]))
                    for reaction_id in sorted(set(reaction_ids[mask]))
                    if np.isfinite(delta[reaction_ids[mask] == reaction_id]).any()
                ]
                depth_added.append({
                    "metric": metric, "category": category, "layer": layer,
                    "paired": _cluster_test(
                        np.asarray(clustered), seed + 900_000 + layer,
                        min(1000, bootstrap_samples), 0,
                    ),
                })
    return {"primary_tests": tests, "span_effects": span_effects, "depth_added_effects": depth_added}


_COLORS = {
    "ring_closure": "#3366cc", "branch": "#dc3912", "stereochemistry": "#109618",
    "motif": "#990099", "reaction_center": "#ff9900",
}


def _svg_line_plot(path: Path, panels: list[dict], title: str, width: int = 1100, panel_height: int = 350) -> None:
    height = 70 + panel_height * len(panels)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">{title}</text>',
    ]
    margin_left, margin_right = 90, 30
    plot_width = width - margin_left - margin_right
    for panel_index, panel in enumerate(panels):
        top = 60 + panel_index * panel_height
        plot_top, plot_bottom = top + 35, top + panel_height - 55
        all_values = [value for series in panel["series"] for value in series["values"] if math.isfinite(value)]
        all_values += [value for series in panel["series"] for bound in series.get("ci", []) for value in bound if math.isfinite(value)]
        y_min = min(all_values + [0.0])
        y_max = max(all_values + [0.0])
        padding = max(1e-4, (y_max - y_min) * 0.12)
        y_min -= padding
        y_max += padding
        x_count = max(len(series["values"]) for series in panel["series"])
        def x(value):
            return margin_left + (plot_width * value / max(1, x_count - 1))
        def y(value):
            return plot_bottom - (plot_bottom - plot_top) * (value - y_min) / (y_max - y_min)
        parts.extend([
            f'<text x="{margin_left}" y="{top+20}" font-family="sans-serif" font-size="16">{panel["title"]}</text>',
            f'<line x1="{margin_left}" y1="{plot_bottom}" x2="{width-margin_right}" y2="{plot_bottom}" stroke="#333"/>',
            f'<line x1="{margin_left}" y1="{plot_top}" x2="{margin_left}" y2="{plot_bottom}" stroke="#333"/>',
            f'<line x1="{margin_left}" y1="{y(0)}" x2="{width-margin_right}" y2="{y(0)}" stroke="#999" stroke-dasharray="4,4"/>',
            f'<text x="{margin_left-8}" y="{y(y_min)+4}" text-anchor="end" font-family="sans-serif" font-size="11">{y_min:.3f}</text>',
            f'<text x="{margin_left-8}" y="{y(y_max)+4}" text-anchor="end" font-family="sans-serif" font-size="11">{y_max:.3f}</text>',
            f'<text x="{width/2}" y="{plot_bottom+42}" text-anchor="middle" font-family="sans-serif" font-size="12">hidden layer (0 = token embedding)</text>',
        ])
        for series_index, series in enumerate(panel["series"]):
            color = series.get("color", "#3366cc")
            if series.get("ci"):
                lower = [bound[0] for bound in series["ci"]]
                upper = [bound[1] for bound in series["ci"]]
                polygon = [(x(i), y(value)) for i, value in enumerate(lower)] + [(x(i), y(value)) for i, value in reversed(list(enumerate(upper)))]
                points = " ".join(f"{px:.1f},{py:.1f}" for px, py in polygon if math.isfinite(px) and math.isfinite(py))
                parts.append(f'<polygon points="{points}" fill="{color}" opacity="0.13"/>')
            points = " ".join(f"{x(i):.1f},{y(value):.1f}" for i, value in enumerate(series["values"]) if math.isfinite(value))
            dash = ' stroke-dasharray="6,4"' if series.get("dashed") else ""
            parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"{dash}/>' )
            legend_x = margin_left + series_index * 185
            parts.append(f'<line x1="{legend_x}" y1="{plot_bottom+18}" x2="{legend_x+22}" y2="{plot_bottom+18}" stroke="{color}" stroke-width="3"{dash}/>' )
            parts.append(f'<text x="{legend_x+27}" y="{plot_bottom+22}" font-family="sans-serif" font-size="11">{series["label"]}</text>')
        for tick in range(0, x_count, max(1, (x_count - 1) // 5)):
            parts.append(f'<text x="{x(tick)}" y="{plot_bottom+16}" text-anchor="middle" font-family="sans-serif" font-size="10">{tick}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_plots(summary: dict, output_dir: Path, layer_count: int) -> None:
    tests = summary["primary_tests"]
    distribution_panels = []
    effect_panels = []
    for metric in ("local_curvature", "semi_global_alignment"):
        for category in EVENT_TYPES:
            rows = sorted(
                (row for row in tests if row["metric"] == metric and row["category"] == category),
                key=lambda row: row["layer"],
            )
            distribution_panels.append({
                "title": f"{metric.replace('_', ' ')} — {category.replace('_', ' ')}",
                "series": [
                    {
                        "label": "event median/IQR", "color": _COLORS[category],
                        "values": [row["event"]["median"] for row in rows],
                        "ci": [[row["event"]["q25"], row["event"]["q75"]] for row in rows],
                    },
                    {
                        "label": "control median/IQR", "color": "#666666", "dashed": True,
                        "values": [row["control"]["median"] for row in rows],
                        "ci": [[row["control"]["q25"], row["control"]["q75"]] for row in rows],
                    },
                ],
            })
        effect_panels.append({
            "title": f"Reaction-clustered paired effect: {metric.replace('_', ' ')}",
            "series": [
                {
                    "label": category.replace("_", " "), "color": _COLORS[category],
                    "values": [row["paired"]["mean_delta"] for row in sorted(
                        (item for item in tests if item["metric"] == metric and item["category"] == category),
                        key=lambda item: item["layer"],
                    )],
                    "ci": [row["paired"]["ci95"] for row in sorted(
                        (item for item in tests if item["metric"] == metric and item["category"] == category),
                        key=lambda item: item["layer"],
                    )],
                }
                for category in EVENT_TYPES
            ],
        })
    _svg_line_plot(output_dir / "layerwise_distributions.svg", distribution_panels, "Frozen ChemFM event/control layer-wise medians")
    _svg_line_plot(output_dir / "paired_effects.svg", effect_panels, "Event minus matched-control geometry (95% reaction bootstrap CI)")

    span_panels = []
    for category in EVENT_TYPES:
        span_panels.append({
            "title": f"Semi-global span persistence — {category.replace('_', ' ')}",
            "series": [
                {
                    "label": bin_name.replace("_", " "),
                    "color": ("#777777", "#3366cc", "#109618", "#dc3912")[bin_index],
                    "values": [next(
                        row["paired"]["mean_delta"] for row in summary["span_effects"]
                        if row["category"] == category and row["layer"] == layer and row["span_bin"] == bin_name
                    ) for layer in range(layer_count)],
                    "ci": [next(
                        row["paired"]["ci95"] for row in summary["span_effects"]
                        if row["category"] == category and row["layer"] == layer and row["span_bin"] == bin_name
                    ) for layer in range(layer_count)],
                }
                for bin_index, bin_name in enumerate(SPAN_BINS)
            ],
        })
    _svg_line_plot(output_dir / "span_persistence.svg", span_panels, "STP-style disruption by anchor span")


def _write_pair_metadata(path: Path, pairs: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for pair in pairs:
            compact = {key: value for key, value in pair.items() if key != "anchors"}
            compact["anchor_count"] = len(pair["anchors"])
            handle.write(json.dumps(compact, sort_keys=True) + "\n")


def _write_tests_csv(path: Path, tests: list[dict]) -> None:
    fields = [
        "metric", "category", "layer", "event_n", "event_mean", "event_median", "event_q25", "event_q75",
        "control_mean", "control_median", "control_q25", "control_q75", "reaction_n", "mean_delta",
        "ci95_low", "ci95_high", "cohens_dz", "p_sign_flip", "q_bh_global",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in tests:
            writer.writerow({
                "metric": row["metric"], "category": row["category"], "layer": row["layer"],
                "event_n": row["event"]["n"], "event_mean": row["event"]["mean"],
                "event_median": row["event"]["median"], "event_q25": row["event"]["q25"], "event_q75": row["event"]["q75"],
                "control_mean": row["control"]["mean"], "control_median": row["control"]["median"],
                "control_q25": row["control"]["q25"], "control_q75": row["control"]["q75"],
                "reaction_n": row["paired"]["reaction_n"], "mean_delta": row["paired"]["mean_delta"],
                "ci95_low": row["paired"]["ci95"][0], "ci95_high": row["paired"]["ci95"][1],
                "cohens_dz": row["paired"]["cohens_dz"], "p_sign_flip": row["paired"]["p"],
                "q_bh_global": row["paired"].get("q_bh_global", math.nan),
            })


def _write_artifact_manifest(output_dir: Path, names: Sequence[str]) -> None:
    records = []
    for name in names:
        path = output_dir / name
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        records.append({"name": name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps({"artifacts": records}, indent=2) + "\n", encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict:
    start_time = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    panel_path = args.panel.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_panel(panel_path, limit=args.limit)
    stereo_path = args.stereo_supplement.resolve()
    stereo_rows = read_stereo_supplement(stereo_path, args.stereo_reactions, args.seed)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)

    annotation_start = time.perf_counter()
    examples = []
    match_diagnostics = []
    for panel_index, row in enumerate(rows):
        example = annotate_example(tokenizer, row, panel_index, "prespecified_256")
        match_diagnostics.append(match_controls_and_anchors(
            example, args.seed, args.anchors_per_event,
            categories=("ring_closure", "branch", "motif", "reaction_center"),
        ))
        examples.append(example)
    for supplement_index, row in enumerate(stereo_rows, start=len(rows)):
        example = annotate_example(
            tokenizer, row, supplement_index, "uspto50k_stereo_supplement",
            infer_reaction_center=False,
        )
        match_diagnostics.append(match_controls_and_anchors(
            example, args.seed, args.anchors_per_event,
            categories=("stereochemistry",),
        ))
        examples.append(example)
    annotation_seconds = time.perf_counter() - annotation_start
    pairs = [pair for example in examples for pair in example.pairs]
    if not pairs:
        raise RuntimeError("event annotation produced no matched pairs")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    model = load_lora_model(
        MODEL_DIR, tokenizer, chemfm_vocab_size=len(tokenizer),
        attention_dropout=0.0, attn_implementation=args.attn_implementation,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("frozen diagnostic found a trainable parameter")
    before_fingerprint = _sampled_parameter_fingerprint(model)
    layer_count = int(model.config.num_hidden_layers) + 1
    pair_count = len(pairs)
    result_arrays = {
        "local_event": np.full((pair_count, layer_count), np.nan, dtype=np.float32),
        "local_control": np.full((pair_count, layer_count), np.nan, dtype=np.float32),
        "semi_event": np.full((pair_count, layer_count), np.nan, dtype=np.float32),
        "semi_control": np.full((pair_count, layer_count), np.nan, dtype=np.float32),
        "semi_valid_counts": np.zeros((pair_count, layer_count), dtype=np.int16),
        "semi_event_bins": np.full((pair_count, layer_count, len(SPAN_BINS)), np.nan, dtype=np.float32),
        "semi_control_bins": np.full((pair_count, layer_count, len(SPAN_BINS)), np.nan, dtype=np.float32),
        "semi_bin_valid_counts": np.zeros((pair_count, layer_count, len(SPAN_BINS)), dtype=np.int16),
    }

    pair_offsets = {}
    offset = 0
    for example in examples:
        pair_offsets[example.panel_index] = offset
        offset += len(example.pairs)
    ordered_examples = sorted(examples, key=lambda example: len(example.input_ids))
    inference_start = time.perf_counter()
    processed = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        if torch.is_grad_enabled():
            raise RuntimeError("torch gradients unexpectedly enabled inside inference block")
        for batch_start in range(0, len(ordered_examples), args.batch_size):
            batch_examples = ordered_examples[batch_start:batch_start + args.batch_size]
            maximum = max(len(example.input_ids) for example in batch_examples)
            input_ids = torch.full(
                (len(batch_examples), maximum), tokenizer.pad_token_id,
                dtype=torch.long, device=device,
            )
            attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for batch_index, example in enumerate(batch_examples):
                length = len(example.input_ids)
                input_ids[batch_index, :length] = torch.tensor(example.input_ids, device=device)
                attention_mask[batch_index, :length] = True
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            if len(outputs.hidden_states) != layer_count:
                raise RuntimeError(f"expected {layer_count} hidden states, got {len(outputs.hidden_states)}")
            _compute_batch_geometry(
                outputs.hidden_states, batch_examples, result_arrays, pair_offsets,
            )
            processed += len(batch_examples)
            if processed == len(ordered_examples) or processed % args.progress_every < len(batch_examples):
                elapsed = time.perf_counter() - inference_start
                rate = processed / max(elapsed, 1e-9)
                eta = (len(ordered_examples) - processed) / max(rate, 1e-9)
                print(json.dumps({
                    "stage": "inference", "processed": processed, "total": len(ordered_examples),
                    "elapsed_seconds": round(elapsed, 1), "eta_seconds": round(eta, 1),
                    "reactions_per_second": round(rate, 3),
                }), flush=True)
            del outputs, input_ids, attention_mask
    inference_seconds = time.perf_counter() - inference_start
    after_fingerprint = _sampled_parameter_fingerprint(model)
    if before_fingerprint != after_fingerprint:
        raise RuntimeError("model parameter fingerprint changed during frozen inference")
    peak_cuda_bytes = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0

    summary_geometry = summarize_geometry(
        pairs, result_arrays, layer_count, args.seed,
        args.bootstrap_samples, args.permutations,
    )
    event_counts = defaultdict(int)
    matched_counts = defaultdict(int)
    class_matches = defaultdict(int)
    for diagnostic in match_diagnostics:
        for category, values in diagnostic.items():
            event_counts[category] += values["events"]
            matched_counts[category] += values["matched"]
            class_matches[category] += values["class_matches"]
    center_status = defaultdict(int)
    center_overlap = []
    for example in examples:
        if example.sample_origin != "prespecified_256":
            continue
        metadata = example.reaction_center_metadata
        center_status[metadata.get("status", "unknown")] += 1
        if "overlap_atoms" in metadata:
            center_overlap.append(metadata["overlap_atoms"] / max(1, metadata["target_component_atoms"]))
    local_joint_valid = (
        np.isfinite(result_arrays["local_event"])
        & np.isfinite(result_arrays["local_control"])
    ).sum(axis=0)
    semi_valid_anchor_counts = result_arrays["semi_valid_counts"].sum(axis=0)
    control_reuse = defaultdict(int)
    for pair in pairs:
        control_reuse[(
            pair["reaction_identity"], pair["category"], pair["segment"], pair["control_index"],
        )] += 1
    metadata = {
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit_at_start": os.popen(f'git -C "{ROOT}" rev-parse HEAD').read().strip(),
        "panel": str(panel_path),
        "panel_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
        "reaction_count": len(examples),
        "main_reaction_count": len(rows),
        "stereochemistry_supplement": {
            "path": str(stereo_path),
            "sha256": hashlib.sha256(stereo_path.read_bytes()).hexdigest() if stereo_rows else None,
            "reaction_count": len(stereo_rows),
            "selection": "seeded sample without replacement from canonical view zero of each 20-view USPTO-50K test reaction containing @, /, or backslash",
            "scope": "stereochemistry event stratum only",
        },
        "serialization": f"{REACTANT_START}{{canonical_source}}{END}{PRODUCT_START}{{canonical_target}}{END}",
        "model_dir": str(MODEL_DIR.resolve()),
        "model_files": _model_file_metadata(MODEL_DIR),
        "model_class": type(model).__name__,
        "model_parameter_fingerprint_before": before_fingerprint,
        "model_parameter_fingerprint_after": after_fingerprint,
        "all_parameters_frozen": True,
        "optimizer_constructed": False,
        "loss_constructed": False,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "rdkit": rdBase.rdkitVersion, "numpy": np.__version__, "scipy": scipy.__version__,
        "python": platform.python_version(),
        "seed": args.seed, "anchors_per_event": args.anchors_per_event,
        "batch_size": args.batch_size, "attention_implementation": args.attn_implementation,
        "layers": layer_count, "hidden_size": int(model.config.hidden_size),
        "pair_count": pair_count,
        "annotation_seconds": annotation_seconds, "inference_seconds": inference_seconds,
        "total_seconds": time.perf_counter() - start_time,
        "peak_cuda_bytes": peak_cuda_bytes,
        "event_counts": dict(event_counts), "matched_counts": dict(matched_counts),
        "token_class_match_counts": dict(class_matches),
        "control_matching": {
            "all_pairs_same_token_class": all(pair["event_class"] == pair["control_class"] for pair in pairs),
            "maximum_control_reuse_within_reaction_category_segment": max(control_reuse.values()),
            "median_control_reuse_within_reaction_category_segment": float(np.median(list(control_reuse.values()))),
        },
        "validity_by_layer": {
            "local_joint_valid_pairs": local_joint_valid.tolist(),
            "local_total_pairs": pair_count,
            "semi_joint_valid_anchors": semi_valid_anchor_counts.tolist(),
            "semi_total_requested_anchors": pair_count * args.anchors_per_event,
        },
        "reaction_center_inference": {
            "statuses": dict(center_status),
            "median_mcs_target_atom_fraction": float(np.median(center_overlap)) if center_overlap else math.nan,
        },
        "inference_unit": "reaction-clustered mean event-minus-matched-control difference",
        "uncertainty": f"{args.bootstrap_samples} reaction-cluster bootstrap resamples",
        "significance": f"two-sided {args.permutations}-draw reaction-level sign-flip; BH over all primary metric/category/layer tests",
        "span_bins": list(SPAN_BINS),
        "stp_reference": {
            "paper": "https://arxiv.org/abs/2602.22617",
            "implementation": "https://github.com/galilai-group/llm-jepa",
            "commit": STP_REFERENCE_COMMIT,
        },
        "motif_smarts": MOTIF_SMARTS,
    }
    result = {"metadata": metadata, **summary_geometry}
    np.savez_compressed(
        output_dir / "pair_geometry.npz",
        **result_arrays,
        anchor_left=np.asarray([[anchor["left"] for anchor in pair["anchors"]] for pair in pairs], dtype=np.uint16),
        anchor_right=np.asarray([[anchor["right"] for anchor in pair["anchors"]] for pair in pairs], dtype=np.uint16),
        anchor_span_bin=np.asarray([[anchor["span_bin"] for anchor in pair["anchors"]] for pair in pairs], dtype=np.uint8),
        category=np.asarray([pair["category"] for pair in pairs]),
        reaction_identity=np.asarray([pair["reaction_identity"] for pair in pairs]),
        sample_origin=np.asarray([pair["sample_origin"] for pair in pairs]),
        segment=np.asarray([pair["segment"] for pair in pairs]),
        event_index=np.asarray([pair["event_index"] for pair in pairs]),
        control_index=np.asarray([pair["control_index"] for pair in pairs]),
    )
    _write_pair_metadata(output_dir / "matched_pairs.jsonl", pairs)
    _write_tests_csv(output_dir / "layerwise_primary_tests.csv", result["primary_tests"])
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    write_plots(result, output_dir, layer_count)
    _write_artifact_manifest(output_dir, (
        "summary.json", "layerwise_primary_tests.csv", "pair_geometry.npz",
        "matched_pairs.jsonl", "layerwise_distributions.svg",
        "paired_effects.svg", "span_persistence.svg",
    ))
    print(json.dumps({
        "stage": "complete", "output": str(output_dir), "reactions": len(examples),
        "pairs": pair_count, "layers": layer_count,
        "inference_seconds": round(inference_seconds, 1),
        "total_seconds": round(metadata["total_seconds"], 1),
    }, sort_keys=True), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stereo-supplement", type=Path, default=DEFAULT_STEREO_SUPPLEMENT)
    parser.add_argument("--stereo-reactions", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--anchors-per-event", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--progress-every", type=int, default=32)
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="sdpa")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
