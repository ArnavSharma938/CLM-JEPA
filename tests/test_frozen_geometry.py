import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frozen_geometry import (
    EVENT_TYPES,
    TOKENIZER_DIR,
    _atom_spans,
    _benjamini_hochberg,
    _compute_example_geometry,
    _compute_batch_geometry,
    _span_bin,
    annotate_example,
    load_reaction_tokenizer,
    match_controls_and_anchors,
    one_minus_cosine,
)


def test_smiles_lexer_matches_rdkit_atom_order_for_bracket_and_two_letter_atoms():
    smiles = "Cl[C@H]1C(Br)=CC[NH+]1[O-]"
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    spans = _atom_spans(smiles)
    assert len(spans) == molecule.GetNumAtoms()
    assert [smiles[start:end] for start, end in spans] == [
        "Cl", "[C@H]", "C", "Br", "C", "C", "[NH+]", "[O-]",
    ]


def test_annotation_covers_requested_events_and_matches_same_class_controls():
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    row = {
        "reaction_identity": "synthetic-test",
        "canonical_source": "F[C@H]1CC(C(=O)O)C1",
        "canonical_target": "F[C@@H]1CC(C(=O)N)C1",
    }
    example = annotate_example(tokenizer, row, panel_index=0)
    diagnostics = match_controls_and_anchors(example, seed=17, anchors_per_event=16)
    present = {event for token in example.tokens for event in token.events}
    assert set(EVENT_TYPES) <= present
    assert all(diagnostics[event]["matched"] > 0 for event in EVENT_TYPES)
    assert all(pair["event_class"] == pair["control_class"] for pair in example.pairs)
    for pair in example.pairs:
        assert len(pair["anchors"]) == 16
        assert all(anchor["event_s"] < pair["event_index"] < anchor["event_t"] for anchor in pair["anchors"])
        assert all(anchor["control_s"] < pair["control_index"] < anchor["control_t"] for anchor in pair["anchors"])
        assert all(
            pair["event_index"] - anchor["event_s"]
            == pair["control_index"] - anchor["control_s"]
            for anchor in pair["anchors"]
        )
        assert all(
            anchor["event_t"] - pair["event_index"]
            == anchor["control_t"] - pair["control_index"]
            for anchor in pair["anchors"]
        )


def test_local_and_semi_global_geometry_use_requested_ordered_differences():
    states = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]])
    pairs = [{
        "event_index": 3,
        "control_index": 1,
        "anchors": [{
            "event_s": 2, "event_t": 4,
            "control_s": 0, "control_t": 2,
            "span_bin": 0,
        }],
    }]
    arrays = {
        "local_event": np.full((1, 1), np.nan, dtype=np.float32),
        "local_control": np.full((1, 1), np.nan, dtype=np.float32),
        "semi_event": np.full((1, 1), np.nan, dtype=np.float32),
        "semi_control": np.full((1, 1), np.nan, dtype=np.float32),
        "semi_valid_counts": np.zeros((1, 1), dtype=np.int16),
        "semi_event_bins": np.full((1, 1, 4), np.nan, dtype=np.float32),
        "semi_control_bins": np.full((1, 1, 4), np.nan, dtype=np.float32),
        "semi_bin_valid_counts": np.zeros((1, 1, 4), dtype=np.int16),
    }
    _compute_example_geometry((states,), 0, pairs, arrays, 0)
    assert arrays["local_event"][0, 0] == pytest.approx(1.0)
    assert arrays["local_control"][0, 0] == pytest.approx(0.0)
    assert arrays["semi_event"][0, 0] == pytest.approx(1.0)
    assert arrays["semi_control"][0, 0] == pytest.approx(0.0)
    assert arrays["semi_event_bins"][0, 0, 0] == pytest.approx(1.0)

    vectorized = {
        key: (np.zeros_like(value) if "valid_counts" in key else np.full_like(value, np.nan))
        for key, value in arrays.items()
    }
    _compute_batch_geometry(
        (states,), [SimpleNamespace(panel_index=0, pairs=pairs)], vectorized, {0: 0},
        anchor_chunk_size=1,
    )
    for key in arrays:
        np.testing.assert_allclose(vectorized[key], arrays[key], equal_nan=True)


def test_one_minus_cosine_and_span_bins_cover_edge_cases():
    first = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    second = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]])
    values = one_minus_cosine(first, second)
    torch.testing.assert_close(values[:2], torch.tensor([0.0, 2.0]))
    assert torch.isnan(values[2])
    assert [_span_bin(value) for value in (2, 3, 8, 9, 24, 25)] == [0, 1, 1, 2, 2, 3]


def test_global_bh_adjustment_is_monotone_and_attached_to_primary_records():
    records = [{"paired": {"p": value}} for value in (0.01, 0.04, 0.03, 0.2)]
    _benjamini_hochberg(records)
    assert [record["paired"]["q_bh_global"] for record in records] == pytest.approx(
        [0.04, 0.05333333333333334, 0.05333333333333334, 0.2]
    )
