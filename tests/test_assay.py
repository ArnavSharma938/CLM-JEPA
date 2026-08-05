import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clm_jepa.assay import identity_mappings, relationship_metrics


def test_identity_mappings_are_deterministic_and_deranged():
    identities = ["A", "A", "B", "B", "C", "C"]
    lengths = {"A": 5, "B": 7, "C": 10}
    atoms = {"A": 3, "B": 4, "C": 8}
    first = identity_mappings(identities, lengths, atoms, 533)
    second = identity_mappings(identities, lengths, atoms, 533)
    assert first == second
    assert all(key != value for key, value in first[0].items())
    assert all(key != value for key, value in first[1].items())


def test_relationship_metrics_detect_pair_signal():
    sources = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [-1.0, -1.0], [-1.0, -1.0]])
    targets = sources.clone()
    identities = ["A", "A", "B", "B", "C", "C"]
    values = relationship_metrics(
        sources, targets, identities,
        {"A": 5, "B": 7, "C": 10},
        {"A": 3, "B": 4, "C": 8},
    )
    assert values["correct_minus_matched"] > 0
    assert values["retrieval_top1"] == 1.0
    assert values["retains_pair_signal"]
