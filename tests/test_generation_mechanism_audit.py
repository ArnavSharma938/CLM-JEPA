import torch

from audit_generation_mechanism import (
    hard_negative_candidates,
    linear_cka,
    retrieval_metrics,
    ridge_explained_variance,
    target_prediction_mask,
)


def test_linear_cka_is_one_for_identical_and_orthogonal_feature_rotation():
    values = torch.tensor([
        [1.0, 2.0], [2.0, -1.0], [-2.0, 0.5], [0.0, -2.0],
    ])
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    assert abs(linear_cka(values, values) - 1.0) < 1e-12
    assert abs(linear_cka(values, values @ rotation) - 1.0) < 1e-12


def test_ridge_and_retrieval_recover_a_heldout_linear_relationship():
    generator = torch.Generator().manual_seed(4)
    sources = torch.randn(40, 8, generator=generator)
    targets = sources @ torch.randn(8, 8, generator=generator)
    identities = [f"reaction-{index:03d}" for index in range(40)]
    ridge = ridge_explained_variance(sources, targets, identities, alpha=1e-5)
    assert ridge["explained_variance"] > 0.999
    exact = retrieval_metrics(sources, sources.clone())
    assert exact["top1"] == 1.0
    assert exact["mrr"] == 1.0


def test_target_prediction_mask_matches_causal_shift():
    labels = torch.tensor([[-100, -100, 5, 6, 7]])
    mask = target_prediction_mask(labels)
    assert mask.tolist() == [[False, True, True, True, False]]


class _CharacterTokenizer:
    def __call__(self, values, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [list(value) for value in values]}


def test_hard_negative_candidates_are_nonself_four_way_rows():
    smiles = ["CCO", "CCN", "CCC", "CCCl", "CCBr"]
    rows = [
        {"target": value, "reaction_identity": f"reaction-{index}"}
        for index, value in enumerate(smiles)
    ]
    candidates, diagnostics = hard_negative_candidates(
        rows, _CharacterTokenizer(), negatives=3
    )
    assert len(candidates) == len(rows)
    assert diagnostics["pairs"] == 15
    for index, values in enumerate(candidates):
        assert values[0] == index
        assert index not in values[1:]
        assert len(set(values)) == 4
