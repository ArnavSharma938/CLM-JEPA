from metrics import (
    canonical_set, rank_augmented_candidates, score_candidates,
)


def test_retro_scores_unordered_precursor_sets():
    rows = [{"src": "CCO", "tgt": "Br.CC"}]
    metrics, _ = score_candidates(rows, [["CC.Br", "invalid"]], "retro")
    assert canonical_set("Br.CC") == canonical_set("CC.Br")
    assert metrics["exact_top1"] == 1.0
    assert metrics["valid_rate"] == 0.5


def test_forward_reports_stereochemistry_error_separately():
    rows = [{"src": "CC", "tgt": "F[C@H](Cl)Br"}]
    metrics, _ = score_candidates(rows, [["F[C@@H](Cl)Br"]], "forward")
    assert metrics["exact_top1"] == 0.0
    assert metrics["connectivity_correct_stereo_wrong"] == 1.0


def test_r_smiles_views_use_official_reciprocal_rank_aggregation():
    ranked = rank_augmented_candidates(
        [
            ["CCO", "CC", "not-smiles"],
            ["OCC", "CCC", "CC"],
        ],
        task="forward",
        n_best=4,
    )
    assert ranked == ["CCO", "CC", "CCC", ""]
