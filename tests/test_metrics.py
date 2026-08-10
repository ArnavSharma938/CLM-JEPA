from metrics import (
    canonical_set, load_records, prediction_records, save_records,
    rank_augmented_candidates, score_candidates, score_prediction_records,
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


def test_saved_predictions_reproduce_metrics_offline(tmp_path):
    rows = prediction_records(["CCO", "not-smiles", "O=C=O"], ["OCC", "CC", "O=C=O"])
    online = score_prediction_records(rows)
    path = tmp_path / "predictions.jsonl"
    save_records(path, rows)
    assert score_prediction_records(load_records(path)) == online == {
        "count": 3, "valid_products": 2, "exact_products": 2,
    }


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
