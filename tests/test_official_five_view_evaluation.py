from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from official_five_view_evaluation import (
    clopper_pearson_interval,
    exact_mcnemar_power,
    holm_adjust,
    official_rank,
    required_sample_size,
)


ROOT = Path(__file__).resolve().parents[1]


def load_official_score_module():
    path = ROOT / "references" / "chemfm" / "finetuning" / "reaction_prediction" / "score.py"
    specification = importlib.util.spec_from_file_location("official_chemfm_score", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.opt = SimpleNamespace(beam_size=10)
    return module


def test_reciprocal_rank_matches_pinned_official_chemfm_exactly():
    views = [
        ["CC", "CCC", "CC", "", "CO", "CN", "CCl", "CBr", "CF", "CI"],
        ["CCC", "CC", "CO", "", "CN", "CCl", "CBr", "CF", "CI", "CS"],
        ["CO", "CC", "CCC", "CN", "", "CCl", "CBr", "CF", "CI", "CS"],
        ["CN", "CC", "CCC", "CO", "CCl", "", "CBr", "CF", "CI", "CS"],
        ["CC", "CO", "CCC", "CN", "CCl", "CBr", "", "CF", "CI", "CS"],
    ]
    official = load_official_score_module()
    official_scores, official_invalid = official.compute_rank(copy.deepcopy(views), alpha=1.0)
    expected = [
        value for value, _ in sorted(official_scores.items(), key=lambda item: item[1], reverse=True)[:10]
    ]
    expected += [""] * (10 - len(expected))
    actual, details = official_rank(copy.deepcopy(views))
    assert actual == expected
    assert details["invalid_by_beam"] == official_invalid


def test_reciprocal_rank_stably_breaks_score_ties_by_first_encounter():
    views = [["CC", "CCC"] + [""] * 8] + [[""] * 10 for _ in range(4)]
    ranked, _ = official_rank(views)
    assert ranked[:2] == ["CC", "CCC"]


def test_prespecified_exact_mcnemar_power_calculation():
    interval = clopper_pearson_interval(4, 256)
    assert interval == pytest.approx([0.004273272485754635, 0.039520756533374946])
    required, achieved = required_sample_size(interval[1], 0.01, 0.80)
    assert required == 3253
    assert achieved >= 0.80
    assert exact_mcnemar_power(required - 1, interval[1], 0.01) < 0.80


def test_holm_adjustment_is_monotone_in_sorted_p_values():
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.20})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.20})
