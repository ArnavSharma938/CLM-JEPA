from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from eval_uspto_mit_five_view_a6000 import (
    _generation_cost_proxy,
    assign_groups,
    clopper_pearson_interval,
    exact_mcnemar_power,
    holm_adjust,
    official_rank,
    paired_binary_endpoint,
    paired_bootstrap_interval,
    read_adapter_architecture,
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


def test_paired_binary_endpoint_and_bootstrap_preserve_pairing():
    left = np.asarray([True, True, False, False])
    right = np.asarray([True, False, True, False])
    result = paired_binary_endpoint(left, right)
    assert result["both_correct"] == 1
    assert result["native_only_correct"] == 1
    assert result["clm_jepa_only_correct"] == 1
    assert result["absolute_difference"] == 0.0
    assert paired_bootstrap_interval(
        right.astype(np.int8) - left.astype(np.int8), seed=11, repetitions=2000,
    ) == pytest.approx([-0.75, 0.75])


def test_adapter_architecture_is_reconstructed_and_legacy_rank8_survives(tmp_path):
    legacy = read_adapter_architecture(tmp_path / "legacy")
    assert (legacy["rank"], legacy["alpha"]) == (8, 8)

    nested = tmp_path / "rank128" / "USPTO-MIT-Synthesis"
    nested.mkdir(parents=True)
    (nested / "adapter_config.json").write_text(
        __import__("json").dumps({
            "r": 128,
            "lora_alpha": 128,
            "lora_dropout": 0.1,
            "modules_to_save": ["embed_tokens", "lm_head"],
            "use_rslora": False,
        }),
        encoding="utf-8",
    )
    observed = read_adapter_architecture(tmp_path / "rank128")
    assert (observed["rank"], observed["alpha"]) == (128, 128)
    assert observed["weights_path"] == nested


def test_length_balanced_assignment_is_complete_deterministic_and_balanced():
    groups = []
    for index, target_length in enumerate((120, 100, 80, 60, 40, 20, 10, 5)):
        groups.append({
            "panel_index": index,
            "reaction_identity": f"r{index}",
            "sources": ["C" * 20] * 5,
            "targets": ["C" * target_length] * 5,
            "source_character_lengths": [20] * 5,
        })
    first = assign_groups(groups, 4, "length-balanced")
    second = assign_groups(groups, 4, "length-balanced")
    assert first == second
    assert sorted(item["panel_index"] for worker in first for item in worker) == list(range(8))
    balanced_loads = [sum(_generation_cost_proxy(item) for item in worker) for worker in first]
    round_robin = assign_groups(groups, 4, "round-robin")
    round_robin_loads = [sum(_generation_cost_proxy(item) for item in worker) for worker in round_robin]
    assert max(balanced_loads) - min(balanced_loads) < max(round_robin_loads) - min(round_robin_loads)
