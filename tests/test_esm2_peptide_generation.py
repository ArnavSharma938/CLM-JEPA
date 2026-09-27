from __future__ import annotations

import torch
import torch.nn.functional as F
import copy
import json

from esm2_peptide_generation.attention import patch_esm_sdpa
from esm2_peptide_generation.analyze_conditioning import _cluster_means, analyze, planning_proxy_detectability
from esm2_peptide_generation.batching import GenerationCollator, length_bucket_batches
from esm2_peptide_generation.cluster_receptors import UnionFind
from esm2_peptide_generation.masking import (
    make_full_span_example,
    per_example_loss_sums,
    selected_cross_entropy_sum,
)
from esm2_peptide_generation.split_data import assign_clusters


def test_full_peptide_is_masked_together_and_only_peptide_has_labels() -> None:
    example = make_full_span_example(
        [4, 5, 6], [7, 8, 9, 10], bos_token_id=1, eos_token_id=2, mask_token_id=3
    )
    assert example.input_ids == (1, 4, 5, 6, 3, 3, 3, 3, 2)
    assert example.target_positions == (4, 5, 6, 7)
    assert example.labels == (-100, -100, -100, -100, 7, 8, 9, 10, -100)
    assert example.supervised_tokens == 4


def test_gathered_peptide_loss_matches_full_position_reference_exactly() -> None:
    torch.manual_seed(24)
    labels = torch.tensor([
        [-100, -100, 2, 5, -100],
        [-100, 1, 4, 3, -100],
    ])
    logits = torch.randn(2, 5, 9, dtype=torch.float32)
    reference = F.cross_entropy(
        logits.transpose(1, 2), labels, reduction="sum", ignore_index=-100
    )
    gathered = selected_cross_entropy_sum(logits, labels)
    torch.testing.assert_close(gathered, reference, rtol=0, atol=0)

    per_example, counts = per_example_loss_sums(logits, labels)
    expected = torch.stack([
        F.cross_entropy(logits[i][labels[i].ne(-100)], labels[i][labels[i].ne(-100)], reduction="sum")
        for i in range(labels.shape[0])
    ])
    torch.testing.assert_close(per_example, expected, rtol=0, atol=0)
    assert counts.tolist() == [2, 3]


def test_generation_collator_carries_supervised_counts_as_cpu_metadata() -> None:
    collator = GenerationCollator({"bos": 1, "eos": 2, "mask": 3, "pad": 0})
    batch = collator([
        {"row": {"pair_id": "a"}, "receptor_tokens": [4, 5], "peptide_tokens": [6, 7, 8]},
        {"row": {"pair_id": "b"}, "receptor_tokens": [9], "peptide_tokens": [10, 11]},
    ])
    assert batch["supervised_counts"] == (3, 2)
    assert batch["supervised_tokens"] == 5
    assert batch["input_ids"].tolist() == [[1, 4, 5, 3, 3, 3, 2], [1, 9, 3, 3, 2, 0, 0]]
    assert batch["labels"].tolist() == [
        [-100, -100, -100, 6, 7, 8, -100],
        [-100, -100, 10, 11, -100, -100, -100],
    ]
    assert batch["padded_input_tokens"] == 14
    assert batch["real_input_tokens"] == 12


def test_length_bucket_batches_obey_padded_token_budget() -> None:
    lengths = [80 + index % 41 for index in range(250)]
    batches = length_bucket_batches(lengths, 1024, seed=17, epoch=2)
    assert sorted(index for batch in batches for index in batch) == list(range(len(lengths)))
    assert all(max(lengths[i] for i in batch) * len(batch) <= 1024 for batch in batches)


def test_union_find_returns_transitive_receptor_families() -> None:
    groups = UnionFind(["a", "b", "c", "d", "e"])
    groups.union("a", "b")
    groups.union("b", "c")
    groups.union("d", "e")
    assert groups.groups() == [["a", "b", "c"], ["d", "e"]]


def test_split_assignment_keeps_each_cluster_whole() -> None:
    pairs = [
        {"receptor_id": f"r{i}", "pair_id": f"p{i}"}
        for i in range(12)
    ]
    clusters = [
        {"receptor_id": f"r{i}", "receptor_cluster_id": f"c{i // 3}"}
        for i in range(12)
    ]
    assigned = assign_clusters(pairs, clusters, seed=4)
    assert len(assigned) == len(pairs)
    for cluster_id in {item["receptor_cluster_id"] for item in clusters}:
        members = [item["receptor_id"] for item in clusters if item["receptor_cluster_id"] == cluster_id]
        assert len({assigned[member] for member in members}) == 1


def test_receptor_clusters_receive_equal_weight_regardless_of_pair_count() -> None:
    means = _cluster_means([
        {"cluster_id": "large", "gain": 0.0},
        {"cluster_id": "large", "gain": 0.0},
        {"cluster_id": "large", "gain": 0.0},
        {"cluster_id": "small", "gain": 1.0},
    ], "gain")
    assert means == {"large": 0.0, "small": 1.0}
    assert sum(means.values()) / len(means) == 0.5


def test_stage6_material_threshold_uses_primary_stage4_equal_cluster_gain(tmp_path) -> None:
    base_dir, dtft_dir = tmp_path / "base", tmp_path / "dtft"
    base_dir.mkdir()
    dtft_dir.mkdir()
    base_true, dtft_true, base_decoy, dtft_decoy = [], [], [], []
    for pair_id, cluster_id, dtft_nll in (("p1", "c1", 0.5), ("p2", "c2", 0.8)):
        base_true.append({"pair_id": pair_id, "receptor_cluster_id": cluster_id, "supervised_tokens": 1, "loss_sum": 1.0})
        dtft_true.append({"pair_id": pair_id, "receptor_cluster_id": cluster_id, "supervised_tokens": 1, "loss_sum": dtft_nll})
        for index in range(4):
            base_decoy.append({"pair_id": pair_id, "supervised_tokens": 1, "loss_sum": 1.1, "receptor_cluster_id": f"d{index}"})
            dtft_decoy.append({"pair_id": pair_id, "supervised_tokens": 1, "loss_sum": dtft_nll + 0.2, "receptor_cluster_id": f"d{index}"})
    for directory, name, rows in (
        (base_dir, "test_true.jsonl", base_true),
        (dtft_dir, "test_true.jsonl", dtft_true),
        (base_dir, "test_decoy.jsonl", base_decoy),
        (dtft_dir, "test_decoy.jsonl", dtft_decoy),
    ):
        (directory / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report = analyze(base_dir, dtft_dir, bootstrap_replicates=100, primary_a_dtft=0.8)
    assert abs(report["A_DTFT"] - 0.35) < 1e-12
    assert report["primary_A_DTFT_for_material_threshold"] == 0.8
    proxy = report["stage6_planning_proxy"]
    assert abs(proxy["delta_material_0.10_A_DTFT"] - 0.08) < 1e-12
    assert proxy["actual_dtft_vs_lora_paired_variance"] is None
    assert proxy["informational_only_not_a_gate0_stopping_criterion"] is True


def test_stage6_proxy_never_claims_actual_lora_precision() -> None:
    result = planning_proxy_detectability(
        {"c1": 0.1, "c2": 0.2, "c3": 0.3}, material_effect=0.02
    )
    assert result["proxy_mde_alpha_0.05_power_0.80"] > 0
    assert result["actual_dtft_vs_lora_paired_variance"] is None
    assert result["informational_only_not_a_gate0_stopping_criterion"] is True


def test_esm_sdpa_matches_eager_outputs_and_gradients() -> None:
    from transformers import EsmConfig, EsmForMaskedLM

    torch.manual_seed(13)
    config = EsmConfig(
        vocab_size=35,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        pad_token_id=1,
        mask_token_id=32,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        token_dropout=False,
    )
    eager = EsmForMaskedLM(config).eval()
    sdpa = copy.deepcopy(eager).eval()
    assert patch_esm_sdpa(sdpa, expected_layers=2) == 2
    input_ids = torch.tensor([[0, 4, 5, 32, 32, 2], [0, 7, 8, 9, 32, 2]])
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, -1] = 0
    eager_output = eager(input_ids=input_ids, attention_mask=attention_mask).logits
    sdpa_output = sdpa(input_ids=input_ids, attention_mask=attention_mask).logits
    torch.testing.assert_close(sdpa_output, eager_output, rtol=2e-5, atol=2e-6)
    eager_output.square().sum().backward()
    sdpa_output.square().sum().backward()
    for (_, eager_parameter), (_, sdpa_parameter) in zip(eager.named_parameters(), sdpa.named_parameters()):
        torch.testing.assert_close(sdpa_parameter.grad, eager_parameter.grad, rtol=3e-5, atol=3e-6)
