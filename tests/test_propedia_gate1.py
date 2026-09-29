from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from esm2_gate1.modeling import LoRALinear, configure_adaptation, dense_targets
from esm2_peptide_generation.gate1_lora import (
    corrected_panel_rows,
    full_run_still_improving,
    hpo_still_improving,
)
from esm2_peptide_generation.masking import make_full_span_example, selected_cross_entropy_sum


class _TargetLayer(nn.Module):
    def __init__(self, hidden: int, intermediate: int, *, device=None) -> None:
        super().__init__()
        self.attention = nn.Module()
        self.attention.self = nn.Module()
        self.attention.self.query = nn.Linear(hidden, hidden, device=device)
        self.attention.self.key = nn.Linear(hidden, hidden, device=device)
        self.attention.self.value = nn.Linear(hidden, hidden, device=device)
        self.attention.output = nn.Module()
        self.attention.output.dense = nn.Linear(hidden, hidden, device=device)
        self.intermediate = nn.Module()
        self.intermediate.dense = nn.Linear(hidden, intermediate, device=device)
        self.output = nn.Module()
        self.output.dense = nn.Linear(intermediate, hidden, device=device)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        q = self.attention.self.query(values)
        k = self.attention.self.key(values)
        v = self.attention.self.value(values)
        values = self.attention.output.dense(q + k + v)
        return self.output.dense(torch.relu(self.intermediate.dense(values)))


class _TinyEsm(nn.Module):
    def __init__(self, hidden: int = 16, intermediate: int = 32, layers: int = 33) -> None:
        super().__init__()
        self.esm = nn.Module()
        self.esm.encoder = nn.Module()
        self.esm.encoder.layer = nn.ModuleList(
            [_TargetLayer(hidden, intermediate) for _ in range(layers)]
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for layer in self.esm.encoder.layer:
            values = layer(values)
        return values


def _meta_650m() -> nn.Module:
    model = nn.Module()
    model.esm = nn.Module()
    model.esm.encoder = nn.Module()
    model.esm.encoder.layer = nn.ModuleList(
        [_TargetLayer(1280, 5120, device="meta") for _ in range(33)]
    )
    return model


def test_lora_targets_and_expected_trainable_counts_match_dtft_definition() -> None:
    targets = dense_targets(_meta_650m())
    assert len(targets) == 198
    assert {int(item.name.split(".")[3]) for item in targets} == set(range(33))
    assert sum(item.lora_scalars(8) for item in targets) == 6_082_560
    assert sum(item.lora_scalars(64) for item in targets) == 48_660_480


@pytest.mark.parametrize("rank", [8, 64])
def test_configured_lora_only_trains_adapters_and_starts_at_base_output(rank: int) -> None:
    torch.manual_seed(17)
    model = _TinyEsm().eval()
    inputs = torch.randn(2, 4, 16)
    expected = model(inputs)

    report = configure_adaptation(model, f"lora{rank}")
    assert report["target_matrix_count"] == 198
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert len(trainable) == 396
    assert all(name.endswith(("lora_A", "lora_B")) for name in trainable)
    assert all(".base." in name or name.endswith(("lora_A", "lora_B")) for name, _ in model.named_parameters())

    torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)
    first_lora = next(module for module in model.modules() if isinstance(module, LoRALinear))
    assert torch.count_nonzero(first_lora.lora_B) == 0
    assert first_lora.alpha == rank and first_lora.scaling == 1.0
    with torch.no_grad():
        first_lora.lora_B.normal_()
    assert torch.linalg.matrix_rank(first_lora.merged_update()) <= rank


def test_corrected_panel_filter_preserves_fixed_decoys_and_no_train_overlap() -> None:
    train = [{"pair_id": "train", "peptide_sequence": "AAA"}]
    validation = [{"pair_id": "val", "peptide_sequence": "BBB"}]
    test = [
        {"pair_id": "keep", "peptide_sequence": "CCC", "receptor_cluster_id": "c1"},
        {"pair_id": "drop", "peptide_sequence": "BBB", "receptor_cluster_id": "c2"},
    ]
    decoys = [
        {"pair_id": pair_id, "decoy_receptor_id": f"d{index}"}
        for pair_id in ("keep", "drop") for index in range(4)
    ]
    original_decoys = [dict(row) for row in decoys]
    panel, corrected_decoys, removed_pair_count, shared_sequence_count = corrected_panel_rows(
        train, validation, test, decoys
    )
    assert [row["pair_id"] for row in panel] == ["keep"]
    assert [row["decoy_receptor_id"] for row in corrected_decoys] == [f"d{i}" for i in range(4)]
    assert removed_pair_count == 1
    assert shared_sequence_count == 1
    assert decoys == original_decoys


def test_full_span_masking_and_peptide_only_loss_contract_is_unchanged() -> None:
    example = make_full_span_example(
        [4, 5], [7, 8, 9], bos_token_id=1, eos_token_id=2, mask_token_id=3
    )
    assert example.input_ids == (1, 4, 5, 3, 3, 3, 2)
    assert example.labels == (-100, -100, -100, 7, 8, 9, -100)
    logits = torch.randn(1, len(example.labels), 12)
    loss = selected_cross_entropy_sum(logits, torch.tensor([example.labels]))
    reference = torch.nn.functional.cross_entropy(
        logits[0, 3:6], torch.tensor([7, 8, 9]), reduction="sum"
    )
    torch.testing.assert_close(loss, reference, rtol=0, atol=0)


def test_hpo_extension_rule_uses_four_epoch_trajectory_and_final_window() -> None:
    make_history = lambda values: [{"epoch": 0.0, "peptide_nll": 3.0}] + [
        {"epoch": 0.5 * (index + 1), "peptide_nll": value}
        for index, value in enumerate(values)
    ]
    assert hpo_still_improving(make_history([1.08, 1.07, 1.06, 1.05, 1.04, 1.03, 1.02, 1.01]))
    assert hpo_still_improving(make_history([1.0] * 8))
    assert not hpo_still_improving(make_history([1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08]))


def test_full_run_extension_rule_uses_terminal_eight_checks() -> None:
    make_history = lambda values: [{"epoch": 0.0, "peptide_nll": 3.0}] + [
        {"epoch": 0.5 * (index + 1), "peptide_nll": value}
        for index, value in enumerate(values)
    ]
    assert full_run_still_improving(make_history([2.0] * 4 + [1.998] * 4))
    assert not full_run_still_improving(make_history([2.0] * 8))
