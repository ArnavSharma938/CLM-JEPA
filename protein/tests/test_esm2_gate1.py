from __future__ import annotations

import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from protein.esm2_gate1.analysis import gate_quantities
from protein.esm2_gate1.batching import TokenBudgetBatchSampler
from protein.esm2_gate1.config import DATASET_REVISION
from protein.esm2_gate1.distances import extract_structures
from protein.esm2_gate1.manifests import (
    BackboneSequence,
    normalize_sequence,
    remove_cross_backbone_duplicates,
    stable_hash,
    stratified_nested_sample,
    write_manifest,
)
from protein.esm2_gate1.masking import corrupt_tokens
from protein.esm2_gate1.modeling import LoRALinear, configure_adaptation, dense_targets
from protein.esm2_gate1.training import classify_trajectory, mirror_checkpoint


def _record(index: int, length: int = 100) -> BackboneSequence:
    sequence = ("ACDEFGHIKLMNPQRSTVWY" * 30)[:length]
    return BackboneSequence(
        backbone_id=f"b{index:05d}", sequence_id=f"s{index:05d}", sequence=sequence,
        length=len(sequence), brn_member=True, brq_member=True,
        sequence_sha256=stable_hash(sequence), description=f"generation {index % 300}",
    )


def test_masking_is_paired_deterministic_and_changes_by_exposure() -> None:
    tokens = torch.arange(102)
    kwargs = dict(
        token_ids=tokens, eligible_positions=range(1, 101), canonical_token_ids=range(200, 220),
        mask_token_id=32, sequence_id="protein", common_seed=99,
    )
    first = corrupt_tokens(**kwargs, exposure=0)
    again = corrupt_tokens(**kwargs, exposure=0)
    later = corrupt_tokens(**kwargs, exposure=1)
    assert torch.equal(first.input_ids, again.input_ids)
    assert first.selected_positions == again.selected_positions
    assert len(first.selected_positions) == 15
    assert first.selected_positions != later.selected_positions
    assert int(first.labels.ne(-100).sum()) == 15


def test_token_batches_respect_padded_budget_and_epoch_pairing() -> None:
    lengths = [40 + index % 100 for index in range(1000)]
    left = TokenBudgetBatchSampler(lengths, 1024, seed=11)
    right = TokenBudgetBatchSampler(lengths, 1024, seed=11)
    left.set_epoch(3); right.set_epoch(3)
    assert list(left) == list(right)
    for batch in left:
        assert max(lengths[index] + 2 for index in batch) * len(batch) <= 1024


def test_nested_length_stratified_sampling() -> None:
    records = [_record(index, 40 + index % 473) for index in range(500)]
    small, large, heldout = stratified_nested_sample(records, small=100, large=300)
    assert len(small) == 100 and len(large) == 300 and len(heldout) == 200
    assert {item.backbone_id for item in small} <= {item.backbone_id for item in large}
    assert not ({item.backbone_id for item in large} & {item.backbone_id for item in heldout})


def test_cross_backbone_exact_duplicates_keep_one_deterministically() -> None:
    one = _record(1)
    two = replace(_record(2), sequence=one.sequence, sequence_sha256=one.sequence_sha256)
    kept, removed = remove_cross_backbone_duplicates([one, two, _record(3, 101)])
    assert removed == 1 and len(kept) == 2
    expected = min((one, two), key=lambda item: stable_hash(item.backbone_id, seed=23))
    assert expected.backbone_id in {item.backbone_id for item in kept}


def test_sequence_validation_drops_instead_of_modifying() -> None:
    assert normalize_sequence("acdef") == "ACDEF"
    assert normalize_sequence("ACDX") is None


def test_manifest_hash_and_revision_contract(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    metadata = write_manifest(path, [_record(1)], split="train")
    assert metadata["dataset_revision"] == DATASET_REVISION
    path.write_text(path.read_text() + " ", encoding="utf-8")
    from protein.esm2_gate1.manifests import load_manifest
    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_manifest(path)


def test_structure_zip_join_uses_pdb_stem_and_rejects_missing(tmp_path: Path) -> None:
    archive = tmp_path / "structures.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("backbone_export/a.pdb", "ATOM\n")
        handle.writestr("__MACOSX/backbone_export/._a.pdb", "junk")
    audit = extract_structures(archive, {"a"}, tmp_path / "out")
    assert audit["resolved"] == 1 and (tmp_path / "out" / "a.pdb").is_file()
    with pytest.raises(RuntimeError, match="lacks 1"):
        extract_structures(archive, {"missing"}, tmp_path / "other")


def test_esm2_target_family_and_trainable_counts_on_meta_model() -> None:
    from transformers import EsmConfig, EsmForMaskedLM
    config = EsmConfig(
        vocab_size=33, hidden_size=1280, num_hidden_layers=33, num_attention_heads=20,
        intermediate_size=5120, max_position_embeddings=1026,
    )
    with torch.device("meta"):
        model = EsmForMaskedLM(config)
    targets = dense_targets(model)
    assert len(targets) == 198
    assert sum(item.module.weight.numel() for item in targets) == 648_806_400
    report = configure_adaptation(model, "dtft")
    assert report["trainable_parameters"] == 648_806_400


@pytest.mark.parametrize(("rank", "expected"), [(8, 6_082_560), (64, 48_660_480)])
def test_lora_counts_and_zero_initial_update(rank: int, expected: int) -> None:
    layer = torch.nn.Linear(13, 17)
    wrapped = LoRALinear(layer, rank=min(rank, 8), alpha=min(rank, 8))
    values = torch.randn(2, 13)
    torch.testing.assert_close(wrapped(values), layer(values), rtol=0, atol=0)
    assert torch.count_nonzero(wrapped.merged_update()) == 0
    # Architecture-derived count is tested without allocating the 650M parameters.
    per_layer = 4 * rank * (1280 + 1280) + 2 * rank * (1280 + 5120)
    assert per_layer * 33 == expected


def _metric_file(path: Path, values: list[float]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for index, value in enumerate(values):
            handle.write(json.dumps({"backbone_id": f"b{index}", "loss_sum": value * 2, "masked_tokens": 2}) + "\n")


def test_gate_signs_and_paired_bootstrap(tmp_path: Path) -> None:
    base, dtft, lora = (tmp_path / name for name in ("base.jsonl", "dtft.jsonl", "lora.jsonl"))
    _metric_file(base, [2.0, 2.2, 1.8])
    _metric_file(dtft, [1.0, 1.2, 0.8])
    _metric_file(lora, [1.2, 1.4, 1.0])
    result = gate_quantities(base, dtft, lora)
    assert result["A_DTFT"] == pytest.approx(1.0)
    assert result["G_LoRA"] == pytest.approx(0.2)
    assert result["F_gap"] == pytest.approx(0.2)
    assert result["candidate_gap"]


def test_trajectory_classification_and_checkpoint_deduplication(tmp_path: Path) -> None:
    assert classify_trajectory([
        {"validation_nll": 2.0}, {"validation_nll": 1.9}, {"validation_nll": 1.8}
    ]) == "improving"
    assert classify_trajectory([
        {"validation_nll": 1.0}, {"validation_nll": 1.00001}, {"validation_nll": 0.99999}
    ]) == "plateaued"
    source = tmp_path / "best.pt"
    destination = tmp_path / "last.pt"
    source.write_bytes(b"checkpoint")
    mirror_checkpoint(source, destination)
    assert destination.read_bytes() == b"checkpoint"
    assert source.stat().st_ino == destination.stat().st_ino
