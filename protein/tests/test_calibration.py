from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
import zipfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from protein.calibration.data import (
    BackboneBatchSampler,
    SequenceRecord,
    VariantRecord,
    load_manifest,
    validate_records,
)
from protein.calibration.diagnostics import checkpoint_diagnostics
from protein.calibration.gate import coverage_gate
from protein.calibration.lora import LoRALinear
from protein.calibration.optimized_forward import OptimizedIF1Forward, batch_backbone_indices, model_logits
from protein.calibration.queue_stage1 import _run
from protein.calibration.queue_stage1 import _train_command
from protein.calibration.prepare_megascale import (
    _domain_key,
    _structure_key,
    _set_cover_clusters,
    assign_cluster_splits,
    build_manifest,
    extract_published_data,
    materialize_search_directories,
)


def test_megascale_domain_and_structure_keys_preserve_derived_wildtypes() -> None:
    assert _domain_key("1A0N.pdb") == "1A0N"
    assert _domain_key("1A0N.pdb_L7S") == "1A0N.pdb_L7S"
    assert _structure_key("1A0N.pdb_L7S") == "1A0N"
    assert _domain_key("EA|run2_0325_0005.pdb") == "EA__SEP__run2_0325_0005"
    assert _structure_key("EA|run2_0325_0005.pdb") == "EA__SEP__run2_0325_0005"


def test_tmalign_fallback_uses_deterministic_set_cover_not_transitive_components() -> None:
    scores = np.eye(4, dtype=np.float32)
    scores[0, 1], scores[1, 0] = 0.6, 0.4
    scores[1, 2], scores[2, 1] = 0.51, 0.2
    scores[2, 3], scores[3, 2] = 0.7, 0.3
    groups = _set_cover_clusters(scores, 0.5, ["a", "b", "c", "d"])
    assert sorted(sorted(group) for group in groups) == [[0, 1], [2, 3]]
    coverage = np.ones((4, 4), dtype=np.float32)
    coverage[0, 1] = coverage[1, 0] = 0.7
    covered = _set_cover_clusters(scores, 0.5, ["a", "b", "c", "d"], coverage, 0.8)
    assert sorted(sorted(group) for group in covered) == [[0], [1, 2, 3]]
from protein.calibration.training import save_checkpoint
from protein.calibration.targets import (
    EXPECTED_DENSE_WEIGHTS,
    EXPECTED_LORA,
    freeze_for_dtft,
    target_matrices,
)


class Attention(nn.Module):
    def __init__(self, dim: int, *, device: str = "cpu") -> None:
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self, name, nn.Linear(dim, dim, bias=True, device=device))


class Layer(nn.Module):
    def __init__(self, decoder: bool, *, device: str = "cpu") -> None:
        super().__init__()
        self.self_attn = Attention(512, device=device)
        if decoder:
            self.encoder_attn = Attention(512, device=device)
        self.fc1 = nn.Linear(512, 2048, device=device)
        self.fc2 = nn.Linear(2048, 512, device=device)


class FakeIF1(nn.Module):
    def __init__(self, *, device: str = "meta") -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.decoder = nn.Module()
        self.encoder.layers = nn.ModuleList([Layer(False, device=device) for _ in range(8)])
        self.decoder.layers = nn.ModuleList([Layer(True, device=device) for _ in range(8)])
        self.gvp = nn.Linear(3, 3, device=device)
        self.output_projection = nn.Linear(512, 33, device=device)


def test_validation_rejects_backbone_leakage_across_distinct_domains(tmp_path: Path) -> None:
    path = tmp_path / "shared.npy"
    np.save(path, np.zeros((3, 3, 3), dtype=np.float32))
    records = [
        SequenceRecord(
            domain_id=f"d{index}", sequence_id=f"s{index}", split=split,
            backbone_path=path, chain_id="A", native_sequence="AAA",
            target_sequence="ACA", ddg=-1.0, substitution_count=1,
            foldseek_qtm_max_to_train=1.0 if split == "train" else 0.5,
        )
        for index, split in enumerate(("train", "test"))
    ]
    with pytest.raises(ValueError, match="Backbone leakage"):
        validate_records(records)


def test_exact_if1_target_budgets_without_allocating_real_weights() -> None:
    model = FakeIF1()
    targets = target_matrices(model)
    assert len(targets) == 128
    assert sum(target.module.weight.numel() for target in targets) == EXPECTED_DENSE_WEIGHTS
    for rank, expected in EXPECTED_LORA.items():
        assert sum(target.lora_scalars(rank) for target in targets) == expected
    trainable = freeze_for_dtft(model)
    assert len(trainable) == 128
    assert all(name.endswith("weight") for name in trainable)
    assert not model.gvp.weight.requires_grad
    assert not model.output_projection.weight.requires_grad


def test_lora_initialization_and_effective_update() -> None:
    torch.manual_seed(3)
    base = nn.Linear(7, 5)
    wrapped = LoRALinear(base, rank=3, alpha=3)
    x = torch.randn(4, 7)
    assert torch.count_nonzero(wrapped.lora_A) > 0
    assert torch.count_nonzero(wrapped.lora_B) == 0
    assert torch.equal(wrapped(x), base(x))
    assert sum(p.numel() for p in wrapped.parameters() if p.requires_grad) == 3 * (7 + 5)


def test_provenance_manifest_and_indel_guard(tmp_path: Path) -> None:
    rows = []
    for index, split in enumerate(("train", "validation", "test")):
        coords = np.zeros((3, 3, 3), dtype=np.float32)
        np.save(tmp_path / f"{split}.npy", coords)
        rows.append(
            {
                "domain_id": f"d{index}",
                "sequence_id": f"s{index}",
                "split": split,
                "backbone_path": f"{split}.npy",
                "chain_id": "A",
                "native_sequence": "AAA",
                "variant_sequence": "ACA",
                "ddg": -1.0 if split == "train" else 1.0,
                "mutation_count": 1,
                "foldseek_qtm_max_to_train": 1.0 if split == "train" else 0.4,
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    metadata = {
        "dataset": "ProteinDPO curated Megascale v2 2023-04-20",
        "split_provenance": "authors-released",
        "foldseek_metric": "query-normalized-tm-score",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    manifest.with_suffix(".jsonl.metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    loaded = load_manifest(manifest)
    assert len(loaded) == 3
    assert loaded[0].target_sequence == "ACA"
    assert loaded[0].variant_sequence == "ACA"
    assert loaded[0].substitution_count == 1
    assert loaded[0].mutation_count == 1
    assert isinstance(loaded[0], SequenceRecord)
    assert isinstance(loaded[0], VariantRecord)
    metadata["split_provenance"] = "reconstructed-proteindpo-methods"
    metadata["split_seed"] = 42
    metadata["split_algorithm"] = (
        "lexicographic-cluster-ids; python-random-v1-seed-42-shuffle; floor-90%-floor-5%-remainder"
    )
    manifest.with_suffix(".jsonl.metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert len(load_manifest(manifest)) == 3
    metadata["split_seed"] = 1
    manifest.with_suffix(".jsonl.metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="provenance contract"):
        load_manifest(manifest)


def test_backbone_batch_sampler_is_complete_deterministic_and_maximally_grouped(tmp_path: Path) -> None:
    path = tmp_path / "coords.npy"
    np.save(path, np.zeros((3, 3, 3), dtype=np.float32))
    records = [
        SequenceRecord(
            domain_id=f"d{i // 5}", sequence_id=f"s{i}", split="train",
            backbone_path=path.with_name(f"b{i // 5}.npy"), chain_id="A",
            native_sequence="AAA", target_sequence="ACA", ddg=-1.0,
            substitution_count=1, foldseek_qtm_max_to_train=1.0,
        )
        for i in range(12)
    ]
    sampler = BackboneBatchSampler(records, 4, 42, shuffle=True)
    first = list(sampler)
    assert sorted(itertools.chain.from_iterable(first)) == list(range(12))
    assert len(first) == 3
    assert first == list(sampler)
    sampler.set_epoch(1)
    assert first != list(sampler)


def test_shared_backbone_forward_matches_reference_logits_loss_and_gradients(tmp_path: Path) -> None:
    class TinyIF1(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            class Encoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.proj = nn.Linear(9, 5)

                def forward(self, coords, padding_mask, confidence):
                    del confidence
                    x = self.proj(coords.flatten(2)).transpose(0, 1)
                    return {"encoder_out": [x], "encoder_padding_mask": [padding_mask]}

            class Decoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.token = nn.Embedding(8, 5)
                    self.output = nn.Linear(5, 8)

                def forward(self, previous, encoder_out):
                    context = encoder_out["encoder_out"][0].transpose(0, 1)
                    x = self.token(previous) + context[:, : previous.shape[1]]
                    return self.output(x).transpose(1, 2), {}

            self.encoder = Encoder()
            self.decoder = Decoder()

        def forward(self, coords, padding_mask, confidence, previous):
            encoded = self.encoder(coords, padding_mask, confidence)
            return self.decoder(previous, encoded)

    paths = [tmp_path / "a.npy", tmp_path / "b.npy"]
    records = []
    for i, path in enumerate((paths[0], paths[0], paths[1], paths[0])):
        records.append(type("R", (), {"backbone_path": path, "chain_id": "A"})())
    coords = torch.randn(4, 2, 3, 3)
    coords[1] = coords[0]
    coords[3] = coords[0]
    batch = {
        "coords": coords,
        "confidence": torch.ones(4, 2),
        "padding_mask": torch.zeros(4, 2, dtype=torch.bool),
        "tokens": torch.randint(0, 8, (4, 3)),
        "records": records,
        "backbone_unique": torch.tensor([0, 2]),
        "backbone_inverse": torch.tensor([0, 0, 1, 0]),
    }
    reference = TinyIF1()
    optimized = deepcopy(reference)
    logits_reference = model_logits(reference, batch, torch.device("cpu"), reuse_backbones=False)
    supplied_tokens = batch["tokens"].clone()
    logits_optimized = model_logits(
        OptimizedIF1Forward(optimized), batch, torch.device("cpu"),
        reuse_backbones=True, tokens=supplied_tokens,
    )
    torch.testing.assert_close(logits_reference, logits_optimized, rtol=0, atol=2e-7)
    target = batch["tokens"][:, 1:]
    loss_reference = torch.nn.functional.cross_entropy(logits_reference, target, reduction="sum")
    loss_optimized = torch.nn.functional.cross_entropy(logits_optimized, target, reduction="sum")
    torch.testing.assert_close(loss_reference, loss_optimized, rtol=0, atol=2e-7)
    loss_reference.backward()
    loss_optimized.backward()
    for left, right in zip(reference.parameters(), optimized.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=2e-6, atol=2e-6)
        left.data.add_(left.grad, alpha=-0.01)
        right.data.add_(right.grad, alpha=-0.01)
        torch.testing.assert_close(left, right, rtol=2e-6, atol=2e-6)

    unique_batch = dict(batch)
    unique_batch["records"] = [
        type("R", (), {"backbone_path": tmp_path / f"unique{i}.npy", "chain_id": "A"})()
        for i in range(4)
    ]
    unique_batch.pop("backbone_unique")
    unique_batch.pop("backbone_inverse")
    unique, inverse = batch_backbone_indices(unique_batch)
    assert unique.tolist() == [0, 1, 2, 3]
    assert inverse.tolist() == [0, 1, 2, 3]
    fallback = TinyIF1()
    expected = model_logits(fallback, unique_batch, torch.device("cpu"), reuse_backbones=False)
    actual = model_logits(fallback, unique_batch, torch.device("cpu"), reuse_backbones=True)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_coverage_gate_uses_both_registered_outcomes() -> None:
    ft = {"heldout_nll": 1.0, "domain_spearman_mean": 0.8}
    dtft = {"heldout_nll": 1.6, "domain_spearman_mean": 0.5}
    lora = {"heldout_nll": 2.0, "domain_spearman_mean": 0.3}
    result = coverage_gate(ft, dtft, lora)
    assert result["stop_h1_h4"]
    assert result["registered_statement"] == "target coverage is currently the dominant adaptation bottleneck"


def test_checkpoint_diagnostics_collects_dense_gradient_and_bounded_pcs() -> None:
    class Alphabet:
        padding_idx = 0

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = LoRALinear(nn.Linear(3, 4), rank=2, alpha=2)

        def forward(self, coords, padding_mask, confidence, previous):
            del coords, padding_mask, confidence
            features = torch.nn.functional.one_hot(previous % 3, num_classes=3).float()
            return self.projection(features).transpose(1, 2), {}

    model = TinyModel()
    batch = {
        "coords": torch.zeros(2, 4, 3, 3),
        "confidence": torch.ones(2, 4),
        "padding_mask": torch.zeros(2, 4, dtype=torch.bool),
        "tokens": torch.tensor([[1, 2, 3, 1], [1, 3, 2, 0]]),
        "records": [object(), object()],
    }
    result = checkpoint_diagnostics(model, [batch], Alphabet(), torch.device("cpu"))
    module = result["modules"]["projection"]
    assert module["mean_gradient"].shape == (4, 3)
    assert module["activation_pcs"]["components"].shape[1] == 3
    assert result["examples"] == 2
    assert not model.projection.base.weight.requires_grad


def test_stage_queue_skips_only_verified_completed_outputs(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "result.json"
    ledger_path = tmp_path / "ledger.json"
    ledger = []
    calls = []

    def fake_run(arguments, *, check):
        assert check
        calls.append(arguments)
        output.write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    command = ["python", "stage.py"]
    _run(command, ledger, ledger_path, [output])
    _run(command, ledger, ledger_path, [output])
    assert calls == [command]
    assert json.loads(ledger_path.read_text(encoding="utf-8"))[0]["status"] == "complete"


def test_hpo_command_retains_registered_diagnostics(tmp_path: Path) -> None:
    panel = tmp_path / "panels.json"
    command = _train_command(
        "lora8",
        7,
        1e-5,
        tmp_path / "manifest.parquet",
        tmp_path / "output",
        tmp_path / "if1.pt",
        panel,
        diagnostics=True,
    )
    assert "--diagnostic-panel" in command
    assert str(panel) in command
    assert "--no-diagnostics" not in command


def test_checkpoint_retains_registered_lora_diagnostics(tmp_path: Path) -> None:
    model = nn.Module()
    model.projection = LoRALinear(nn.Linear(3, 2), rank=2, alpha=2)
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint,
        model,
        None,
        {},
        {"epoch": 1.0},
        include_optimizer=False,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert {"metadata", "trainable_state", "rng", "adapter"} <= set(payload)
    assert "projection.A" in payload["adapter"]
    assert "projection.B" in payload["adapter"]
    assert "projection.BA" in payload["adapter"]


def test_seed42_split_materialization_and_manifest_round_trip(tmp_path: Path) -> None:
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    cluster_tsv = tmp_path / "clusters.tsv"
    cluster_tsv.write_text(
        "".join(f"d{i}.pdb\td{i}.pdb\n" for i in range(20)), encoding="utf-8"
    )
    for i in range(20):
        (pdb_dir / f"d{i}.pdb").write_text("MODEL\nEND\n", encoding="utf-8")
    split = assign_cluster_splits(cluster_tsv)
    assert {label: list(split.values()).count(label) for label in ("train", "validation", "test")} == {
        "train": 18,
        "validation": 1,
        "test": 1,
    }
    assert split == assign_cluster_splits(cluster_tsv)
    work = tmp_path / "foldseek"
    assert materialize_search_directories(pdb_dir, cluster_tsv, work) == {
        "train": 18,
        "heldout": 2,
    }
    assert materialize_search_directories(pdb_dir, cluster_tsv, work) == {
        "train": 18,
        "heldout": 2,
    }

    rows = []
    for i in range(20):
        rows.extend(
            [
                {"WT_name": f"d{i}.pdb", "aa_seq": "AAA", "mut_type": "wt", "ddG_ML": 0.0},
                {"WT_name": f"d{i}.pdb", "aa_seq": "ACA", "mut_type": "A2C", "ddG_ML": -1.0},
            ]
        )
    rows.append({"WT_name": "d0.pdb", "aa_seq": "AAC", "mut_type": "A3C", "ddG_ML": "-"})
    source = tmp_path / "megascale.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    qtm = tmp_path / "qtm.tsv"
    qtm.write_text("".join(f"d{i}.pdb\td0.pdb\t0.4\n" for i in range(20)), encoding="utf-8")
    manifest = tmp_path / "manifest.parquet"
    counts = build_manifest(source, pdb_dir, cluster_tsv, qtm, manifest)
    assert counts == {"test": 1, "train": 18, "validation": 1}
    loaded = load_manifest(manifest)
    assert len(loaded) == 20
    assert loaded[0].target_sequence == "ACA"
    assert loaded[0].substitution_count == 1
    assert loaded[0].variant_sequence == "ACA"
    assert loaded[0].mutation_count == 1
    metadata = json.loads(manifest.with_suffix(".parquet.metadata.json").read_text(encoding="utf-8"))
    assert metadata["split_seed"] == 42
    assert metadata["source_csv_sha256"]
    assert metadata["cluster_tsv_sha256"]


def test_selective_archive_extraction_is_checksum_guarded(tmp_path: Path, monkeypatch) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    processed = downloads / "Processed_K50_dG_datasets.zip"
    structures = downloads / "AlphaFold_model_PDBs.zip"
    with zipfile.ZipFile(processed, "w") as archive:
        archive.writestr(
            "nested/Tsuboyama2023_Dataset2_Dataset3_20230416.csv",
            "WT_name,aa_seq,mut_type,ddG_ML\n",
        )
        archive.writestr("unused.csv", "unused")
    with zipfile.ZipFile(structures, "w") as archive:
        archive.writestr("nested/domain.pdb", "MODEL\nEND\n")
        archive.writestr("nested/EA:design.pdb", "MODEL\nEND\n")
        archive.writestr("__MACOSX/nested/._domain.pdb", "metadata")
    import protein.calibration.prepare_megascale as preparation

    monkeypatch.setitem(
        preparation.DATASET_URLS,
        "processed",
        ("unused", hashlib.md5(processed.read_bytes()).hexdigest()),  # nosec B324
    )
    monkeypatch.setitem(
        preparation.DATASET_URLS,
        "pdbs",
        ("unused", hashlib.md5(structures.read_bytes()).hexdigest()),  # nosec B324
    )
    output = tmp_path / "source"
    inventory = extract_published_data(downloads, output)
    assert inventory["structure_count"] == 2
    assert (output / "Tsuboyama2023_Dataset2_Dataset3_20230416.csv").is_file()
    assert (output / "pdbs" / "domain.pdb").is_file()
    assert (output / "pdbs" / "EA__SEP__design.pdb").is_file()
    assert not (output / "pdbs" / "._domain.pdb").exists()
    assert not (output / "unused.csv").exists()
    (output / "pdbs" / "stale.pdb").write_text("stale", encoding="utf-8")
    rerun = extract_published_data(downloads, output)
    assert rerun["structure_count"] == 2
    assert not (output / "pdbs" / "stale.pdb").exists()


def test_evaluate_sequences_produces_neutral_and_legacy_metrics(tmp_path: Path) -> None:
    from protein.calibration.evaluation import evaluate_sequences, evaluate_variants

    class Alphabet:
        padding_idx = 0

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            class Encoder(nn.Module):
                def forward(self, coords, padding_mask, confidence):
                    del confidence
                    return {"encoder_out": [torch.zeros(coords.shape[1], coords.shape[0], 5)], "encoder_padding_mask": [padding_mask]}

            class Decoder(nn.Module):
                def forward(self, previous, encoder_out):
                    b, t = previous.shape
                    return torch.zeros(b, 8, t), {}

            self.encoder = Encoder()
            self.decoder = Decoder()

    record = SequenceRecord(
        domain_id="d0",
        sequence_id="s0",
        split="test",
        backbone_path=tmp_path / "b0.npy",
        chain_id="A",
        native_sequence="AAA",
        target_sequence="ACA",
        ddg=-1.0,
        substitution_count=1,
        foldseek_qtm_max_to_train=0.5,
    )
    batch = {
        "coords": torch.zeros(1, 3, 3, 3),
        "confidence": torch.ones(1, 3),
        "padding_mask": torch.zeros(1, 3, dtype=torch.bool),
        "tokens": torch.tensor([[1, 2, 3, 0]]),
        "records": [record],
    }
    out_dir = tmp_path / "eval"
    metrics = evaluate_sequences(DummyModel(), [batch], [batch], Alphabet(), torch.device("cpu"), out_dir)
    assert "n_sequences" in metrics
    assert "n_variants" in metrics
    assert (out_dir / "sequence_metrics.parquet").is_file()
    assert (out_dir / "variant_metrics.parquet").is_file()
    assert evaluate_variants is evaluate_sequences
