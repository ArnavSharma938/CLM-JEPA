from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from esm2_gate1.modeling import configure_adaptation

from . import evaluate as evaluation_module
from .batching import GenerationDataset
from .config import MODEL_ID, MODEL_REVISION
from .prepare_data import sha256_file
from .token_cache import load_token_cache
from .training import evaluate_examples, load_base_model, train


GATE0_A_DTFT = 0.1973830387773504
GATE0_C_DTFT = 0.06507106981284858
GATE0_DTFT_BEST_SHA256 = "031313c623a689fc10a77c6819a35c3659ec43d7ab30303906a2d3911e984088"
INITIAL_LRS = (1e-5, 3e-5, 1e-4, 3e-4)
INPUT_TOKEN_BUDGET = 8_192
SUPERVISED_TOKEN_BUDGET = 8_192
WORKERS = 2
BOOTSTRAP_DRAWS = 20_000


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def canonical_lf_sha256(path: Path) -> str:
    """Hash the frozen JSONL representation independently of Windows CRLF conversion."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def frozen_manifest_hash(manifest_dir: Path, name: str) -> str:
    split = json.loads((manifest_dir / "frozen_split_summary.json").read_text(encoding="utf-8"))
    evaluation = json.loads((manifest_dir / "frozen_evaluation_manifests.json").read_text(encoding="utf-8"))
    if name in {"train", "validation", "test"}:
        expected = split["manifest_sha256"][name]
    elif name == "hpo_4096":
        expected = evaluation["hpo_subset"]["hpo_manifest_sha256"]
    elif name == "test_decoys":
        expected = evaluation["fixed_decoys"]["fixed_decoy_manifest_sha256"]
    else:
        raise ValueError(f"unknown frozen manifest {name}")
    actual = canonical_lf_sha256(manifest_dir / f"{name}.jsonl")
    if actual != expected:
        raise RuntimeError(f"{name} manifest content differs from frozen Gate-0 SHA-256")
    return expected


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return sha256_file(path)


def corrected_panel_rows(
    train_rows: list[dict], validation_rows: list[dict], test_rows: list[dict],
    fixed_decoys: list[dict],
) -> tuple[list[dict], list[dict], int, int]:
    validation_peptides = {row["peptide_sequence"] for row in validation_rows}
    test_peptides = {row["peptide_sequence"] for row in test_rows}
    overlap_pair_count = sum(row["peptide_sequence"] in validation_peptides for row in test_rows)
    corrected = [row for row in test_rows if row["peptide_sequence"] not in validation_peptides]
    corrected_ids = {row["pair_id"] for row in corrected}
    train_overlap_count = len({row["peptide_sequence"] for row in train_rows} & {row["peptide_sequence"] for row in corrected})
    if train_overlap_count:
        raise RuntimeError("corrected test panel overlaps training peptides")
    decoys_by_pair: dict[str, list[dict]] = defaultdict(list)
    for row in fixed_decoys:
        if row["pair_id"] in corrected_ids:
            decoys_by_pair[row["pair_id"]].append(row)
    if set(decoys_by_pair) != corrected_ids or any(len(rows) != 4 for rows in decoys_by_pair.values()):
        raise RuntimeError("fixed decoys are incomplete or not four-per-pair on corrected panel")
    for pair_id, rows in decoys_by_pair.items():
        if len({row["decoy_receptor_id"] for row in rows}) != 4:
            raise RuntimeError(f"duplicate fixed decoy receptor for pair {pair_id}")
    corrected_decoys = [row for row in fixed_decoys if row["pair_id"] in corrected_ids]
    return corrected, corrected_decoys, overlap_pair_count, len(validation_peptides & test_peptides)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare_corrected_panel(
    *, manifest_dir: Path, output_dir: Path,
) -> dict:
    """Create evaluation-only copies, preserving frozen rows and decoy assignments."""
    train_rows = read_jsonl(manifest_dir / "train.jsonl")
    validation_rows = read_jsonl(manifest_dir / "validation.jsonl")
    test_rows = read_jsonl(manifest_dir / "test.jsonl")
    fixed_decoys = read_jsonl(manifest_dir / "test_decoys.jsonl")
    hpo_rows = read_jsonl(manifest_dir / "hpo_4096.jsonl")
    canonical_hashes = {
        name: frozen_manifest_hash(manifest_dir, name)
        for name in ("train", "validation", "test", "hpo_4096", "test_decoys")
    }

    train_peptides = {row["peptide_sequence"] for row in train_rows}
    corrected, corrected_decoys, overlap_pair_count, overlap_sequence_count = corrected_panel_rows(
        train_rows, validation_rows, test_rows, fixed_decoys
    )
    overlap_pairs = [row for row in test_rows if row["peptide_sequence"] in {r["peptide_sequence"] for r in validation_rows}]
    if len(overlap_pairs) != 150 or len(corrected) != 2_213:
        raise RuntimeError(
            f"frozen overlap audit changed: excluded={len(overlap_pairs)}, retained={len(corrected)}"
        )
    if len({row["receptor_cluster_id"] for row in corrected}) != 481:
        raise RuntimeError("corrected primary panel no longer has the frozen 481 clusters")
    corrected_ids = {row["pair_id"] for row in corrected}
    if len(corrected_ids) != len(corrected):
        raise RuntimeError("duplicate pair IDs in corrected panel")
    hpo_ids = {row["pair_id"] for row in hpo_rows}
    if len(hpo_rows) != 4_096 or not hpo_ids.issubset({row["pair_id"] for row in train_rows}):
        raise RuntimeError("HPO subset does not match the frozen training population")

    panel_path = output_dir / "test_corrected.jsonl"
    decoy_path = output_dir / "test_corrected_decoys.jsonl"
    panel_hash = write_jsonl(panel_path, corrected)
    decoy_hash = write_jsonl(decoy_path, corrected_decoys)
    result = {
        "status": "verified",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "original_test_pair_count": len(test_rows),
        "validation_test_shared_peptide_sequence_count": overlap_sequence_count,
        "excluded_test_pair_count": overlap_pair_count,
        "corrected_test_pair_count": len(corrected),
        "corrected_test_receptor_cluster_count": len({row["receptor_cluster_id"] for row in corrected}),
        "train_test_peptide_overlap_count": len(train_peptides & {row["peptide_sequence"] for row in corrected}),
        "validation_manifest_sha256": sha256_file(manifest_dir / "validation.jsonl"),
        "validation_manifest_canonical_lf_sha256": canonical_hashes["validation"],
        "training_manifest_sha256": sha256_file(manifest_dir / "train.jsonl"),
        "training_manifest_canonical_lf_sha256": canonical_hashes["train"],
        "frozen_test_manifest_sha256": sha256_file(manifest_dir / "test.jsonl"),
        "frozen_test_manifest_canonical_lf_sha256": canonical_hashes["test"],
        "frozen_decoy_manifest_sha256": sha256_file(manifest_dir / "test_decoys.jsonl"),
        "frozen_decoy_manifest_canonical_lf_sha256": canonical_hashes["test_decoys"],
        "hpo_manifest_sha256": sha256_file(manifest_dir / "hpo_4096.jsonl"),
        "hpo_manifest_canonical_lf_sha256": canonical_hashes["hpo_4096"],
        "corrected_test_manifest": str(panel_path),
        "corrected_test_manifest_sha256": panel_hash,
        "corrected_decoy_manifest": str(decoy_path),
        "corrected_decoy_manifest_sha256": decoy_hash,
        "decoy_count_per_pair": 4,
        "gate0_A_DTFT_full_precision": GATE0_A_DTFT,
        "gate0_C_DTFT_full_precision": GATE0_C_DTFT,
        "delta_task": 0.10 * GATE0_A_DTFT,
        "delta_cond": 0.10 * GATE0_C_DTFT,
        "note": "Evaluation-only filtered copies; frozen train, validation, test, and decoy assignments are unchanged.",
    }
    atomic_json(output_dir / "preflight.json", result)
    return result


def verify_existing_reference_assets(
    *, panel_path: Path, decoys_path: Path, gate0_root: Path, output_path: Path,
) -> dict:
    panel = read_jsonl(panel_path)
    pair_ids = {row["pair_id"] for row in panel}
    expected_decoys: dict[str, tuple[str, ...]] = defaultdict(tuple)
    grouped_decoys: dict[str, list[str]] = defaultdict(list)
    for row in read_jsonl(decoys_path):
        grouped_decoys[row["pair_id"]].append(row["decoy_receptor_id"])
    expected_decoys = {pair_id: tuple(sorted(ids)) for pair_id, ids in grouped_decoys.items()}
    if set(expected_decoys) != pair_ids or any(len(ids) != 4 for ids in expected_decoys.values()):
        raise RuntimeError("corrected manifest does not contain four fixed decoys for every pair")
    model_paths = {
        "base": gate0_root / "base",
        "dtft_best": gate0_root / "dtft_best",
        "svd_r8": gate0_root / "svd_screen" / "svd_r8",
        "svd_r64": gate0_root / "svd_screen" / "svd_r64",
    }
    panels_by_id = {row["pair_id"]: row for row in panel}
    model_audits = {}
    for name, directory in model_paths.items():
        true_rows = [row for row in read_jsonl(directory / "test_true.jsonl") if row["pair_id"] in pair_ids]
        decoy_rows = [row for row in read_jsonl(directory / "test_decoy.jsonl") if row["pair_id"] in pair_ids]
        if {row["pair_id"] for row in true_rows} != pair_ids:
            raise RuntimeError(f"{name} true-score file does not cover corrected panel IDs")
        grouped: dict[str, list[str]] = defaultdict(list)
        for row in decoy_rows:
            grouped[row["pair_id"]].append(row["receptor_id"])
            if row["receptor_cluster_id"] != panels_by_id[row["pair_id"]]["receptor_cluster_id"]:
                raise RuntimeError(f"{name} decoy score lost its true receptor-cluster assignment")
        observed_decoys = {pair_id: tuple(sorted(ids)) for pair_id, ids in grouped.items()}
        if observed_decoys != expected_decoys:
            raise RuntimeError(f"{name} evaluation used a different fixed decoy set")
        model_audits[name] = {
            "true_score_sha256": sha256_file(directory / "test_true.jsonl"),
            "decoy_score_sha256": sha256_file(directory / "test_decoy.jsonl"),
            "retained_true_pair_count": len(true_rows),
            "retained_decoy_score_count": len(decoy_rows),
            "exact_fixed_decoys_match": True,
        }
    dtft_summary = json.loads((model_paths["dtft_best"] / "evaluation_summary.json").read_text(encoding="utf-8"))
    actual_dtft_hash = dtft_summary["test_true"]["checkpoint_sha256"]
    if actual_dtft_hash != GATE0_DTFT_BEST_SHA256:
        raise RuntimeError("Gate-0 DTFT scores are not from the validation-selected epoch-1 checkpoint")
    result = {
        "status": "verified",
        "corrected_panel_sha256": sha256_file(panel_path),
        "corrected_decoy_manifest_sha256": sha256_file(decoys_path),
        "pair_count": len(pair_ids),
        "receptor_cluster_count": len({row["receptor_cluster_id"] for row in panel}),
        "fixed_decoys_per_pair": 4,
        "dtft_checkpoint_sha256": actual_dtft_hash,
        "model_score_assets": model_audits,
    }
    atomic_json(output_path, result)
    return result


def prepare_precision_extension(
    *, primary_manifest_dir: Path, extension_manifest_dir: Path, output_dir: Path,
) -> dict:
    """Filter the pre-existing Gate-0 unused-cluster panel only when precision requires it."""
    train_rows = read_jsonl(primary_manifest_dir / "train.jsonl")
    validation_rows = read_jsonl(primary_manifest_dir / "validation.jsonl")
    extension_validation = read_jsonl(extension_manifest_dir / "validation.jsonl")
    if canonical_lf_sha256(extension_manifest_dir / "validation.jsonl") != frozen_manifest_hash(
        primary_manifest_dir, "validation"
    ):
        raise RuntimeError("precision extension validation manifest differs from frozen validation")
    if [row["pair_id"] for row in extension_validation] != [row["pair_id"] for row in validation_rows]:
        raise RuntimeError("precision extension validation row membership changed")
    original_report = json.loads((extension_manifest_dir / "expansion_report.json").read_text(encoding="utf-8"))
    if original_report["source_test_sha256"] != frozen_manifest_hash(primary_manifest_dir, "test"):
        raise RuntimeError("precision extension was not built from the frozen original test set")
    if original_report["source_train_sha256"] != frozen_manifest_hash(primary_manifest_dir, "train"):
        raise RuntimeError("precision extension was built from another training manifest")
    expected_expanded_hash = original_report["expanded_test_sha256"]
    if canonical_lf_sha256(extension_manifest_dir / "test.jsonl") != expected_expanded_hash:
        raise RuntimeError("unused-cluster test extension does not match its frozen hash")
    expected_decoy_hash = original_report["fixed_decoys"]["fixed_decoy_manifest_sha256"]
    if canonical_lf_sha256(extension_manifest_dir / "test_decoys.jsonl") != expected_decoy_hash:
        raise RuntimeError("unused-cluster decoy manifest does not match its frozen hash")

    expanded_rows = read_jsonl(extension_manifest_dir / "test.jsonl")
    fixed_decoys = read_jsonl(extension_manifest_dir / "test_decoys.jsonl")
    corrected, corrected_decoys, overlap_pairs, overlap_sequences = corrected_panel_rows(
        train_rows, validation_rows, expanded_rows, fixed_decoys
    )
    primary_corrected = read_jsonl(Path("runs/protein/propedia26_gate1/inputs/test_corrected.jsonl"))
    primary_ids = {row["pair_id"] for row in primary_corrected}
    expanded_ids = {row["pair_id"] for row in corrected}
    if not primary_ids.issubset(expanded_ids):
        raise RuntimeError("precision extension dropped corrected primary-panel pairs")
    if len(expanded_ids) != len(corrected):
        raise RuntimeError("duplicate pair IDs in corrected precision extension")
    panel_path = output_dir / "test_precision_extension.jsonl"
    decoy_path = output_dir / "test_precision_extension_decoys.jsonl"
    panel_hash = write_jsonl(panel_path, corrected)
    decoy_hash = write_jsonl(decoy_path, corrected_decoys)
    result = {
        "status": "evaluation_only_extension_verified",
        "source_expansion_report_sha256": sha256_file(extension_manifest_dir / "expansion_report.json"),
        "source_expanded_test_sha256": expected_expanded_hash,
        "source_fixed_decoy_manifest_sha256": expected_decoy_hash,
        "validation_manifest_sha256": frozen_manifest_hash(primary_manifest_dir, "validation"),
        "training_manifest_sha256": frozen_manifest_hash(primary_manifest_dir, "train"),
        "original_expanded_test_pairs": len(expanded_rows),
        "validation_shared_peptide_sequences_removed": overlap_sequences,
        "validation_overlapping_pair_rows_removed": overlap_pairs,
        "corrected_expanded_test_pairs": len(corrected),
        "corrected_expanded_test_receptor_clusters": len({row["receptor_cluster_id"] for row in corrected}),
        "primary_panel_pairs_retained": len(primary_ids),
        "train_test_peptide_overlap_count": 0,
        "decoys_per_pair": 4,
        "corrected_panel_sha256": panel_hash,
        "corrected_decoy_manifest_sha256": decoy_hash,
        "panel_path": str(panel_path),
        "decoy_manifest_path": str(decoy_path),
        "note": "Gate-0 manifests and assignments are unchanged; evaluation extension only.",
    }
    atomic_json(output_dir / "precision_extension_preflight.json", result)
    return result


def best_two_check_mean(history: list[dict]) -> tuple[float, int | None]:
    values = [float(row["peptide_nll"]) for row in history if float(row.get("epoch", 0)) > 0]
    if len(values) < 2:
        return math.inf, None
    means = [(left + right) / 2 for left, right in zip(values, values[1:])]
    index = min(range(len(means)), key=means.__getitem__)
    return means[index], index


def hpo_still_improving(history: list[dict]) -> bool:
    values = [float(row["peptide_nll"]) for row in history if float(row.get("epoch", 0)) > 0]
    if len(values) < 8:
        return False
    pair_means = [(left + right) / 2 for left, right in zip(values, values[1:])]
    best_window_touches_final = pair_means[-1] <= min(pair_means) + 1e-12
    trailing_improvement = (
        sum(values[-8:-4]) / 4 - sum(values[-4:]) / 4 >= 0.001
    )
    return best_window_touches_final or trailing_improvement


def _candidate_summary(directory: Path, lr: float, rank: int) -> dict:
    history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
    execution = json.loads((directory / "execution.json").read_text(encoding="utf-8"))
    score, best_window_start = best_two_check_mean(history)
    values = [float(row["peptide_nll"]) for row in history if float(row.get("epoch", 0)) > 0]
    best_index = min(range(len(values)), key=values.__getitem__)
    return {
        "rank": rank,
        "peak_learning_rate": lr,
        "trajectory": history,
        "best_two_consecutive_check_mean_nll": score,
        "best_two_check_window_start_index": best_window_start,
        "minimum_validation_nll": min(values),
        "terminal_validation_nll": values[-1],
        "best_validation_nll": values[best_index],
        "best_validation_epoch": next(
            row["epoch"] for row in history
            if float(row.get("epoch", 0)) > 0 and float(row["peptide_nll"]) == values[best_index]
        ),
        "still_improving_at_four_epoch_screen": hpo_still_improving(history),
        "numerical_stability": execution["numerical_stability"],
        "maximum_gradient_norm_before_clip": execution["maximum_gradient_norm_before_clip"],
        "optimizer_steps": execution["optimizer_steps"],
        "epochs_completed": execution["epochs_completed"],
        "best_checkpoint_sha256": sha256_file(directory / "best.pt"),
        "config_sha256": sha256_file(directory / "config.json"),
    }


def _atomic_ledger(run_root: Path, stage: str, rank: int, detail: dict) -> None:
    atomic_json(run_root / "ledger.json", {
        "active_stage": stage,
        "rank": rank,
        "updated_unix_time": time.time(),
        **detail,
    })


def run_rank_hpo(
    *, rank: int, manifest_dir: Path, cache_path: Path, base_summary_path: Path,
    output_dir: Path, run_root: Path,
) -> dict:
    if rank not in {8, 64}:
        raise ValueError("Gate 1 only tunes vanilla LoRA ranks 8 and 64")
    mode = f"lora{rank}"
    hpo_manifest = manifest_dir / "hpo_4096.jsonl"
    validation_manifest = manifest_dir / "validation.jsonl"
    base_summary = json.loads(base_summary_path.read_text(encoding="utf-8"))
    baseline = base_summary["validation_true"]
    frozen_train_hash = frozen_manifest_hash(manifest_dir, "train")
    frozen_validation_hash = frozen_manifest_hash(manifest_dir, "validation")
    frozen_hpo_hash = frozen_manifest_hash(manifest_dir, "hpo_4096")
    if baseline["manifest_sha256"] != frozen_validation_hash:
        raise RuntimeError("Base validation score does not match the frozen validation manifest")
    common = {
        "rank": rank,
        "hpo_seed": 7,
        "evidence_seed": 11,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "objective": "full-span peptide mask; cross-entropy only on peptide positions",
        "adaptation_mode": mode,
        "target_matrix_count": 198,
        "target_definition": "Q/K/V, attention output, FFN input/output across all 33 layers",
        "trainable_count": 6_082_560 if rank == 8 else 48_660_480,
        "training_manifest_sha256": sha256_file(manifest_dir / "train.jsonl"),
        "training_manifest_canonical_lf_sha256": frozen_train_hash,
        "hpo_manifest_sha256": sha256_file(hpo_manifest),
        "hpo_manifest_canonical_lf_sha256": frozen_hpo_hash,
        "validation_manifest_sha256": sha256_file(validation_manifest),
        "validation_manifest_canonical_lf_sha256": frozen_validation_hash,
        "token_cache_sha256": sha256_file(cache_path),
        "base_summary_sha256": sha256_file(base_summary_path),
        "base_validation_nll": baseline["peptide_nll"],
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01, "fused": True},
        "precision": "BF16 autocast; TF32 matmuls",
        "attention_backend": "uncompiled PyTorch SDPA",
        "input_token_budget": INPUT_TOKEN_BUDGET,
        "supervised_peptide_token_budget": SUPERVISED_TOKEN_BUDGET,
        "workers": WORKERS,
        "gradient_clip_norm": 1.0,
        "warmup_fraction": 0.05,
        "cosine_floor_fraction": 0.1,
        "schedule_horizon_epochs": 10,
        "validation_cadence_epochs": 0.5,
        "initial_candidates": list(INITIAL_LRS),
    }
    rank_dir = output_dir / f"rank{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    report_path = rank_dir / "hpo_report.json"
    outcomes: dict[float, dict] = {}

    def save_progress(status: str, current_lr: float | None = None) -> None:
        atomic_json(report_path, {
            **common,
            "status": status,
            "candidate_results": {f"{lr:.0e}": item for lr, item in sorted(outcomes.items())},
            "selected_peak_learning_rate": None,
            "current_candidate_lr": current_lr,
        })

    def run_one(lr: float, maximum_epochs: int = 4, extend: bool = False) -> dict:
        directory = rank_dir / f"lr_{lr:.0e}"
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "execution.json").exists():
            existing_execution = json.loads((directory / "execution.json").read_text(encoding="utf-8"))
            if float(existing_execution["epochs_completed"]) >= maximum_epochs:
                result = _candidate_summary(directory, lr, rank)
                outcomes[lr] = result
                return result
        resume = directory / "latest.pt" if (directory / "latest.pt").exists() else None
        _atomic_ledger(run_root, "lora_hpo", rank, {
            "current_candidate_lr": lr,
            "candidate_run_dir": str(directory),
            "latest_checkpoint": str(directory / "latest.pt") if resume else None,
        })
        train(
            train_manifest=hpo_manifest,
            validation_manifest=validation_manifest,
            cache_path=cache_path,
            output_dir=directory,
            peak_lr=lr,
            seed=7,
            token_budget=INPUT_TOKEN_BUDGET,
            supervised_budget=SUPERVISED_TOKEN_BUDGET,
            maximum_epochs=maximum_epochs,
            schedule_epochs=10,
            workers=WORKERS,
            attention_backend="sdpa",
            adaptation_mode=mode,
            initial_validation=baseline if resume is None else None,
            resume=resume,
        )
        result = _candidate_summary(directory, lr, rank)
        result["extended_to_six_epochs"] = extend
        outcomes[lr] = result
        save_progress("candidate_complete", lr)
        return result

    for lr in INITIAL_LRS:
        try:
            save_progress("running_initial_grid", lr)
            run_one(lr, maximum_epochs=4)
        except (FloatingPointError, torch.cuda.OutOfMemoryError) as error:
            outcomes[lr] = {"rank": rank, "peak_learning_rate": lr, "numerical_stability": "failed", "failure": repr(error)}
            torch.cuda.empty_cache()
            save_progress("candidate_failed", lr)

    for lr in INITIAL_LRS:
        if outcomes.get(lr, {}).get("numerical_stability") != "stable":
            continue
        if outcomes[lr]["still_improving_at_four_epoch_screen"]:
            try:
                run_one(lr, maximum_epochs=6, extend=True)
                outcomes[lr]["extended_to_six_epochs"] = True
            except (FloatingPointError, torch.cuda.OutOfMemoryError) as error:
                outcomes[lr]["numerical_stability"] = "failed_during_extension"
                outcomes[lr]["extension_failure"] = repr(error)
                torch.cuda.empty_cache()
        save_progress("initial_candidates_scored", lr)

    def choose(candidates: list[float]) -> float:
        stable = [lr for lr in candidates if outcomes.get(lr, {}).get("numerical_stability") == "stable"]
        if not stable:
            raise RuntimeError(f"all rank-{rank} LR candidates failed")
        best = min(float(outcomes[lr]["best_two_consecutive_check_mean_nll"]) for lr in stable)
        return min(lr for lr in stable if float(outcomes[lr]["best_two_consecutive_check_mean_nll"]) <= best + 0.002)

    candidates = [lr for lr in INITIAL_LRS if outcomes.get(lr, {}).get("numerical_stability") == "stable"]
    selected = choose(candidates)
    boundary_expansion = None
    if selected == 3e-4:
        boundary_expansion = 1e-3
    elif selected == 1e-5:
        boundary_expansion = 3e-6
    if boundary_expansion is not None:
        try:
            run_one(boundary_expansion, maximum_epochs=4)
            if outcomes[boundary_expansion]["still_improving_at_four_epoch_screen"]:
                run_one(boundary_expansion, maximum_epochs=6, extend=True)
                outcomes[boundary_expansion]["extended_to_six_epochs"] = True
            if outcomes[boundary_expansion].get("numerical_stability") == "stable":
                candidates.append(boundary_expansion)
        except (FloatingPointError, torch.cuda.OutOfMemoryError) as error:
            outcomes[boundary_expansion] = {
                "rank": rank, "peak_learning_rate": boundary_expansion,
                "numerical_stability": "failed", "failure": repr(error),
            }
            torch.cuda.empty_cache()
    selected = choose(candidates)
    report = {
        **common,
        "status": "complete",
        "candidate_results": {f"{lr:.0e}": item for lr, item in sorted(outcomes.items())},
        "boundary_expansion": f"{boundary_expansion:.0e}" if boundary_expansion is not None else None,
        "selected_peak_learning_rate": selected,
        "selection_rule": "minimum best two-consecutive-check mean; within 0.002 NLL select the lower LR; one outer probe only if the selected initial-grid LR is an edge",
    }
    atomic_json(report_path, report)
    _atomic_ledger(run_root, "lora_hpo_complete", rank, {
        "selected_peak_learning_rate": selected,
        "hpo_report": str(report_path),
        "hpo_report_sha256": sha256_file(report_path),
    })
    return report


def full_run_still_improving(history: list[dict]) -> bool:
    values = [float(row["peptide_nll"]) for row in history if float(row.get("epoch", 0)) > 0]
    if len(values) < 8:
        return False
    newest_record = any(
        index >= len(values) - 2 and value <= min(values[:index]) - 0.001
        for index, value in enumerate(values) if index > 0
    )
    trailing_improvement = sum(values[-8:-4]) / 4 - sum(values[-4:]) / 4 >= 0.001
    return newest_record or trailing_improvement


def run_evidence(
    *, rank: int, seed: int, manifest_dir: Path, cache_path: Path,
    hpo_report_path: Path, output_dir: Path, run_root: Path,
) -> dict:
    if rank not in {8, 64} or seed not in {11, 23}:
        raise ValueError("Gate 1 evidence supports ranks 8/64 and seeds 11/23 only")
    hpo = json.loads(hpo_report_path.read_text(encoding="utf-8"))
    if hpo["rank"] != rank or hpo.get("status") != "complete":
        raise RuntimeError("rank-matched HPO selection is not complete")
    lr = float(hpo["selected_peak_learning_rate"])
    if canonical_lf_sha256(manifest_dir / "train.jsonl") != hpo["training_manifest_canonical_lf_sha256"]:
        raise RuntimeError("full-run training content differs from the HPO/frozen Gate-0 training manifest")
    if canonical_lf_sha256(manifest_dir / "validation.jsonl") != hpo["validation_manifest_canonical_lf_sha256"]:
        raise RuntimeError("full-run validation content differs from the HPO/frozen Gate-0 validation manifest")
    if canonical_lf_sha256(manifest_dir / "hpo_4096.jsonl") != hpo["hpo_manifest_canonical_lf_sha256"]:
        raise RuntimeError("HPO pair content differs from the frozen 4,096-pair subset")
    mode = f"lora{rank}"
    output_dir.mkdir(parents=True, exist_ok=True)
    execution_path = output_dir / "execution.json"
    if execution_path.exists() and (output_dir / "convergence_summary.json").exists():
        existing = json.loads(execution_path.read_text(encoding="utf-8"))
        if existing.get("numerical_stability") == "stable":
            return existing

    def continue_to(ceiling: int, resume: Path | None) -> dict:
        _atomic_ledger(run_root, "lora_evidence", rank, {
            "seed": seed,
            "peak_learning_rate": lr,
            "ceiling_epochs": ceiling,
            "run_dir": str(output_dir),
            "resume_checkpoint": str(resume) if resume else None,
        })
        return train(
            train_manifest=manifest_dir / "train.jsonl",
            validation_manifest=manifest_dir / "validation.jsonl",
            cache_path=cache_path,
            output_dir=output_dir,
            peak_lr=lr,
            seed=seed,
            token_budget=INPUT_TOKEN_BUDGET,
            supervised_budget=SUPERVISED_TOKEN_BUDGET,
            maximum_epochs=ceiling,
            schedule_epochs=10,
            workers=WORKERS,
            attention_backend="sdpa",
            adaptation_mode=mode,
            minimum_epochs_for_stopping=5,
            stopping_patience=4,
            resume=resume,
        )

    latest = output_dir / "latest.pt"
    if execution_path.exists():
        execution_path.unlink()
    execution = continue_to(12, latest if latest.exists() else None)
    ceiling = 12
    while (
        not execution["stopped_early"]
        and int(execution["completed_full_epochs"]) >= ceiling
        and ceiling < 20
        and full_run_still_improving(json.loads((output_dir / "history.json").read_text(encoding="utf-8")))
    ):
        ceiling = min(ceiling + 4, 20)
        execution = continue_to(ceiling, output_dir / "latest.pt")

    history = json.loads((output_dir / "history.json").read_text(encoding="utf-8"))
    still_improving = full_run_still_improving(history) if int(execution["completed_full_epochs"]) >= ceiling else False
    if execution["stopped_early"]:
        convergence = "early_stopped_after_minimum"
    elif ceiling == 20 and int(execution["completed_full_epochs"]) >= 20 and still_improving:
        convergence = "unresolved_hard_ceiling"
    elif int(execution["completed_full_epochs"]) >= ceiling and still_improving:
        convergence = "still_improving_at_current_ceiling"
    else:
        convergence = "plateau_at_ceiling"
    best_row = min((row for row in history if float(row.get("epoch", 0)) > 0), key=lambda row: row["peptide_nll"])
    result = {
        **execution,
        "rank": rank,
        "seed": seed,
        "peak_learning_rate": lr,
        "epochs_ceiling_reached": ceiling,
        "convergence_classification": convergence,
        "best_epoch": best_row["epoch"],
        "best_validation_peptide_nll": best_row["peptide_nll"],
        "terminal_validation_peptide_nll": history[-1]["peptide_nll"],
        "selected_checkpoint": str(output_dir / "best.pt"),
        "selected_checkpoint_sha256": sha256_file(output_dir / "best.pt"),
        "last_eight_validation_observations": [row for row in history if float(row.get("epoch", 0)) > 0][-8:],
        "hpo_report_sha256": sha256_file(hpo_report_path),
    }
    atomic_json(output_dir / "convergence_summary.json", result)
    _atomic_ledger(run_root, "lora_evidence_complete", rank, {
        "seed": seed,
        "convergence_classification": convergence,
        "selected_checkpoint": str(output_dir / "best.pt"),
        "selected_checkpoint_sha256": result["selected_checkpoint_sha256"],
        "run_dir": str(output_dir),
    })
    return result


def _load_adapter_checkpoint(model, checkpoint: Path, rank: int, seed: int) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["config"].get("mode") != f"lora{rank}":
        raise RuntimeError("checkpoint is not for the requested LoRA rank")
    if payload["config"].get("seed") != seed:
        raise RuntimeError("checkpoint is not from the requested evidence seed")
    if payload["config"].get("model_revision") != MODEL_REVISION:
        raise RuntimeError("checkpoint was trained from a different ESM-2 revision")
    parameters = dict(model.named_parameters())
    expected = {name for name, parameter in parameters.items() if parameter.requires_grad}
    state = payload["trainable_state"]
    if expected != set(state) or any(not name.endswith(("lora_A", "lora_B")) for name in expected):
        raise RuntimeError("checkpoint trainable set does not match adapter-only target modules")
    with torch.no_grad():
        for name, value in state.items():
            if parameters[name].shape != value.shape:
                raise RuntimeError(f"adapter tensor shape mismatch: {name}")
            parameters[name].copy_(value.to(parameters[name].device))
    return payload


def run_adapter_evaluation(
    *, rank: int, seed: int, checkpoint_path: Path, cache_path: Path,
    panel_path: Path, decoys_path: Path, output_dir: Path,
) -> dict:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Gate-1 ESM-2 evaluation requires CUDA BF16")
    started = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = True
    cache = load_token_cache(cache_path)
    model, _ = load_base_model("sdpa")
    counts = configure_adaptation(model, f"lora{rank}")
    expected_count = 6_082_560 if rank == 8 else 48_660_480
    if counts["target_matrix_count"] != 198 or counts["trainable_parameters"] != expected_count:
        raise RuntimeError("evaluation model has a different LoRA target set than training")
    model.to(torch.device("cuda"))
    payload = _load_adapter_checkpoint(model, checkpoint_path, rank, seed)
    dataset = GenerationDataset(cache, panel_path)
    true_examples = [
        {"row": item["row"], "receptor_tokens": item["receptor_tokens"], "peptide_tokens": item["peptide_tokens"]}
        for item in dataset.items
    ]
    decoy_examples = evaluation_module.decoy_examples(dataset, decoys_path, cache)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, examples in (("test_true", true_examples), ("test_decoy", decoy_examples)):
        result = evaluate_examples(
            model, examples, cache, token_budget=INPUT_TOKEN_BUDGET, workers=WORKERS,
        )
        rows = result.pop("per_example")
        score_path = output_dir / f"{name}.jsonl"
        score_hash = write_jsonl(score_path, rows)
        results[name] = {
            **result,
            "model": f"lora_r{rank}_seed{seed}",
            "rank": rank,
            "seed": seed,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "panel_manifest_sha256": sha256_file(panel_path),
            "decoy_manifest_sha256": sha256_file(decoys_path) if name == "test_decoy" else None,
            "score_sha256": score_hash,
            "attention_backend": "uncompiled PyTorch SDPA",
            "input_token_budget": INPUT_TOKEN_BUDGET,
        }
    results["total_wall_seconds"] = time.perf_counter() - started
    results["status"] = "complete"
    atomic_json(output_dir / "evaluation_summary.json", results)
    del model
    torch.cuda.empty_cache()
    return results


def _nll_rows(path: Path, pair_ids: set[str]) -> dict[str, float]:
    rows = read_jsonl(path)
    selected = [row for row in rows if row["pair_id"] in pair_ids]
    result = {}
    for row in selected:
        if row["pair_id"] in result:
            raise RuntimeError(f"duplicate true-pair score for {row['pair_id']} in {path}")
        result[row["pair_id"]] = float(row["loss_sum"]) / int(row["supervised_tokens"])
    if set(result) != pair_ids:
        raise RuntimeError(f"evaluation score IDs do not match corrected panel in {path}")
    return result


def _decoy_rows(path: Path, pair_ids: set[str]) -> tuple[dict[str, float], dict[str, tuple[str, ...]]]:
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in read_jsonl(path):
        pair_id = row["pair_id"]
        if pair_id not in pair_ids:
            continue
        grouped[pair_id].append((row["receptor_id"], float(row["loss_sum"]) / int(row["supervised_tokens"])))
    if set(grouped) != pair_ids or any(len(rows) != 4 for rows in grouped.values()):
        raise RuntimeError(f"fixed decoy scores are incomplete in {path}")
    means = {pair_id: float(np.mean([score for _, score in rows])) for pair_id, rows in grouped.items()}
    decoy_ids = {pair_id: tuple(sorted(receptor for receptor, _ in rows)) for pair_id, rows in grouped.items()}
    return means, decoy_ids


def _bootstrap_many(values: dict[str, np.ndarray], *, draws: int, seed: int) -> dict[str, list[float]]:
    names = list(values)
    arrays = [np.asarray(values[name], dtype=np.float64) for name in names]
    lengths = {array.size for array in arrays}
    if len(lengths) != 1 or not lengths or 0 in lengths:
        raise ValueError("paired bootstrap arrays must have the same nonzero cluster count")
    n = arrays[0].size
    generator = np.random.default_rng(seed)
    samples = {name: np.empty(draws, dtype=np.float64) for name in names}
    for start in range(0, draws, 512):
        stop = min(draws, start + 512)
        indices = generator.integers(0, n, size=(stop - start, n))
        for name, array in zip(names, arrays):
            samples[name][start:stop] = array[indices].mean(axis=1)
    return {name: np.percentile(sample, [2.5, 97.5]).astype(float).tolist() for name, sample in samples.items()}


def _cluster_means(pair_values: dict[str, float], panel_rows: list[dict]) -> tuple[list[str], np.ndarray]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in panel_rows:
        grouped[row["receptor_cluster_id"]].append(pair_values[row["pair_id"]])
    clusters = sorted(grouped)
    return clusters, np.asarray([np.mean(grouped[cluster]) for cluster in clusters], dtype=np.float64)


def _score_model_directory(directory: Path, pair_ids: set[str]) -> tuple[dict[str, float], dict[str, float], dict[str, tuple[str, ...]]]:
    true = _nll_rows(directory / "test_true.jsonl", pair_ids)
    decoy, decoy_ids = _decoy_rows(directory / "test_decoy.jsonl", pair_ids)
    return true, decoy, decoy_ids


def _classify(ci: list[float], threshold: float) -> str:
    if ci[0] > threshold:
        return "meaningful deficit"
    if ci[1] < threshold:
        return "no materially meaningful deficit"
    return "UNRESOLVED"


def analyze_primary_panel(
    *, panel_path: Path, gate0_reference_dir: Path, svd_root: Path,
    gate1_root: Path, draws: int = BOOTSTRAP_DRAWS,
) -> dict:
    panel_rows = read_jsonl(panel_path)
    pair_ids = {row["pair_id"] for row in panel_rows}
    if len(pair_ids) != 2_213 or len({row["receptor_cluster_id"] for row in panel_rows}) != 481:
        raise RuntimeError("analysis requires the corrected 2,213-pair/481-cluster primary panel")
    clusters = sorted({row["receptor_cluster_id"] for row in panel_rows})
    base_true, base_decoy, base_decoy_ids = _score_model_directory(gate0_reference_dir / "base", pair_ids)
    dtft_true, dtft_decoy, dtft_decoy_ids = _score_model_directory(gate0_reference_dir / "dtft_best", pair_ids)
    if base_decoy_ids != dtft_decoy_ids:
        raise RuntimeError("Base and DTFT use different fixed decoy receptors")

    base_true_cluster = _cluster_means(base_true, panel_rows)[1]
    dtft_true_cluster = _cluster_means(dtft_true, panel_rows)[1]
    a_cluster = base_true_cluster - dtft_true_cluster
    base_specificity_pair = {pair_id: base_decoy[pair_id] - base_true[pair_id] for pair_id in pair_ids}
    dtft_specificity_pair = {pair_id: dtft_decoy[pair_id] - dtft_true[pair_id] for pair_id in pair_ids}
    base_specificity_cluster = _cluster_means(base_specificity_pair, panel_rows)[1]
    dtft_specificity_cluster = _cluster_means(dtft_specificity_pair, panel_rows)[1]
    c_cluster = dtft_specificity_cluster - base_specificity_cluster
    a_point = float(a_cluster.mean())
    c_point = float(c_cluster.mean())
    if abs(a_point - GATE0_A_DTFT) > 1e-8 or abs(c_point - GATE0_C_DTFT) > 1e-8:
        raise RuntimeError(
            f"corrected Gate-0 reference mismatch: A={a_point:.16g}, C={c_point:.16g}"
        )
    thresholds = {"delta_task": 0.10 * GATE0_A_DTFT, "delta_cond": 0.10 * GATE0_C_DTFT}

    common_cluster_ids = _cluster_means(base_true, panel_rows)[0]
    outputs: dict[str, dict] = {}
    reference_decoy_ids = base_decoy_ids
    for rank in (8, 64):
        rank_data = {}
        for seed in (11, 23):
            eval_dir = gate1_root / "evaluations" / f"rank{rank}_seed{seed}"
            if not (eval_dir / "evaluation_summary.json").exists():
                continue
            lora_true, lora_decoy, lora_decoy_ids = _score_model_directory(eval_dir, pair_ids)
            if lora_decoy_ids != reference_decoy_ids:
                raise RuntimeError(f"rank-{rank} seed-{seed} changed fixed decoy receptors")
            g_pair = {pair_id: lora_true[pair_id] - dtft_true[pair_id] for pair_id in pair_ids}
            lora_s_pair = {pair_id: lora_decoy[pair_id] - lora_true[pair_id] for pair_id in pair_ids}
            c_lora_pair = {pair_id: lora_s_pair[pair_id] - base_specificity_pair[pair_id] for pair_id in pair_ids}
            d_cond_pair = {pair_id: dtft_specificity_pair[pair_id] - lora_s_pair[pair_id] for pair_id in pair_ids}
            g_cluster = _cluster_means(g_pair, panel_rows)[1]
            lora_s_cluster = _cluster_means(lora_s_pair, panel_rows)[1]
            c_lora_cluster = _cluster_means(c_lora_pair, panel_rows)[1]
            d_cond_cluster = _cluster_means(d_cond_pair, panel_rows)[1]
            if _cluster_means(g_pair, panel_rows)[0] != common_cluster_ids:
                raise RuntimeError("LoRA and reference receptor cluster order differs")
            ci = _bootstrap_many({
                "G_LoRA": g_cluster,
                "C_LoRA": c_lora_cluster,
                "D_cond": d_cond_cluster,
            }, draws=draws, seed=71_000 + rank * 100 + seed)
            g_point = float(g_cluster.mean())
            f_point = g_point / GATE0_A_DTFT
            c_lora_point = float(c_lora_cluster.mean())
            d_point = float(d_cond_cluster.mean())
            q_point = c_lora_point / GATE0_C_DTFT
            task_class = _classify(ci["G_LoRA"], thresholds["delta_task"])
            cond_class = _classify(ci["D_cond"], thresholds["delta_cond"])
            rank_data[str(seed)] = {
                "rank": rank,
                "seed": seed,
                "true_pair_nll_equal_cluster": float(_cluster_means(lora_true, panel_rows)[1].mean()),
                "G_LoRA": g_point,
                "G_LoRA_95_cluster_bootstrap_CI": ci["G_LoRA"],
                "F_gap": f_point,
                "F_gap_95_cluster_bootstrap_CI": [ci["G_LoRA"][0] / GATE0_A_DTFT, ci["G_LoRA"][1] / GATE0_A_DTFT],
                "F_gap_in_original_borderline_region_0.08_to_0.10": 0.08 <= f_point < 0.10,
                "task_delta_material": thresholds["delta_task"],
                "task_classification": task_class,
                "C_LoRA": c_lora_point,
                "C_LoRA_95_cluster_bootstrap_CI": ci["C_LoRA"],
                "D_cond": d_point,
                "D_cond_95_cluster_bootstrap_CI": ci["D_cond"],
                "D_cond_over_C_DTFT": d_point / GATE0_C_DTFT,
                "Q_LoRA": q_point,
                "Q_LoRA_95_cluster_bootstrap_CI": [ci["C_LoRA"][0] / GATE0_C_DTFT, ci["C_LoRA"][1] / GATE0_C_DTFT],
                "conditional_delta_material": thresholds["delta_cond"],
                "conditional_classification": cond_class,
                "replication_trigger": {
                    "F_gap_at_least_0.08": f_point >= 0.08,
                    "task_CI_reaches_threshold": ci["G_LoRA"][0] <= thresholds["delta_task"] <= ci["G_LoRA"][1],
                    "D_cond_over_C_DTFT_at_least_0.08": d_point / GATE0_C_DTFT >= 0.08,
                    "conditional_CI_reaches_threshold": ci["D_cond"][0] <= thresholds["delta_cond"] <= ci["D_cond"][1],
                },
                "seed_checkpoint_sha256": json.loads((eval_dir / "evaluation_summary.json").read_text(encoding="utf-8"))["test_true"]["checkpoint_sha256"],
                "cluster_count": len(clusters),
                "bootstrap_draws": draws,
            }
        if not rank_data:
            continue
        rank_data["replication_triggered"] = any(rank_data["11"]["replication_trigger"].values())
        if "23" in rank_data:
            for endpoint, label_key in (("task", "task_classification"), ("conditioning", "conditional_classification")):
                first, second = rank_data["11"][label_key], rank_data["23"][label_key]
                if first == second == "meaningful deficit":
                    rank_data[f"{endpoint}_replication_status"] = "replicated meaningful deficit"
                elif first == second == "no materially meaningful deficit":
                    rank_data[f"{endpoint}_replication_status"] = "replicated no meaningful deficit"
                else:
                    rank_data[f"{endpoint}_replication_status"] = "UNRESOLVED / seed-sensitive"
        else:
            rank_data["task_replication_status"] = rank_data["11"]["task_classification"]
            rank_data["conditioning_replication_status"] = rank_data["11"]["conditional_classification"]
        outputs[str(rank)] = rank_data

    svd_comparison = {}
    for rank in (8, 64):
        true, decoy, decoy_ids = _score_model_directory(svd_root / f"svd_r{rank}", pair_ids)
        if decoy_ids != reference_decoy_ids:
            raise RuntimeError(f"SVD-r{rank} does not use the same corrected-panel decoys")
        a_rank = float((_cluster_means(base_true, panel_rows)[1] - _cluster_means(true, panel_rows)[1]).mean())
        s_rank_pair = {pair_id: decoy[pair_id] - true[pair_id] for pair_id in pair_ids}
        c_rank = float((_cluster_means(s_rank_pair, panel_rows)[1] - base_specificity_cluster).mean())
        svd_comparison[str(rank)] = {
            "task_gain_retention_R": a_rank / GATE0_A_DTFT,
            "conditional_signal_retention_Q": c_rank / GATE0_C_DTFT,
            "note": "descriptive post-hoc SVD only; not trained LoRA",
        }

    result = {
        "status": "primary_panel_analyzed",
        "panel": {
            "path": str(panel_path),
            "sha256": sha256_file(panel_path),
            "pair_count": len(pair_ids),
            "receptor_cluster_count": len(clusters),
            "clusters_are_independent_statistical_units": True,
        },
        "gate0_reference": {
            "A_DTFT": GATE0_A_DTFT,
            "C_DTFT": GATE0_C_DTFT,
            "delta_task": thresholds["delta_task"],
            "delta_cond": thresholds["delta_cond"],
            "A_recomputed_equal_cluster": a_point,
            "C_recomputed_equal_cluster": c_point,
        },
        "bootstrap_draws": draws,
        "rank_results": outputs,
        "same_rank_svd_descriptive_comparison": svd_comparison,
        "primary_estimand": "equal-receptor-cluster-weighted mean of within-cluster paired per-pair NLL differences",
        "precision_extension_used": False,
    }
    atomic_json(gate1_root / "primary_panel_analysis.json", result)
    return result


def write_final_summary(gate1_root: Path) -> dict:
    analysis_path = gate1_root / "primary_panel_analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    ranks = {}
    for rank in (8, 64):
        rank_result = analysis["rank_results"].get(str(rank))
        if not rank_result:
            continue
        task_status = rank_result["task_replication_status"]
        conditional_status = rank_result["conditioning_replication_status"]
        task_meaningful = task_status in {"meaningful deficit", "replicated meaningful deficit"}
        task_none = task_status in {"no materially meaningful deficit", "replicated no meaningful deficit"}
        conditional_meaningful = conditional_status in {"meaningful deficit", "replicated meaningful deficit"}
        conditional_none = conditional_status in {"no materially meaningful deficit", "replicated no meaningful deficit"}
        if not task_meaningful and not task_none or not conditional_meaningful and not conditional_none:
            pattern = "UNRESOLVED"
        elif task_meaningful and conditional_meaningful:
            pattern = "both generic task adaptation and receptor-conditioned adaptation"
        elif task_meaningful:
            pattern = "primarily generic task adaptation"
        elif conditional_meaningful:
            pattern = "primarily receptor-conditioned adaptation"
        else:
            pattern = "neither"
        convergence_by_seed = {}
        for seed in ("11", "23"):
            convergence_path = gate1_root / "evidence" / f"rank{rank}_seed{seed}" / "convergence_summary.json"
            if convergence_path.exists():
                convergence_by_seed[seed] = json.loads(convergence_path.read_text(encoding="utf-8"))
        hpo_path = gate1_root / "lr_selection" / f"rank{rank}" / "hpo_report.json"
        hpo = json.loads(hpo_path.read_text(encoding="utf-8")) if hpo_path.exists() else None
        ranks[str(rank)] = {
            "selected_learning_rate": hpo.get("selected_peak_learning_rate") if hpo else None,
            "hpo_report_sha256": sha256_file(hpo_path) if hpo_path.exists() else None,
            "training_by_seed": {
                seed: {
                    "best_epoch": convergence["best_epoch"],
                    "best_validation_peptide_nll": convergence["best_validation_peptide_nll"],
                    "terminal_validation_peptide_nll": convergence["terminal_validation_peptide_nll"],
                    "convergence_classification": convergence["convergence_classification"],
                    "total_wall_seconds": convergence["wall_seconds"],
                    "optimizer_steps": convergence["optimizer_steps"],
                    "selected_checkpoint_sha256": convergence["selected_checkpoint_sha256"],
                }
                for seed, convergence in convergence_by_seed.items()
            },
            "seed23_trained": "23" in rank_result,
            "task_status": task_status,
            "conditional_status": conditional_status,
            "deficit_pattern": pattern,
            "per_seed": {key: value for key, value in rank_result.items() if key in {"11", "23"}},
        }

    if "8" not in ranks or "64" not in ranks:
        overall = "incomplete"
        branch = "complete both rank-specific HPO, evidence training, and primary-panel evaluation"
    elif any("UNRESOLVED" in ranks[rank][key] for rank in ("8", "64") for key in ("task_status", "conditional_status")):
        overall = "UNRESOLVED"
        branch = "use only the predeclared unused-cluster precision extension for unresolved endpoints; no mechanistic follow-up yet"
    else:
        overall = "classified"
        if ranks["8"]["task_status"] in {"replicated meaningful deficit", "meaningful deficit"} and ranks["64"]["task_status"] in {"replicated no meaningful deficit", "no materially meaningful deficit"}:
            branch = "rank capacity becomes a plausible next hypothesis; do not infer it as proven"
        elif ranks["64"]["task_status"] == "replicated meaningful deficit":
            branch = "later examine subspace/structure/optimization diagnostics; no cause is established here"
        elif all(
            ranks[rank]["task_status"] in {"replicated no meaningful deficit", "no materially meaningful deficit"}
            and ranks[rank]["conditional_status"] in {"replicated no meaningful deficit", "no materially meaningful deficit"}
            for rank in ("8", "64")
        ):
            branch = "stop the LoRA-failure mechanism branch on this task"
        elif any(ranks[rank]["conditional_status"] in {"replicated meaningful deficit", "meaningful deficit"} for rank in ("8", "64")):
            branch = "report receptor-conditioned loss specifically; do not infer a mechanism"
        else:
            branch = "report observed endpoint pattern without causal claims; no H1-H4 follow-up in Gate 1"
    result = {
        "status": overall,
        "panel": analysis["panel"],
        "gate0_threshold_reference": analysis["gate0_reference"],
        "bootstrap_draws": analysis["bootstrap_draws"],
        "ranks": ranks,
        "same_rank_svd_descriptive_comparison": analysis["same_rank_svd_descriptive_comparison"],
        "precision_extension_used": analysis.get("precision_extension_used", False),
        "scientific_interpretation": "Gate 1 tests observed vanilla LoRA-versus-DTFT behavior only; no H1-H4 causal claim is made.",
        "next_mechanistic_branch": branch,
        "primary_panel_analysis_sha256": sha256_file(analysis_path),
    }
    atomic_json(gate1_root / "gate1_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Propedia26 Gate-1 vanilla LoRA experiments")
    subparsers = parser.add_subparsers(dest="action", required=True)
    prep = subparsers.add_parser("prepare-panel")
    prep.add_argument("--manifest-dir", type=Path, default=Path("data/protein/propedia26/manifests"))
    prep.add_argument("--output-dir", type=Path, default=Path("runs/protein/propedia26_gate1/inputs"))

    refcheck = subparsers.add_parser("verify-references")
    refcheck.add_argument("--panel", type=Path, default=Path("runs/protein/propedia26_gate1/inputs/test_corrected.jsonl"))
    refcheck.add_argument("--decoys", type=Path, default=Path("runs/protein/propedia26_gate1/inputs/test_corrected_decoys.jsonl"))
    refcheck.add_argument("--gate0-root", type=Path, default=Path("runs/protein/propedia26_gate0"))
    refcheck.add_argument("--output", type=Path, default=Path("runs/protein/propedia26_gate1/inputs/reference_asset_audit.json"))

    expansion = subparsers.add_parser("prepare-extension")
    expansion.add_argument("--primary-manifest-dir", type=Path, default=Path("data/protein/propedia26/manifests"))
    expansion.add_argument("--extension-manifest-dir", type=Path, default=Path("data/protein/propedia26/manifests_stage6"))
    expansion.add_argument("--output-dir", type=Path, default=Path("runs/protein/propedia26_gate1/inputs/precision_extension"))

    hpo_parser = subparsers.add_parser("hpo")
    hpo_parser.add_argument("--rank", type=int, choices=(8, 64), required=True)
    hpo_parser.add_argument("--manifest-dir", type=Path, default=Path("data/protein/propedia26/manifests"))
    hpo_parser.add_argument("--cache", type=Path, default=Path("data/protein/propedia26/token_cache.pt"))
    hpo_parser.add_argument("--base-summary", type=Path, default=Path("runs/protein/propedia26_gate0/base_evaluation_summary.json"))
    hpo_parser.add_argument("--output-dir", type=Path, default=Path("runs/protein/propedia26_gate1/lr_selection"))
    hpo_parser.add_argument("--run-root", type=Path, default=Path("runs/protein/propedia26_gate1"))

    evidence_parser = subparsers.add_parser("evidence")
    evidence_parser.add_argument("--rank", type=int, choices=(8, 64), required=True)
    evidence_parser.add_argument("--seed", type=int, choices=(11, 23), default=11)
    evidence_parser.add_argument("--manifest-dir", type=Path, default=Path("data/protein/propedia26/manifests"))
    evidence_parser.add_argument("--cache", type=Path, default=Path("data/protein/propedia26/token_cache.pt"))
    evidence_parser.add_argument("--hpo-report", type=Path, required=True)
    evidence_parser.add_argument("--output-dir", type=Path, required=True)
    evidence_parser.add_argument("--run-root", type=Path, default=Path("runs/protein/propedia26_gate1"))

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--rank", type=int, choices=(8, 64), required=True)
    eval_parser.add_argument("--seed", type=int, choices=(11, 23), required=True)
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--cache", type=Path, default=Path("data/protein/propedia26/token_cache.pt"))
    eval_parser.add_argument("--panel", type=Path, default=Path("runs/protein/propedia26_gate1/inputs/test_corrected.jsonl"))
    eval_parser.add_argument("--decoys", type=Path, default=Path("runs/protein/propedia26_gate1/inputs/test_corrected_decoys.jsonl"))
    eval_parser.add_argument("--output-dir", type=Path, required=True)

    analysis_parser = subparsers.add_parser("analyze")
    analysis_parser.add_argument("--panel", type=Path, default=Path("runs/protein/propedia26_gate1/inputs/test_corrected.jsonl"))
    analysis_parser.add_argument("--references", type=Path, default=Path("runs/protein/propedia26_gate1/inputs/gate0_references"))
    analysis_parser.add_argument("--svd-root", type=Path, default=Path("runs/protein/propedia26_gate1/inputs/gate0_references/svd_screen"))
    analysis_parser.add_argument("--run-root", type=Path, default=Path("runs/protein/propedia26_gate1"))
    analysis_parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)

    final_parser = subparsers.add_parser("finalize")
    final_parser.add_argument("--run-root", type=Path, default=Path("runs/protein/propedia26_gate1"))

    args = parser.parse_args()
    if args.action == "prepare-panel":
        print(json.dumps(prepare_corrected_panel(manifest_dir=args.manifest_dir, output_dir=args.output_dir), indent=2, sort_keys=True))
    elif args.action == "verify-references":
        print(json.dumps(verify_existing_reference_assets(
            panel_path=args.panel, decoys_path=args.decoys,
            gate0_root=args.gate0_root, output_path=args.output,
        ), indent=2, sort_keys=True))
    elif args.action == "prepare-extension":
        print(json.dumps(prepare_precision_extension(
            primary_manifest_dir=args.primary_manifest_dir,
            extension_manifest_dir=args.extension_manifest_dir,
            output_dir=args.output_dir,
        ), indent=2, sort_keys=True))
    elif args.action == "hpo":
        print(json.dumps(run_rank_hpo(
            rank=args.rank, manifest_dir=args.manifest_dir, cache_path=args.cache,
            base_summary_path=args.base_summary, output_dir=args.output_dir, run_root=args.run_root,
        ), indent=2, sort_keys=True))
    elif args.action == "evidence":
        print(json.dumps(run_evidence(
            rank=args.rank, seed=args.seed, manifest_dir=args.manifest_dir,
            cache_path=args.cache, hpo_report_path=args.hpo_report,
            output_dir=args.output_dir, run_root=args.run_root,
        ), indent=2, sort_keys=True))
    elif args.action == "evaluate":
        print(json.dumps(run_adapter_evaluation(
            rank=args.rank, seed=args.seed, checkpoint_path=args.checkpoint,
            cache_path=args.cache, panel_path=args.panel, decoys_path=args.decoys,
            output_dir=args.output_dir,
        ), indent=2, sort_keys=True))
    elif args.action == "analyze":
        print(json.dumps(analyze_primary_panel(
            panel_path=args.panel, gate0_reference_dir=args.references,
            svd_root=args.svd_root, gate1_root=args.run_root, draws=args.draws,
        ), indent=2, sort_keys=True))
    elif args.action == "finalize":
        print(json.dumps(write_final_summary(args.run_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
