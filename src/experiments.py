from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs"
EVIDENCE = ROOT / "gates"
DATASET = "uspto_mit_synthesis"
SEEDS = (533, 917)
LR_CANDIDATES = (5e-5, 1e-4, 2e-4)
TRAIN_AUGMENTATIONS = 5
VALIDATION_AUGMENTATIONS = 5
TRAIN_GROUPS = 256
VALIDATION_GROUPS = 32
FULL_TRAIN_ROWS = 2_195_175
PILOT_DIR = ROOT / "data" / "gate45_v2"
TRAIN_MANIFEST = PILOT_DIR / "uspto_mit_train.csv"
VALIDATION_MANIFEST = PILOT_DIR / "uspto_mit_validation.csv"
PILOT_METADATA = PILOT_DIR / "manifest.json"
SOURCE_TRAIN = ROOT / "data" / "uspto_mit_synthesis" / "train_r_smiles.csv"
SOURCE_VALIDATION = ROOT / "data" / "uspto_mit_synthesis" / "validation_r_smiles.csv"

# Twelve prespecified, stratified trials cover every retained k, lambda_eff,
# and loss-dropout level without evaluating the full 4 x 4 x 3 grid.
JEPA_CANDIDATES = (
    {"k": 0, "lambda_eff": 0.5, "dropout": 0.0},
    {"k": 0, "lambda_eff": 1.0, "dropout": 0.5},
    {"k": 0, "lambda_eff": 4.0, "dropout": 0.75},
    {"k": 1, "lambda_eff": 1.0, "dropout": 0.0},
    {"k": 1, "lambda_eff": 2.0, "dropout": 0.5},
    {"k": 1, "lambda_eff": 0.5, "dropout": 0.75},
    {"k": 2, "lambda_eff": 2.0, "dropout": 0.0},
    {"k": 2, "lambda_eff": 4.0, "dropout": 0.5},
    {"k": 2, "lambda_eff": 1.0, "dropout": 0.75},
    {"k": 3, "lambda_eff": 4.0, "dropout": 0.0},
    {"k": 3, "lambda_eff": 0.5, "dropout": 0.5},
    {"k": 3, "lambda_eff": 2.0, "dropout": 0.75},
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(molecule, isomericSmiles=True) if molecule else ""


def select_groups(path: Path, split: str, augmentations: int, group_count: int):
    selected = []
    candidate_count = group_count * 8
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        group = []
        group_index = 0
        for row in reader:
            group.append(row)
            if len(group) != augmentations:
                continue
            score = int.from_bytes(
                hashlib.sha256(
                    f"533|{split}|{group_index}".encode()
                ).digest(),
                "big",
            )
            entry = (-score, group_index, [dict(item) for item in group])
            if len(selected) < candidate_count:
                heapq.heappush(selected, entry)
            elif score < -selected[0][0]:
                heapq.heapreplace(selected, entry)
            group = []
            group_index += 1
        if group:
            raise ValueError(f"{path} row count is not divisible by {augmentations}")
    chosen = sorted(
        ((-negative, index, rows) for negative, index, rows in selected),
        key=lambda item: item[0],
    )
    output = []
    audit = {
        "candidate_groups_evaluated": 0,
        "invalid_target_groups_skipped": 0,
        "target_identity_mismatch_groups_skipped": 0,
    }
    for _, source_group, rows in chosen:
        audit["candidate_groups_evaluated"] += 1
        identities = [canonicalize(item["target"]) for item in rows]
        if any(not identity for identity in identities):
            audit["invalid_target_groups_skipped"] += 1
            continue
        if len(set(identities)) != 1:
            audit["target_identity_mismatch_groups_skipped"] += 1
            continue
        group_id = f"{split}-group-{source_group:09d}"
        for augmentation_index, row in enumerate(rows):
            row["group_id"] = group_id
            row["augmentation_index"] = str(augmentation_index)
            output.append(row)
        if len(output) == group_count * augmentations:
            break
    if len(output) != group_count * augmentations:
        raise RuntimeError(
            f"only {len(output) // augmentations} valid groups were available from "
            f"{candidate_count} deterministic candidates in {path}"
        )
    return output, audit


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_pilot_manifests():
    if PILOT_METADATA.exists():
        metadata = json.loads(PILOT_METADATA.read_text(encoding="utf-8"))
        for path, key in (
            (TRAIN_MANIFEST, "train_manifest_sha256"),
            (VALIDATION_MANIFEST, "validation_manifest_sha256"),
        ):
            if not path.exists() or file_sha256(path) != metadata[key]:
                raise RuntimeError(
                    f"immutable pilot manifest does not match {PILOT_METADATA}: {path}"
                )
        return metadata
    if TRAIN_MANIFEST.exists() or VALIDATION_MANIFEST.exists():
        raise RuntimeError(
            "pilot manifests exist without provenance metadata; do not overwrite them"
        )
    train_rows, train_audit = select_groups(
        SOURCE_TRAIN, "train", TRAIN_AUGMENTATIONS, TRAIN_GROUPS
    )
    validation_rows, validation_audit = select_groups(
        SOURCE_VALIDATION,
        "validation",
        VALIDATION_AUGMENTATIONS,
        VALIDATION_GROUPS,
    )
    write_csv(TRAIN_MANIFEST, train_rows)
    write_csv(VALIDATION_MANIFEST, validation_rows)
    metadata = {
        "schema_version": 1,
        "selection_seed": 533,
        "dataset": DATASET,
        "source_train": str(SOURCE_TRAIN.relative_to(ROOT)),
        "source_train_sha256": file_sha256(SOURCE_TRAIN),
        "source_validation": str(SOURCE_VALIDATION.relative_to(ROOT)),
        "source_validation_sha256": file_sha256(SOURCE_VALIDATION),
        "train_groups": TRAIN_GROUPS,
        "train_augmentations_per_group": TRAIN_AUGMENTATIONS,
        "train_rows": len(train_rows),
        "validation_groups": VALIDATION_GROUPS,
        "validation_augmentations_per_group": VALIDATION_AUGMENTATIONS,
        "validation_rows": len(validation_rows),
        "train_selection_audit": train_audit,
        "validation_selection_audit": validation_audit,
        "selected_target_identity_overlap_count": len(
            {canonicalize(row["target"]) for row in train_rows}
            & {canonicalize(row["target"]) for row in validation_rows}
        ),
        "selected_reaction_pair_overlap_count": len(
            {
                (canonicalize(row["source"]), canonicalize(row["target"]))
                for row in train_rows
            }
            & {
                (canonicalize(row["source"]), canonicalize(row["target"]))
                for row in validation_rows
            }
        ),
        "train_manifest_sha256": file_sha256(TRAIN_MANIFEST),
        "validation_manifest_sha256": file_sha256(VALIDATION_MANIFEST),
    }
    PILOT_METADATA.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def config_slug(config) -> str:
    return (
        f"k{config['k']}-l{config['lambda_eff']:g}-d{config['dropout']:g}"
    )


def latest_checkpoint(path: Path, *, max_epoch: int | None = None) -> Path | None:
    checkpoints = []
    for child in path.glob("epoch_*"):
        if (child / "training_state.pt").exists():
            epoch = int(child.name.split("_")[-1])
            if max_epoch is None or epoch <= max_epoch:
                checkpoints.append((epoch, child))
    return max(checkpoints, default=(0, None))[1]


def invoke(
    gate, condition, seed, learning_rate, epochs, destination, checkpoint_dir,
    *, config=None, max_train_rows=None, resume=False,
    stop_after_epoch=None, continue_from_checkpoint=False,
):
    if destination.exists():
        if resume:
            return json.loads(destination.read_text(encoding="utf-8"))
        raise FileExistsError(
            f"refusing to overwrite completed run {destination}; pass --resume to reuse it"
        )
    command = [
        sys.executable,
        str(ROOT / "src" / "train.py"),
        "--gate", str(gate),
        "--dataset", DATASET,
        "--condition", condition,
        "--seed", str(seed),
        "--learning-rate", str(learning_rate),
        "--epochs", str(epochs),
        "--batch-size", "8",
        "--gradient-accumulation-steps", "1",
        "--gradient-checkpointing",
        "--fused-adamw",
        "--pin-memory",
        "--data-fraction", str(
            (TRAIN_GROUPS * TRAIN_AUGMENTATIONS) / FULL_TRAIN_ROWS
        ),
        "--train-manifest", str(TRAIN_MANIFEST),
        "--validation-manifest", str(VALIDATION_MANIFEST),
        "--checkpoint-dir", str(checkpoint_dir),
        "--output", str(destination),
    ]
    if stop_after_epoch is not None:
        command += ["--stop-after-epoch", str(stop_after_epoch)]
    if config is not None:
        command += [
            "--k", str(config["k"]),
            "--lambda-eff", str(config["lambda_eff"]),
            "--dropout", str(config["dropout"]),
        ]
    if max_train_rows is not None:
        command += ["--max-train-rows", str(max_train_rows)]
    if resume or continue_from_checkpoint:
        checkpoint = latest_checkpoint(
            checkpoint_dir, max_epoch=stop_after_epoch
        )
        if checkpoint is not None:
            command += ["--resume-from", str(checkpoint)]
            # Checkpoints produced before rung-resume support did not persist
            # elapsed wall time. Preserve an approximate cumulative value from
            # the checkpoint directory lifetime; optimizer steps/FLOPs remain exact.
            command += [
                "--prior-wall-time-seconds",
                str(max(
                    0.0,
                    (checkpoint / "training_state.pt").stat().st_mtime
                    - checkpoint_dir.stat().st_ctime,
                )),
            ]
    print("RUN", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env=os.environ.copy())
    return json.loads(destination.read_text(encoding="utf-8"))


def trial_score(result, validity_floor: float):
    metrics = result["validation_metrics"]
    return (
        int(metrics["valid_rate"] >= validity_floor),
        metrics["exact_top1"],
    )


def compact_result(result):
    return {
        "condition": result["condition"],
        "seed": result["seed"],
        "config": result["config"],
        "selected_epoch": result["selected_epoch"],
        "validation_native_loss": result["validation_native_loss"],
        "validation_metrics": result["validation_metrics"],
        "diagnostics": result["diagnostics"],
        "compute": result["compute"],
    }


def gate4(*, resume: bool):
    metadata = prepare_pilot_manifests()
    root = OUTPUT / "gate4_v2"
    native_results = []
    for learning_rate in LR_CANDIDATES:
        destination = root / "trials" / f"native-lr{learning_rate:g}.json"
        checkpoint_dir = root / "checkpoints" / f"native-lr{learning_rate:g}"
        native_results.append(invoke(
            4, "native", SEEDS[0], learning_rate, 4,
            destination, checkpoint_dir, resume=resume, stop_after_epoch=2,
        ))
    # Native LR is frozen before any JEPA trial. Validity is a hard guardrail,
    # and the plan's frozen generative selector is the only ranking metric.
    # Python's stable candidate order resolves exact ties without consulting a
    # forbidden auxiliary validation signal.
    native_validity_floor = max(
        result["validation_metrics"]["valid_rate"] for result in native_results
    )
    selected_native = max(
        native_results,
        key=lambda result: trial_score(result, native_validity_floor),
    )
    frozen_lr = selected_native["config"]["learning_rate"]

    survivors = list(JEPA_CANDIDATES)
    stage_results = {}
    selected_seed533 = None
    # Aggressive ASHA-style halving: all 12 candidates receive one epoch,
    # then only 4 receive epoch 2 and only 2 receive epochs 3-4. Promotions
    # resume the same fixed-four-epoch schedule instead of retraining prior rungs.
    for budget, keep in ((1, 4), (2, 2), (4, 1)):
        promoted = []
        for config in survivors:
            slug = config_slug(config)
            destination = root / "trials" / f"jepa-e{budget}-{slug}.json"
            checkpoint_dir = root / "checkpoints" / f"jepa-{slug}"
            result = invoke(
                4, "clm_jepa", SEEDS[0], frozen_lr, 4,
                destination, checkpoint_dir, config=config, resume=resume,
                stop_after_epoch=budget,
                continue_from_checkpoint=budget > 1,
            )
            stage_results[(budget, slug)] = result
            promoted.append((trial_score(result, native_validity_floor), config, result))
        promoted.sort(key=lambda item: item[0], reverse=True)
        survivors = [item[1] for item in promoted[:keep]]
        selected_seed533 = promoted[0][2]
    selected = survivors[0]
    replication = invoke(
        4, "clm_jepa", SEEDS[1], frozen_lr, 4,
        root / "trials" / f"replicate-s{SEEDS[1]}-{config_slug(selected)}.json",
        root / "checkpoints" / f"replicate-s{SEEDS[1]}-{config_slug(selected)}",
        config=selected, resume=resume,
    )
    replication_valid = (
        selected_seed533["validation_metrics"]["valid_rate"] >= native_validity_floor
        and replication["validation_metrics"]["valid_rate"] >= native_validity_floor
        and math.isfinite(selected_seed533["validation_native_loss"])
        and math.isfinite(replication["validation_native_loss"])
    )
    frozen = {
        "gate": 4,
        "decision": "PASS" if replication_valid else "FAIL",
        "scope": (
            "immutable reduced USPTO-MIT R-SMILES pilot; hyperparameter selection only"
        ),
        "manifest": metadata,
        "native_learning_rate": frozen_lr,
        "native_validity_floor": native_validity_floor,
        "jepa": selected,
        "seeds": list(SEEDS),
        "gate3_k_candidates_retained_by_plan": [0, 1, 2, 3],
        "unique_jepa_trials": len(JEPA_CANDIDATES),
        "successive_halving_budgets": [1, 2, 4],
        "successive_halving_survivors": [12, 4, 2, 1],
        "native_lr_tuning_budget_epochs": 2,
        "jepa_epoch_equivalents": 20,
        "selected_native": compact_result(selected_native),
        "selected_seed_533": compact_result(selected_seed533),
        "second_seed_replication": compact_result(replication),
        "stop_condition": (
            None if replication_valid
            else "selected JEPA configuration failed the frozen native validity guardrail or produced non-finite validation loss"
        ),
    }
    frozen_path = root / "frozen_config.json"
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    evidence_path = EVIDENCE / "gate4" / "results.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    return frozen


def gate5(*, resume: bool):
    frozen_path = OUTPUT / "gate4_v2" / "frozen_config.json"
    if not frozen_path.exists():
        raise FileNotFoundError("Gate 5 requires a completed Gate 4 frozen configuration")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen["decision"] != "PASS":
        raise RuntimeError("Gate 5 is blocked because Gate 4 did not pass")
    prepare_pilot_manifests()
    root = OUTPUT / "gate5"
    results = []
    for seed in SEEDS:
        for condition in ("native", "monitor", "clm_jepa", "shuffled"):
            config = None if condition == "native" else frozen["jepa"]
            result = invoke(
                5, condition, seed, frozen["native_learning_rate"], 4,
                root / "runs" / f"{condition}-s{seed}.json",
                root / "checkpoints" / f"{condition}-s{seed}",
                config=config, resume=resume,
            )
            results.append(result)
    jepa_only = invoke(
        5, "jepa_only", SEEDS[0], frozen["native_learning_rate"], 4,
        root / "runs" / f"jepa_only-s{SEEDS[0]}.json",
        root / "checkpoints" / f"jepa_only-s{SEEDS[0]}",
        config=frozen["jepa"], max_train_rows=30, resume=resume,
    )

    comparisons = []
    passed = True
    for seed in SEEDS:
        rows = {
            result["condition"]: result
            for result in results if result["seed"] == seed
        }
        native = rows["native"]
        jepa = rows["clm_jepa"]
        shuffled = rows["shuffled"]
        controls = [rows[name] for name in ("native", "monitor", "clm_jepa", "shuffled")]
        optimizer_steps = {row["compute"]["optimizer_steps"] for row in controls}
        manifest_hashes = {
            (
                row["config"]["train_manifest_sha256"],
                row["config"]["validation_manifest_sha256"],
            )
            for row in controls
        }
        effective_batches = {
            (
                row["config"]["physical_batch_size"],
                row["config"]["gradient_accumulation_steps"],
                row["config"]["effective_batch_size"],
            )
            for row in controls
        }
        fairness_ok = (
            len(optimizer_steps) == 1
            and len(manifest_hashes) == 1
            and len(effective_batches) == 1
        )
        directional = (
            jepa["validation_metrics"]["exact_top1"]
            > native["validation_metrics"]["exact_top1"]
            and jepa["validation_metrics"]["exact_top1"]
            > shuffled["validation_metrics"]["exact_top1"]
            and jepa["validation_metrics"]["valid_rate"]
            >= native["validation_metrics"]["valid_rate"]
            and jepa["validation_native_loss"]
            <= native["validation_native_loss"]
        )
        seed_passes = fairness_ok and directional
        passed &= seed_passes
        comparisons.append({
            "seed": seed,
            "passes": seed_passes,
            "directional": directional,
            "fairness_ok": fairness_ok,
            "native": compact_result(native),
            "monitor": compact_result(rows["monitor"]),
            "clm_jepa": compact_result(jepa),
            "shuffled": compact_result(shuffled),
        })
    summary = {
        "gate": 5,
        "decision": "PASS" if passed else "FAIL",
        "scope": "USPTO-MIT pilot control confirmation on the frozen Gate 4 configuration",
        "frozen_gate4_config": {
            "learning_rate": frozen["native_learning_rate"],
            **frozen["jepa"],
        },
        "control_conditions": ["native", "monitor", "clm_jepa", "shuffled"],
        "seeds": list(SEEDS),
        "comparisons": comparisons,
        "jepa_only_failure_diagnostic": compact_result(jepa_only),
        "stop_condition": (
            None if passed else
            "cLM-JEPA did not beat native and matched-shuffled JEPA on both seeds without validity/native-loss regression and matched exposure"
        ),
    }
    summary_path = root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    evidence_path = EVIDENCE / "gate5" / "results.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=int, choices=(4, 5), required=True)
    parser.add_argument(
        "--resume", action="store_true",
        help="reuse completed outputs and resume the latest incomplete epoch checkpoint",
    )
    args = parser.parse_args()
    if os.environ.get("WANDB_MODE") is None:
        os.environ["WANDB_MODE"] = "offline"
    result = gate4(resume=args.resume) if args.gate == 4 else gate5(resume=args.resume)
    print(json.dumps(result, indent=2))
    if result["decision"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
