from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from .analysis import combined_distance_trend, coverage_check, distance_trend, gate_quantities, nll_delta
from .config import DENSE_HPO_LRS, EVIDENCE_SEED, HPO_SEED, LORA_HPO_LRS, REPLICATION_SEED, TrainConfig
from .evaluate import evaluate_many
from .hpo import select_learning_rate
from .training import train


EVALUATION_SETS = ("standard_test", "remote_ood_test", "natural_test")
HPO_STEPS = 40
EVIDENCE_EPOCH_CAP = 4


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_training(
    root: Path, manifests: Path, load: str, mode: str, seed: int, lr: float,
    backend: str, *, hpo: bool = False, compile_model: bool = False,
    microbatch_tokens: int = 4096, num_workers: int = 4,
    gradient_checkpointing: bool = True,
) -> Path:
    section = "hpo" if hpo else "evidence"
    suffix = f"lr_{lr:g}" if hpo else f"seed_{seed}"
    output = root / section / load / mode / suffix
    best = output / "best.pt"
    complete = output / "execution.json"
    if complete.exists() and (hpo or best.exists()):
        return best
    config = TrainConfig(
        mode=mode,
        seed=seed,
        learning_rate=lr,
        train_manifest=manifests / f"train_{load}.jsonl",
        validation_manifest=manifests / "validation.jsonl",
        output_dir=output,
        attention_backend=backend,
        maximum_epochs=EVIDENCE_EPOCH_CAP,
        maximum_optimizer_steps=HPO_STEPS if hpo else None,
        # At 10k, ~64 optimizer steps make one epoch.  A 100-step interval
        # produced too few checks for patience-six early stopping to operate
        # within the 10-epoch cap; 50 keeps validation inexpensive and useful.
        validation_every_steps=20 if hpo else 50,
        early_stopping_checks=3,
        save_checkpoints=not hpo,
        compile_model=compile_model,
        microbatch_tokens=microbatch_tokens,
        num_workers=num_workers,
        gradient_checkpointing=gradient_checkpointing,
    )
    resume = output / "last.pt"
    if resume.exists():
        existing_config_path = output / "config.json"
        if not existing_config_path.exists():
            raise RuntimeError(f"checkpoint exists without config: {resume}")
        existing_config = json.loads(existing_config_path.read_text(encoding="utf-8"))
        if existing_config != config.serializable():
            raise RuntimeError(f"refusing to resume {resume} under a different configuration")
    train(config, resume=resume if resume.exists() else None)
    _release()
    return best


def _evaluate_method(
    root: Path, manifests: Path, load: str, mode: str, seed: int | None,
    checkpoint: Path | None, backend: str, token_budget: int, num_workers: int,
    compile_model: bool,
) -> None:
    leaf = f"seed_{seed}" if seed is not None else "fixed"
    pending = {
        dataset: root / "evaluation" / load / mode / leaf / dataset
        for dataset in EVALUATION_SETS
        if not (root / "evaluation" / load / mode / leaf / dataset / "summary.json").exists()
    }
    if pending:
        evaluate_many(
            {dataset: manifests / f"{dataset}.jsonl" for dataset in pending}, pending,
            mode=mode, checkpoint=checkpoint, realizations=3,
            token_budget=token_budget, num_workers=num_workers,
            attention_backend=backend, compile_model=compile_model,
        )
        _release()


def _protein_file(root: Path, load: str, mode: str, seed: int | None, dataset: str) -> Path:
    leaf = f"seed_{seed}" if seed is not None else "fixed"
    return root / "evaluation" / load / mode / leaf / dataset / "protein_nll.jsonl"


def _hpo(
    root: Path, manifests: Path, backend: str, compile_model: bool,
    microbatch_tokens: int, num_workers: int, gradient_checkpointing: bool,
) -> dict[str, float]:
    selected = {}
    # One representative per optimizer family is sufficient for a compact
    # reliability screen: FT/DTFT share dense AdamW dynamics, while both LoRA
    # ranks use the same vanilla parameterization and scaling.
    representatives = {"dtft": ("ft", "dtft"), "lora8": ("lora8", "lora64")}
    for mode, family in representatives.items():
        candidates = DENSE_HPO_LRS if mode in {"ft", "dtft"} else LORA_HPO_LRS
        histories = {}
        for learning_rate in candidates:
            _run_training(
                root, manifests, "10k", mode, HPO_SEED, learning_rate, backend,
                hpo=True, compile_model=compile_model,
                microbatch_tokens=microbatch_tokens, num_workers=num_workers,
                gradient_checkpointing=gradient_checkpointing,
            )
            path = root / "hpo" / "10k" / mode / f"lr_{learning_rate:g}" / "history.json"
            histories[learning_rate] = json.loads(path.read_text(encoding="utf-8"))
        selection = select_learning_rate(histories)
        directory = root / "hpo" / "10k" / mode
        (directory / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8")
        for member in family:
            member_directory = root / "hpo" / "10k" / member
            member_directory.mkdir(parents=True, exist_ok=True)
            member_selection = {**selection, "representative_method": mode, "applies_to": list(family)}
            (member_directory / "selection.json").write_text(
                json.dumps(member_selection, indent=2, sort_keys=True), encoding="utf-8"
            )
            selected[member] = float(selection["selected_learning_rate"])
    return selected


def run(
    root: Path, manifests: Path, backend: str, compile_model: bool = False,
    microbatch_tokens: int = 4096, num_workers: int = 4,
) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    profile_path = root / "execution_profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.exists() else {}
    backend = str(profile.get("backend", backend))
    selected_training = profile.get("selected_training", {})
    selected_evaluation = profile.get("selected_evaluation", {})
    gradient_checkpointing = bool(selected_training.get("gradient_checkpointing", True))
    compile_model = bool(selected_training.get("compile_model", compile_model))
    evaluation_token_budget = int(selected_evaluation.get("token_budget", 4096))
    evaluation_workers = int(selected_evaluation.get("workers", 2))
    evaluation_compile = bool(selected_evaluation.get("compile_model", False))
    selected_lrs = _hpo(
        root, manifests, backend, compile_model, microbatch_tokens, num_workers,
        gradient_checkpointing,
    )
    # Base is load-independent, but mirrored under 10k for uniform gate paths.
    _evaluate_method(
        root, manifests, "10k", "base", None, None, backend,
        evaluation_token_budget, evaluation_workers,
        evaluation_compile,
    )
    for mode in ("ft", "dtft", "lora8", "lora64"):
        checkpoint = _run_training(
            root, manifests, "10k", mode, EVIDENCE_SEED, selected_lrs[mode], backend,
            compile_model=compile_model, microbatch_tokens=microbatch_tokens,
            num_workers=num_workers,
            gradient_checkpointing=gradient_checkpointing,
        )
        _evaluate_method(
            root, manifests, "10k", mode, EVIDENCE_SEED, checkpoint, backend,
            evaluation_token_budget, evaluation_workers,
            evaluation_compile,
        )

    coverage = {}
    for dataset in ("standard_test", "remote_ood_test"):
        coverage[dataset] = coverage_check(
            _protein_file(root, "10k", "base", None, dataset),
            _protein_file(root, "10k", "ft", EVIDENCE_SEED, dataset),
            _protein_file(root, "10k", "dtft", EVIDENCE_SEED, dataset),
        )
    run_ft_50k = any(
        value["fraction_of_base_to_ft_gain"] is not None
        and value["fraction_of_base_to_ft_gain"] > 0.10
        for value in coverage.values()
    )
    # Reuse the single Base evaluation without recomputation.
    for dataset in EVALUATION_SETS:
        source = _protein_file(root, "10k", "base", None, dataset)
        target_dir = root / "evaluation" / "50k" / "base" / "fixed" / dataset
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "protein_nll.jsonl"
        if not target.exists():
            target.hardlink_to(source)
        summary_source = source.parent / "summary.json"
        summary_target = target_dir / "summary.json"
        if not summary_target.exists():
            summary_target.hardlink_to(summary_source)
    modes50 = ["dtft", "lora8", "lora64"] + (["ft"] if run_ft_50k else [])
    for mode in modes50:
        checkpoint = _run_training(
            root, manifests, "50k", mode, EVIDENCE_SEED, selected_lrs[mode], backend,
            compile_model=compile_model, microbatch_tokens=microbatch_tokens,
            num_workers=num_workers,
            gradient_checkpointing=gradient_checkpointing,
        )
        _evaluate_method(
            root, manifests, "50k", mode, EVIDENCE_SEED, checkpoint, backend,
            evaluation_token_budget, evaluation_workers,
            evaluation_compile,
        )

    gates = {}
    combined_distance_trends = {}
    candidate_pairs = set()
    for load in ("10k", "50k"):
        gates[load] = {}
        combined_distance_trends[load] = {}
        for rank in ("lora8", "lora64"):
            gates[load][rank] = {}
            for dataset in ("standard_test", "remote_ood_test"):
                result = gate_quantities(
                    _protein_file(root, load, "base", None, dataset),
                    _protein_file(root, load, "dtft", EVIDENCE_SEED, dataset),
                    _protein_file(root, load, rank, EVIDENCE_SEED, dataset),
                )
                gates[load][rank][dataset] = result
                result["distance_trend"] = distance_trend(
                    _protein_file(root, load, "dtft", EVIDENCE_SEED, dataset),
                    _protein_file(root, load, rank, EVIDENCE_SEED, dataset),
                    load,
                )
                if result["replication_trigger"]:
                    candidate_pairs.add((load, rank))
            combined_distance_trends[load][rank] = combined_distance_trend(
                [
                    _protein_file(root, load, "dtft", EVIDENCE_SEED, dataset)
                    for dataset in ("standard_test", "remote_ood_test")
                ],
                [
                    _protein_file(root, load, rank, EVIDENCE_SEED, dataset)
                    for dataset in ("standard_test", "remote_ood_test")
                ],
                load,
            )

    replicated = {}
    coverage_confound = run_ft_50k
    # A seed-23 LoRA replication cannot resolve a dense-target coverage confound.
    # Stop that branch here rather than spending another training seed prematurely.
    replication_loads = set() if coverage_confound else {item[0] for item in candidate_pairs}
    for load in sorted(replication_loads):
        dtft_checkpoint = _run_training(
            root, manifests, load, "dtft", REPLICATION_SEED, selected_lrs["dtft"], backend,
            compile_model=compile_model, microbatch_tokens=microbatch_tokens,
            num_workers=num_workers,
            gradient_checkpointing=gradient_checkpointing,
        )
        _evaluate_method(
            root, manifests, load, "dtft", REPLICATION_SEED, dtft_checkpoint, backend,
            evaluation_token_budget, evaluation_workers,
            evaluation_compile,
        )
        for rank in sorted(item[1] for item in candidate_pairs if item[0] == load):
            checkpoint = _run_training(
                root, manifests, load, rank, REPLICATION_SEED, selected_lrs[rank], backend,
                compile_model=compile_model, microbatch_tokens=microbatch_tokens,
                num_workers=num_workers,
                gradient_checkpointing=gradient_checkpointing,
            )
            _evaluate_method(
                root, manifests, load, rank, REPLICATION_SEED, checkpoint, backend,
                evaluation_token_budget, evaluation_workers,
                evaluation_compile,
            )
            replicated[f"{load}/{rank}"] = {
                dataset: gate_quantities(
                    _protein_file(root, load, "base", None, dataset),
                    _protein_file(root, load, "dtft", REPLICATION_SEED, dataset),
                    _protein_file(root, load, rank, REPLICATION_SEED, dataset),
                )
                for dataset in ("standard_test", "remote_ood_test")
            }
    replicated_gap = any(
        replicated.get(f"{load}/{rank}", {}).get(dataset, {}).get("candidate_gap", False)
        for load, rank in candidate_pairs
        for dataset, first in gates[load][rank].items()
        if first["replication_trigger"]
    )
    load_dependence = []
    rank_sensitivity = []
    r64_material_gaps = []
    for dataset in ("standard_test", "remote_ood_test"):
        for rank in ("lora8", "lora64"):
            small = gates["10k"][rank][dataset]["F_gap"]
            moderate = gates["50k"][rank][dataset]["F_gap"]
            load_dependence.append({
                "rank": rank,
                "dataset": dataset,
                "F_gap_10k": small,
                "F_gap_50k": moderate,
                "delta_50k_minus_10k": (
                    moderate - small if small is not None and moderate is not None else None
                ),
                "gap_grew": bool(
                    small is not None and moderate is not None and moderate > small
                ),
            })
        if (
            gates["10k"]["lora8"][dataset]["candidate_gap"]
            and not gates["10k"]["lora64"][dataset]["candidate_gap"]
        ):
            rank_sensitivity.append({"load": "10k", "dataset": dataset})
        if (
            gates["50k"]["lora8"][dataset]["candidate_gap"]
            and not gates["50k"]["lora64"][dataset]["candidate_gap"]
        ):
            rank_sensitivity.append({"load": "50k", "dataset": dataset})
        for load in ("10k", "50k"):
            if gates[load]["lora64"][dataset]["candidate_gap"]:
                r64_material_gaps.append({"load": load, "dataset": dataset})
    if coverage_confound:
        decision = "FT materially exceeds DTFT: target coverage remains a confound"
    elif not candidate_pairs or not replicated_gap:
        decision = "No meaningful LoRA gap through 50k"
    else:
        decision = "Replicated DTFT-to-LoRA gap survives and FT approximately equals DTFT"
    retention = {}
    for load, modes in (("10k", ("ft", "dtft", "lora8", "lora64")), ("50k", tuple(modes50))):
        retention[load] = {
            mode: nll_delta(
                _protein_file(root, load, "base", None, "natural_test"),
                _protein_file(root, load, mode, EVIDENCE_SEED, "natural_test"),
            )
            for mode in modes
        }
    summary = {
        "selected_learning_rates": selected_lrs,
        "attention_backend": backend,
        "torch_compile": compile_model,
        "microbatch_tokens": microbatch_tokens,
        "num_workers": num_workers,
        "gradient_checkpointing": gradient_checkpointing,
        "evaluation_token_budget": evaluation_token_budget,
        "evaluation_workers": evaluation_workers,
        "evaluation_torch_compile": evaluation_compile,
        "coverage_10k": coverage,
        "ft_50k_triggered": run_ft_50k,
        "seed11_gates": gates,
        "seed11_combined_distance_trends": combined_distance_trends,
        "candidate_load_rank_pairs": sorted([list(item) for item in candidate_pairs]),
        "seed23_replication": replicated,
        "screen_patterns": {
            "load_dependence": load_dependence,
            "rank_sensitivity": rank_sensitivity,
            "r64_material_gaps": r64_material_gaps,
        },
        "natural_retention": retention,
        "gate1_decision": decision,
    }
    (root / "gate1_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--attention-backend", required=True)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--microbatch-tokens", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    run(
        args.output_root, args.manifest_root, args.attention_backend, args.compile_model,
        args.microbatch_tokens, args.num_workers,
    )


if __name__ == "__main__":
    main()
