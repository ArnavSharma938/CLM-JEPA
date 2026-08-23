"""Frozen audit of pair-center contraction and cross-batch NTP utility.

No optimizer is constructed and no parameter is updated. Auxiliary gradients
are measured on four fixed training batches; their NTP effect is measured
against four disjoint held-out validation batches.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F
from transformers import set_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_chemfm_mechanism import (  # noqa: E402
    add_vectors,
    device_batch,
    disable_stochastic_behavior,
    parameter_fingerprint,
    vector_relation,
)
from audit_sigreg_pair_specificity import (  # noqa: E402
    accumulate_autograd_result,
    empty_vector,
    fixed_batches,
    ntp_gradient,
    sigreg_state_gradient,
    vector_dot,
    vector_norm,
    verify_endpoint_fast_path,
)
from chemfm import MODEL_DIR, TOKENIZER_DIR, ReactionCollator, load_lora_model, load_reaction_tokenizer  # noqa: E402
from jepa import (  # noqa: E402
    CLMJEPA,
    add_predictor_tokens,
    matched_derangement,
)
from historical_pcsf import (  # noqa: E402
    pair_center_standard_deviation,
    pair_centers,
    pcsf_loss,
)
from train import (  # noqa: E402
    file_sha256,
    load_adapter_checkpoint,
    read_rows,
    reaction_row_fingerprint,
    validate_serialization_endings,
)


SEED = 533
BATCH_SIZE = 16
PHYSICAL_BATCH = 2
BATCHES = 4
SIGREG_DRAWS = 2
OUTER = 2.0
SIGREG_RELATIVE = 4.0 * 0.01 / 0.99
PCSF_BETA = 4.2
PCSF_RHO = 0.8
PCSF_EPSILON = 1e-8

TRAIN_MANIFEST = ROOT / "data" / "clm_jepa_uspto_mit_pilot_1280" / "uspto_mit_train.csv"
VALIDATION_PANEL = ROOT / "data" / "clm_jepa_uspto_mit_validation_256" / "uspto_mit_validation_length_stratified_256.csv"
REFERENCE_CACHE = ROOT / "runs" / "pcsf" / "reference" / "native_epoch4_pair_centers.pt"
DEFAULT_OUTPUT = ROOT / "runs" / "diagnostics" / "contraction_ntp_directional_audit" / "audit.json"

CHECKPOINTS = {
    **{
        f"native_e{epoch}": ROOT / "runs" / "sigreg_batch16_pilot" / "matched_b4" / "native_checkpoints" / f"epoch_{epoch}"
        for epoch in (1, 2, 4)
    },
    **{
        f"mse_e{epoch}": ROOT / "runs" / "mse_ablation" / "stage1" / "mse_checkpoints" / f"epoch_{epoch}"
        for epoch in (1, 2)
    },
    **{
        f"pcsf_e{epoch}": ROOT / "runs" / "pcsf" / "training" / "checkpoints" / f"epoch_{epoch}"
        for epoch in (1, 2, 4)
    },
    **{
        f"mse_sigreg_e{epoch}": ROOT / "runs" / "mse_ablation" / "stage1" / "mse_sigreg_checkpoints" / f"epoch_{epoch}"
        for epoch in (1, 2, 4)
    },
}


def named_trainable(model) -> list[tuple[str, torch.nn.Parameter]]:
    result = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    if not result:
        raise RuntimeError("model has no trainable parameters")
    return result


def selected(vector: Mapping[str, torch.Tensor], names: set[str]) -> dict[str, torch.Tensor]:
    return {name: value for name, value in vector.items() if name in names}


def numeric_summary(values: Iterable[float | None]) -> dict[str, float] | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return {
        "n": len(finite),
        "mean": statistics.fmean(finite),
        "sample_sd": statistics.stdev(finite) if len(finite) > 1 else 0.0,
        "min": min(finite),
        "median": statistics.median(finite),
        "max": max(finite),
    }


def build_validation_batches(rows, collator, count: int, physical_batch: int):
    permutation = torch.randperm(
        len(rows), generator=torch.Generator().manual_seed(SEED + 17011)
    ).tolist()
    selected_indices = permutation[: count * BATCH_SIZE]
    batches = []
    for batch_index in range(count):
        indices = selected_indices[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE]
        selected_rows = [rows[index] for index in indices]
        batches.append({
            "indices": indices,
            "rows": selected_rows,
            "chunks": [
                collator(selected_rows[start:start + physical_batch])
                for start in range(0, BATCH_SIZE, physical_batch)
            ],
        })
    return selected_indices, batches


def state_objective_gradients(
    source: torch.Tensor,
    target: torch.Tensor,
    reference_centers: torch.Tensor,
    permutation: list[int],
    draw_seeds: list[int],
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    source_leaf = source.detach().clone().requires_grad_(True)
    target_leaf = target.detach().clone().requires_grad_(True)
    centers = pair_centers(source_leaf, target_leaf)
    sigma = pair_center_standard_deviation(centers, epsilon=PCSF_EPSILON)
    mse = F.mse_loss(source_leaf, target_leaf)
    shuffled_mse = F.mse_loss(source_leaf, target_leaf[permutation])
    pcsf, _, reference_sigma = pcsf_loss(
        source_leaf, target_leaf, reference_centers.to(source_leaf.device),
        rho=PCSF_RHO, epsilon=PCSF_EPSILON,
    )
    objectives = {
        "spread": sigma,
        "mse": mse,
        "mse_shuffled": shuffled_mse,
        "pcsf": pcsf,
    }
    gradients: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, value in objectives.items():
        gradients[name] = torch.autograd.grad(
            value, (source_leaf, target_leaf), retain_graph=True,
        )

    sigreg_values = []
    sigreg_gradients = []
    for draw_index, seed in enumerate(draw_seeds):
        value, gradient, direction_hash = sigreg_state_gradient(source, target, seed=seed)
        sigreg_gradients.append(gradient)
        sigreg_values.append({
            "draw": draw_index, "seed": seed, "loss": value,
            "direction_hash": direction_hash,
        })
    gradients["sigreg"] = tuple(
        torch.stack([gradient[side] for gradient in sigreg_gradients]).mean(0)
        for side in (0, 1)
    )
    return gradients, {
        "pair_center_sigma": float(sigma.detach()),
        "reference_pair_center_sigma": float(reference_sigma.detach()),
        "pcsf_active": bool(float(pcsf.detach()) > 0.0),
        "mse_loss": float(mse.detach()),
        "shuffled_mse_loss": float(shuffled_mse.detach()),
        "pcsf_loss": float(pcsf.detach()),
        "sigreg_draws": sigreg_values,
    }


def exact_endpoint_forward(method, model, raw):
    """Use the maintained 3B-row endpoint path while omitting unused logits."""
    return method(
        model, device_batch(raw, model.device), k=0, jepa_weight=0.0,
        native_weight=1.0, monitor_only=True, stop_gradient_target=False,
        jepa_loss_type="mse", sigreg_tradeoff=0.0, jepa_ratio=1.0,
        force_jepa_active=True, endpoint_only=True,
    )


def collect_states(method, model, chunks) -> tuple[torch.Tensor, torch.Tensor]:
    sources, targets = [], []
    with torch.no_grad():
        for raw in chunks:
            output = exact_endpoint_forward(method, model, raw)
            sources.append(output.source_states.detach().float())
            targets.append(output.target_states.detach().float())
    return torch.cat(sources), torch.cat(targets)


def exact_endpoint_vjps(model, method, chunks, endpoint_gradients, named_parameters):
    vectors = {name: empty_vector(named_parameters) for name in endpoint_gradients}
    parameters = tuple(parameter for _, parameter in named_parameters)
    offset = 0
    for raw in chunks:
        size = int(raw["input_ids"].size(0))
        output = exact_endpoint_forward(method, model, raw)
        names = list(endpoint_gradients)
        for index, name in enumerate(names):
            source_gradient, target_gradient = endpoint_gradients[name]
            gradients = torch.autograd.grad(
                (output.source_states, output.target_states), parameters,
                grad_outputs=(
                    source_gradient[offset:offset + size].to(output.source_states.dtype),
                    target_gradient[offset:offset + size].to(output.target_states.dtype),
                ),
                retain_graph=index < len(names) - 1, allow_unused=True,
            )
            accumulate_autograd_result(vectors[name], named_parameters, gradients)
        offset += size
        del output
    return vectors


def effect(metric: Mapping[str, torch.Tensor], objective: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    dot = vector_dot(metric, objective)
    metric_norm = vector_norm(metric)
    objective_norm = vector_norm(objective)
    denominator = metric_norm * objective_norm
    return {
        "gradient_dot": dot,
        "descent_change_per_unit_learning_rate": -dot,
        "descent_change_per_unit_objective_norm": -dot / objective_norm if objective_norm else None,
        "cosine_with_metric_gradient": dot / denominator if denominator else None,
        "objective_norm": objective_norm,
    }


def evaluate_scopes(
    vectors: Mapping[str, Mapping[str, torch.Tensor]],
    heldout_ntp: Mapping[str, torch.Tensor],
    scope_names: Mapping[str, set[str]],
) -> dict[str, Any]:
    result = {}
    for scope, names in scope_names.items():
        scoped = {label: selected(vector, names) for label, vector in vectors.items()}
        eval_ntp = selected(heldout_ntp, names)
        spread = scoped["spread"]
        objectives = {
            "ntp_train": scoped["ntp_train"],
            "mse_raw": scoped["mse"],
            "mse_active_weighted": add_vectors((scoped["mse"], OUTER)),
            "pair_specific_mse_residual_raw": add_vectors(
                (scoped["mse"], 1.0), (scoped["mse_shuffled"], -1.0)
            ),
            "pair_specific_mse_residual_active_weighted": add_vectors(
                (scoped["mse"], OUTER), (scoped["mse_shuffled"], -OUTER)
            ),
            "pcsf_raw": scoped["pcsf"],
            "pcsf_active_weighted": add_vectors((scoped["pcsf"], OUTER * PCSF_BETA)),
            "sigreg_raw": scoped["sigreg"],
            "sigreg_active_weighted": add_vectors((scoped["sigreg"], OUTER * SIGREG_RELATIVE)),
            "mse_only_full_active": add_vectors((scoped["mse"], OUTER)),
            "pcsf_full_active": add_vectors(
                (scoped["mse"], OUTER), (scoped["pcsf"], OUTER * PCSF_BETA)
            ),
            "mse_sigreg_full_active": add_vectors(
                (scoped["mse"], OUTER), (scoped["sigreg"], OUTER * SIGREG_RELATIVE)
            ),
        }
        result[scope] = {
            "spread_direction": {
                name: effect(spread, vector) for name, vector in objectives.items()
            },
            "heldout_ntp_direction": {
                name: effect(eval_ntp, vector) for name, vector in objectives.items()
            },
            "train_vs_heldout_ntp": vector_relation(scoped["ntp_train"], eval_ntp),
            "heldout_ntp_loss_gradient_norm": vector_norm(eval_ntp),
        }
    return result


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "batches": len(records),
        "pair_center_sigma": numeric_summary(row["state"]["pair_center_sigma"] for row in records),
        "reference_pair_center_sigma": numeric_summary(row["state"]["reference_pair_center_sigma"] for row in records),
        "pcsf_active_fraction": statistics.fmean(float(row["state"]["pcsf_active"]) for row in records),
        "scopes": {},
    }
    for scope in records[0]["scopes"]:
        spread_names = records[0]["scopes"][scope]["spread_direction"]
        ntp_names = records[0]["scopes"][scope]["heldout_ntp_direction"]
        result["scopes"][scope] = {
            "spread_direction": {
                name: {
                    key: numeric_summary(
                        row["scopes"][scope]["spread_direction"][name][key]
                        for row in records
                    )
                    for key in (
                        "descent_change_per_unit_learning_rate",
                        "descent_change_per_unit_objective_norm",
                        "cosine_with_metric_gradient",
                        "objective_norm",
                    )
                }
                for name in spread_names
            },
            "heldout_ntp_direction": {
                name: {
                    key: numeric_summary(
                        row["scopes"][scope]["heldout_ntp_direction"][name][key]
                        for row in records
                    )
                    for key in (
                        "descent_change_per_unit_learning_rate",
                        "descent_change_per_unit_objective_norm",
                        "cosine_with_metric_gradient",
                        "objective_norm",
                    )
                }
                for name in ntp_names
            },
        }
    return result


def add_time_matched_ratios(results: dict[str, Any]) -> None:
    for label, value in results.items():
        epoch = int(label.rsplit("e", 1)[1])
        native_records = results[f"native_e{epoch}"]["batches"]
        ratios = []
        for row, native_row in zip(value["batches"], native_records):
            ratio = (
                row["state"]["pair_center_sigma"]
                / native_row["state"]["pair_center_sigma"]
            )
            row["time_matched_native_sigma_ratio"] = ratio
            ratios.append(ratio)
        value["aggregate"]["time_matched_native_sigma_ratio"] = numeric_summary(ratios)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batches", type=int, default=BATCHES)
    parser.add_argument("--physical-batch", type=int, default=PHYSICAL_BATCH)
    parser.add_argument("--sigreg-draws", type=int, default=SIGREG_DRAWS)
    parser.add_argument("--only", action="append", choices=sorted(CHECKPOINTS))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is required")
    if BATCH_SIZE % args.physical_batch:
        raise ValueError("physical batch must divide 16")
    if not 1 <= args.batches <= BATCHES or not 1 <= args.sigreg_draws <= 4:
        raise ValueError("unsupported batch/draw count")

    checkpoints = CHECKPOINTS if not args.only else {label: CHECKPOINTS[label] for label in args.only}
    required_native = {
        f"native_e{int(label.rsplit('e', 1)[1])}" for label in checkpoints
    }
    checkpoints = {label: path for label, path in CHECKPOINTS.items() if label in checkpoints or label in required_native}
    for path in (*checkpoints.values(), TRAIN_MANIFEST, VALIDATION_PANEL, REFERENCE_CACHE):
        if not path.exists():
            raise FileNotFoundError(path)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    set_seed(SEED)
    training_rows = read_rows("uspto_mit_synthesis", path=TRAIN_MANIFEST)
    validation_rows = read_rows("uspto_mit_synthesis", path=VALIDATION_PANEL)
    reference = torch.load(REFERENCE_CACHE, map_location="cpu", weights_only=False)
    expected_fingerprints = [reaction_row_fingerprint(row) for row in training_rows]
    if reference["row_fingerprints"] != expected_fingerprints:
        raise ValueError("PCSF reference cache does not match the training manifest")

    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    original_vocab = len(tokenizer)
    predictor_ids = add_predictor_tokens(tokenizer)
    collator = ReactionCollator(tokenizer, task="forward")
    validate_serialization_endings(collator, training_rows, tokenizer.eos_token_id)
    train_indices, train_batches = fixed_batches(
        training_rows, collator, args.batches, args.physical_batch
    )
    validation_indices, validation_batches = build_validation_batches(
        validation_rows, collator, args.batches, args.physical_batch
    )

    model = load_lora_model(
        MODEL_DIR, tokenizer, attention_dropout=0.0,
        chemfm_vocab_size=original_vocab,
        attn_implementation="sdpa",
    ).cuda()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.train()
    controls = disable_stochastic_behavior(model)
    method = CLMJEPA(
        predictor_ids, tokenizer.eos_token_id, tokenizer.pad_token_id,
        sigreg_seed=SEED,
    )
    all_parameters = named_trainable(model)
    parameters = [(name, value) for name, value in all_parameters if ".lora_" in name]
    lora_names = {name for name, _ in parameters}
    if not parameters:
        raise RuntimeError("model has no trainable LoRA A/B parameters")
    scopes = {"lora_ab": lora_names}
    fast_path_equivalence = None
    started = time.perf_counter()
    results: dict[str, Any] = {}
    no_update: dict[str, Any] = {}

    for checkpoint_index, (label, checkpoint) in enumerate(checkpoints.items()):
        load_adapter_checkpoint(model, checkpoint)
        before = parameter_fingerprint(model)
        if fast_path_equivalence is None:
            # The source/target-only 2B-row shortcut is rejected if it changes
            # BF16 kernels materially. This audit uses the maintained 3B-row
            # path with only the unused vocabulary projection omitted.
            fast_path_equivalence = verify_endpoint_fast_path(
                method, model, train_batches[0]["chunks"][0],
                [(name, parameter) for name, parameter in parameters if name in lora_names],
            )
        records = []
        for batch_index, (train_batch, validation_batch) in enumerate(zip(train_batches, validation_batches)):
            batch_started = time.perf_counter()
            source, target = collect_states(method, model, train_batch["chunks"])
            refs = reference["pair_centers"][train_batch["indices"]].float().to(source.device)
            state_gradients, state_values = state_objective_gradients(
                source, target, refs, train_batch["derangement"],
                [104729 + 1009 * batch_index + draw for draw in range(args.sigreg_draws)],
            )
            vectors = exact_endpoint_vjps(
                model, method, train_batch["chunks"], state_gradients, parameters
            )
            train_ntp_loss, vectors["ntp_train"], train_tokens = ntp_gradient(
                model, train_batch["chunks"], parameters
            )
            eval_ntp_loss, heldout_ntp, eval_tokens = ntp_gradient(
                model, validation_batch["chunks"], parameters
            )
            records.append({
                "batch_index": batch_index,
                "training_manifest_indices_zero_based": train_batch["indices"],
                "validation_panel_indices_zero_based": validation_batch["indices"],
                "derangement": train_batch["derangement"],
                "state": state_values,
                "train_ntp_loss": train_ntp_loss,
                "heldout_ntp_loss": eval_ntp_loss,
                "train_target_tokens": train_tokens,
                "heldout_target_tokens": eval_tokens,
                "scopes": evaluate_scopes(vectors, heldout_ntp, scopes),
                "runtime_seconds": time.perf_counter() - batch_started,
            })
            print(json.dumps({
                "checkpoint": label, "checkpoint_index": checkpoint_index + 1,
                "checkpoints": len(checkpoints), "batch": batch_index + 1,
                "batches": args.batches, "seconds": records[-1]["runtime_seconds"],
            }), flush=True)
            del source, target, vectors, heldout_ntp
            torch.cuda.empty_cache()
        after = parameter_fingerprint(model)
        no_update[label] = {"before": before, "after": after, "unchanged": before == after}
        if before != after:
            raise RuntimeError(f"parameters changed during frozen audit: {label}")
        results[label] = {"batches": records, "aggregate": aggregate(records)}

    add_time_matched_ratios(results)
    output = {
        "scope": "frozen pair-center contraction and disjoint-batch NTP directional audit; no training, optimizer, generation, or parameter updates",
        "configuration": {
            "seed": SEED,
            "training_batches": args.batches,
            "heldout_batches": args.batches,
            "examples_per_batch": BATCH_SIZE,
            "physical_batch": args.physical_batch,
            "readout": "k=0 final source EOS and final target EOS",
            "sigreg_draws_averaged_per_measurement": args.sigreg_draws,
            "active_objectives": {
                "mse_only": "2*MSE",
                "pcsf": "2*(MSE + 4.2*PCSF)",
                "mse_sigreg": "2*(MSE + 0.0404040404*SIGReg)",
            },
            "expected_dropout_objective": "0.5 times each active auxiliary vector; direction and cosine unchanged",
            "pcsf": {"rho": PCSF_RHO, "beta": PCSF_BETA, "epsilon": PCSF_EPSILON},
            "gradient_scopes": {name: len(names) for name, names in scopes.items()},
            "excluded_from_primary_gradient_scope": "large modules_to_save token-I/O tensors; established audit convention, avoiding host-memory paging while measuring every LoRA A/B update",
            "optimizer_constructed": False,
            "gradient_checkpointing": "non-reentrant; required for exact multi-objective VJPs on the 6 GB RTX 4050",
            "stochasticity": "model/dropout disabled; SIGReg fresh draws explicitly seeded and averaged",
            "endpoint_fast_path_equivalence": fast_path_equivalence,
            "endpoint_execution": "maintained native+source+target padded row geometry with endpoint_only backbone path; 2B-row shortcut measured but not used",
            **controls,
        },
        "sign_conventions": {
            "spread": "v=-grad(sigma_PC).dot(g); positive increases spread under infinitesimal gradient descent",
            "ntp": "Delta L_eval=-grad(L_eval).dot(g); negative improves held-out NTP, positive worsens it",
        },
        "provenance": {
            "train_manifest": str(TRAIN_MANIFEST.resolve()),
            "train_manifest_sha256": file_sha256(TRAIN_MANIFEST),
            "validation_panel": str(VALIDATION_PANEL.resolve()),
            "validation_panel_sha256": file_sha256(VALIDATION_PANEL),
            "pcsf_reference_cache": str(REFERENCE_CACHE.resolve()),
            "pcsf_reference_cache_sha256": file_sha256(REFERENCE_CACHE),
            "training_indices_zero_based": train_indices,
            "validation_indices_zero_based": validation_indices,
            "checkpoints": {label: str(path.resolve()) for label, path in checkpoints.items()},
        },
        "no_parameter_update_validation": no_update,
        "results": results,
        "runtime_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "runtime_seconds": output["runtime_seconds"],
        "states": list(results),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
