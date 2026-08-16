"""Frozen temporal audit of SIGReg pair specificity in ChemFM cLM-JEPA.

The script constructs no optimizer and performs no parameter update.  It uses
four disjoint batches of 16 fixed USPTO-MIT reactions and exact endpoint-state
VJPs to measure LoRA-parameter gradients under four independently resampled
SIGReg slice sets per batch/checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import set_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_chemfm_mechanism import (  # noqa: E402
    add_vectors,
    device_batch,
    disable_stochastic_behavior,
    geometry,
    parameter_fingerprint,
    vector_difference_ratio,
    vector_relation,
)
from chemfm import (  # noqa: E402
    IGNORE_INDEX,
    MODEL_DIR,
    TOKENIZER_DIR,
    ReactionCollator,
    load_lora_model,
    load_reaction_tokenizer,
)
from jepa import CLMJEPA, SIGReg, add_predictor_tokens, extract_source_and_target, matched_derangement  # noqa: E402
from train import load_adapter_checkpoint, read_rows, validate_serialization_endings  # noqa: E402


SEED = 533
BATCH_SIZE = 16
PHYSICAL_BATCH = 4
BATCHES = 4
SIGREG_DRAWS = 4
SHUFFLE_SEED = 1907
DRAW_SEED = 104729
OUTER_COEFFICIENT = 2.0
SIGREG_RELATIVE_COEFFICIENT = 4.0 * 0.01 / 0.99
APPLIED_SIGREG_COEFFICIENT = OUTER_COEFFICIENT * SIGREG_RELATIVE_COEFFICIENT

TRAIN_MANIFEST = ROOT / "data" / "clm_jepa_uspto_mit_pilot_1280" / "uspto_mit_train.csv"
CHECKPOINT_ROOT = ROOT / "runs" / "mse_ablation" / "stage1" / "mse_sigreg_checkpoints"
NATIVE_CHECKPOINT = ROOT / "runs" / "sigreg_batch16_pilot" / "matched_b4" / "native_checkpoints" / "epoch_4"
DEFAULT_OUTPUT = ROOT / "runs" / "diagnostics" / "sigreg_pair_specificity_audit" / "audit.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lora_parameters(model) -> list[tuple[str, torch.nn.Parameter]]:
    selected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ".lora_" in name
    ]
    if not selected:
        raise RuntimeError("no trainable LoRA parameters found")
    return selected


def empty_vector(parameters: Iterable[tuple[str, torch.nn.Parameter]]) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros(parameter.shape, dtype=torch.float32)
        for name, parameter in parameters
    }


def accumulate_autograd_result(
    accumulator: dict[str, torch.Tensor],
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    gradients: tuple[torch.Tensor | None, ...],
) -> None:
    for (name, _), gradient in zip(named_parameters, gradients):
        if gradient is not None:
            accumulator[name].add_(gradient.detach().cpu().float())


def vector_norm(vector: Mapping[str, torch.Tensor]) -> float:
    return math.sqrt(sum(float(value.square().sum()) for value in vector.values()))


def vector_dot(first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]) -> float:
    return sum(
        float(torch.dot(first[name].flatten(), second[name].flatten()))
        for name in set(first) & set(second)
    )


def relation(first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    return vector_relation(first, second)


def gradient_descent_effect(
    metric_gradient: Mapping[str, torch.Tensor],
    objective_gradient: Mapping[str, torch.Tensor],
) -> dict[str, float | None]:
    """First-order metric change for theta <- theta - eta*g, per eta=1."""
    dot = vector_dot(metric_gradient, objective_gradient)
    objective_norm = vector_norm(objective_gradient)
    metric_norm = vector_norm(metric_gradient)
    denominator = objective_norm * metric_norm
    return {
        "gradient_dot": dot,
        "descent_effect_per_unit_learning_rate": -dot,
        "descent_effect_per_unit_objective_gradient_norm": (
            -dot / objective_norm if objective_norm else None
        ),
        "cosine_with_metric_gradient": dot / denominator if denominator else None,
    }


def reference_endpoint_forward(method: CLMJEPA, model, raw: Mapping[str, Any]):
    return method(
        model,
        device_batch(raw, model.device),
        k=0,
        jepa_weight=0.0,
        native_weight=1.0,
        monitor_only=True,
        stop_gradient_target=False,
        jepa_loss_type="mse",
        sigreg_tradeoff=0.0,
        jepa_ratio=1.0,
        force_jepa_active=True,
    )


def endpoint_forward(method: CLMJEPA, model, raw: Mapping[str, Any]):
    """Mathematically equivalent k=0 endpoint path without unused computation.

    CLMJEPA evaluates independent native/source/target rows in one padded batch.
    Causal-transformer rows do not interact, so the independent native row and
    vocabulary projection cannot affect source/target endpoints or their exact
    mathematical gradients.  Omitting them changes only BF16 kernel batching;
    ``verify_endpoint_fast_path`` records the resulting numerical tolerance
    against the maintained path before the audit proceeds.
    ``verify_endpoint_fast_path`` checks states and LoRA gradients against the
    maintained reference path before the audit proceeds.
    """
    sources, targets = extract_source_and_target(raw)
    rows = sources + targets
    padded = pad_sequence(rows, batch_first=True, padding_value=method.pad_token_id)
    attention = padded.ne(method.pad_token_id)
    padded = padded.to(model.device)
    attention = attention.to(model.device)
    backbone = model.get_base_model().model
    output = backbone(
        input_ids=padded,
        attention_mask=attention,
        use_cache=False,
        return_dict=True,
    )
    hidden = output.last_hidden_state
    batch_size = len(sources)
    indices = attention.sum(1) - 1
    row_indices = torch.arange(batch_size, device=hidden.device)

    class EndpointOutput:
        pass

    result = EndpointOutput()
    result.source_states = hidden[row_indices, indices[:batch_size]]
    result.target_states = hidden[row_indices + batch_size, indices[batch_size:]]
    return result


def verify_endpoint_fast_path(method, model, raw, named_lora) -> dict[str, float | bool]:
    parameters = tuple(parameter for _, parameter in named_lora)
    reference = reference_endpoint_forward(method, model, raw)
    reference_loss = F.mse_loss(reference.source_states, reference.target_states)
    reference_gradients = torch.autograd.grad(reference_loss, parameters, allow_unused=True)
    reference_vector = empty_vector(named_lora)
    accumulate_autograd_result(reference_vector, named_lora, reference_gradients)

    fast = endpoint_forward(method, model, raw)
    fast_loss = F.mse_loss(fast.source_states, fast.target_states)
    fast_gradients = torch.autograd.grad(fast_loss, parameters, allow_unused=True)
    fast_vector = empty_vector(named_lora)
    accumulate_autograd_result(fast_vector, named_lora, fast_gradients)
    state_difference = torch.cat((
        (reference.source_states - fast.source_states).detach().float().flatten(),
        (reference.target_states - fast.target_states).detach().float().flatten(),
    ))
    gradient_relation = relation(fast_vector, reference_vector)
    gradient_relative_error = vector_norm(
        add_vectors((fast_vector, 1.0), (reference_vector, -1.0))
    ) / max(vector_norm(reference_vector), 1e-30)
    result = {
        "reference_mse": float(reference_loss.detach()),
        "fast_mse": float(fast_loss.detach()),
        "mse_absolute_error": abs(float(reference_loss.detach()) - float(fast_loss.detach())),
        "state_max_absolute_error": float(state_difference.abs().max()),
        "state_mean_absolute_error": float(state_difference.abs().mean()),
        "lora_gradient_cosine": gradient_relation["cosine_similarity"],
        "lora_gradient_relative_l2_error": gradient_relative_error,
    }
    result["passed"] = (
        result["state_max_absolute_error"] <= 4e-2
        and result["mse_absolute_error"] <= 5e-5
        and result["lora_gradient_cosine"] is not None
        and result["lora_gradient_cosine"] >= 0.999
        and result["lora_gradient_relative_l2_error"] <= 0.04
    )
    del reference, fast
    return result


def collect_states(method, model, chunks) -> tuple[torch.Tensor, torch.Tensor]:
    source, target = [], []
    with torch.no_grad():
        for raw in chunks:
            output = endpoint_forward(method, model, raw)
            source.append(output.source_states.detach().float())
            target.append(output.target_states.detach().float())
    return torch.cat(source), torch.cat(target)


def sigreg_state_gradient(
    source: torch.Tensor, target: torch.Tensor, *, seed: int
) -> tuple[float, tuple[torch.Tensor, torch.Tensor], str]:
    source_leaf = source.detach().clone().requires_grad_(True)
    target_leaf = target.detach().clone().requires_grad_(True)
    regularizer = SIGReg(seed=seed)
    loss = regularizer(torch.stack((source_leaf, target_leaf)))
    source_gradient, target_gradient = torch.autograd.grad(
        loss, (source_leaf, target_leaf)
    )
    # Reconstruct the exact local direction draw for provenance.
    generator = torch.Generator(device=source.device).manual_seed(seed)
    directions = torch.randn(
        source.size(-1), regularizer.num_slices,
        generator=generator, device=source.device,
    )
    directions = directions / directions.norm(dim=0, keepdim=True)
    direction_hash = hashlib.sha256(
        directions[:, :8].detach().cpu().float().numpy().tobytes()
    ).hexdigest()
    return float(loss.detach()), (source_gradient, target_gradient), direction_hash


def endpoint_metrics(
    source: torch.Tensor, target: torch.Tensor, permutation: list[int]
) -> tuple[dict[str, float], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    source_leaf = source.detach().clone().requires_grad_(True)
    target_leaf = target.detach().clone().requires_grad_(True)
    shuffled = target_leaf[permutation]
    true_mse = F.mse_loss(source_leaf, target_leaf)
    shuffled_mse = F.mse_loss(source_leaf, shuffled)
    discrimination = shuffled_mse - true_mse
    centers = (source_leaf + target_leaf) / 2.0
    center_separation = F.mse_loss(centers, centers[permutation])
    joint_variance = 0.5 * (
        source_leaf.var(0, unbiased=False).mean()
        + target_leaf.var(0, unbiased=False).mean()
    )
    cosine_margin = (
        F.cosine_similarity(source_leaf, target_leaf, dim=-1)
        - F.cosine_similarity(source_leaf, shuffled, dim=-1)
    ).mean()
    objectives = {
        "true_pair_distance": true_mse,
        "shuffled_pair_distance": shuffled_mse,
        "center_separation": center_separation,
        "joint_variance": joint_variance,
        "cosine_retrieval_margin": cosine_margin,
    }
    gradients = {}
    for index, (name, value) in enumerate(objectives.items()):
        gradients[name] = torch.autograd.grad(
            value,
            (source_leaf, target_leaf),
            retain_graph=index < len(objectives) - 1,
        )
    with torch.no_grad():
        squared_distances = torch.cdist(source.float(), target.float()).square() / source.size(-1)
        cosine_scores = F.normalize(source.float(), dim=-1) @ F.normalize(target.float(), dim=-1).T
        frozen = {
            "true_pair_mse": float(true_mse.detach()),
            "shuffled_pair_mse": float(shuffled_mse.detach()),
            "squared_distance_pair_margin": float(discrimination.detach()),
            "center_separation": float(center_separation.detach()),
            "joint_variance": float(joint_variance.detach()),
            "cosine_pair_margin": float(cosine_margin.detach()),
            "euclidean_retrieval_top1": float(
                squared_distances.argmin(1).eq(torch.arange(len(source), device=source.device)).float().mean()
            ),
            "cosine_retrieval_top1": float(
                cosine_scores.argmax(1).eq(torch.arange(len(source), device=source.device)).float().mean()
            ),
        }
    return frozen, gradients


def endpoint_vjps(
    model,
    method,
    chunks,
    endpoint_gradients: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    named_lora: list[tuple[str, torch.nn.Parameter]],
) -> dict[str, dict[str, torch.Tensor]]:
    vectors = {name: empty_vector(named_lora) for name in endpoint_gradients}
    parameters = tuple(parameter for _, parameter in named_lora)
    offset = 0
    for raw in chunks:
        size = int(raw["input_ids"].size(0))
        output = endpoint_forward(method, model, raw)
        names = list(endpoint_gradients)
        for index, name in enumerate(names):
            source_gradient, target_gradient = endpoint_gradients[name]
            gradients = torch.autograd.grad(
                (output.source_states, output.target_states),
                parameters,
                grad_outputs=(
                    source_gradient[offset:offset + size].to(output.source_states.dtype),
                    target_gradient[offset:offset + size].to(output.target_states.dtype),
                ),
                retain_graph=index < len(names) - 1,
                allow_unused=True,
            )
            accumulate_autograd_result(vectors[name], named_lora, gradients)
        offset += size
        del output
    return vectors


def ntp_gradient(model, chunks, named_lora) -> tuple[float, dict[str, torch.Tensor], int]:
    total_tokens = sum(int(raw["labels"][:, 1:].ne(IGNORE_INDEX).sum()) for raw in chunks)
    nll_total = 0.0
    vector = empty_vector(named_lora)
    parameters = tuple(parameter for _, parameter in named_lora)
    for raw in chunks:
        batch = device_batch(raw, model.device)
        output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = output.logits[:, :-1].float()
        labels = batch["labels"][:, 1:]
        active = labels.ne(IGNORE_INDEX)
        nll = F.cross_entropy(logits[active], labels[active], reduction="sum")
        nll_total += float(nll.detach())
        gradients = torch.autograd.grad(nll / total_tokens, parameters, allow_unused=True)
        accumulate_autograd_result(vector, named_lora, gradients)
        del output
    return nll_total / total_tokens, vector, total_tokens


def frozen_geometry(source: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    centers = (source + target) / 2.0
    return {
        "source": geometry(source),
        "target": geometry(target),
        "pair_centers": geometry(centers),
    }


def summarize_vector(
    vector: Mapping[str, torch.Tensor], ntp: Mapping[str, torch.Tensor],
    pair_discrimination: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    return {
        "norm": vector_norm(vector),
        "relative_to_ntp": relation(vector, ntp)["relative_norm_auxiliary_over_native"],
        "relation_to_ntp": relation(vector, ntp),
        "relation_to_pair_discrimination_gradient": relation(vector, pair_discrimination),
    }


def audit_batch(label, model, method, chunks, permutation, draw_seeds, named_lora):
    started = time.perf_counter()
    ntp_loss, ntp, ntp_tokens = ntp_gradient(model, chunks, named_lora)
    source, target = collect_states(method, model, chunks)
    metric_values, endpoint_metric_gradients = endpoint_metrics(source, target, permutation)
    sigreg_values = []
    endpoint_gradients = dict(endpoint_metric_gradients)
    for draw, seed in enumerate(draw_seeds):
        value, state_gradient, direction_hash = sigreg_state_gradient(source, target, seed=seed)
        endpoint_gradients[f"sigreg_draw_{draw}"] = state_gradient
        sigreg_values.append({"draw": draw, "seed": seed, "loss": value, "direction_hash": direction_hash})
    vectors = endpoint_vjps(model, method, chunks, endpoint_gradients, named_lora)
    true_mse = vectors["true_pair_distance"]
    shuffled_mse = vectors["shuffled_pair_distance"]
    pair_discrimination = add_vectors((shuffled_mse, 1.0), (true_mse, -1.0))
    pair_residual = add_vectors((true_mse, 1.0), (shuffled_mse, -1.0))
    metric_gradient_names = {
        "pair_discrimination": pair_discrimination,
        "true_pair_distance": true_mse,
        "center_separation": vectors["center_separation"],
        "joint_variance": vectors["joint_variance"],
        "cosine_retrieval_margin": vectors["cosine_retrieval_margin"],
    }
    base_objectives = {
        "ntp": ntp,
        "mse_raw": true_mse,
        "mse_active_weighted": add_vectors((true_mse, OUTER_COEFFICIENT)),
        "mse_shuffled_raw": shuffled_mse,
        "pair_specific_mse_residual": pair_residual,
        "pair_discrimination_gradient": pair_discrimination,
    }
    base_summaries = {
        name: summarize_vector(vector, ntp, pair_discrimination)
        for name, vector in base_objectives.items()
    }
    base_summaries["mse_true_vs_shuffled"] = {
        "relation": relation(true_mse, shuffled_mse),
        "difference_over_true": vector_difference_ratio(true_mse, shuffled_mse),
        "pair_residual_over_ntp": vector_norm(pair_residual) / max(vector_norm(ntp), 1e-30),
        "identity_check_g_pair_equals_negative_g_M_cosine": relation(pair_residual, pair_discrimination)["cosine_similarity"],
    }
    draw_results = []
    for draw_info in sigreg_values:
        sigreg = vectors[f"sigreg_draw_{draw_info['draw']}"]
        weighted_sigreg = add_vectors((sigreg, APPLIED_SIGREG_COEFFICIENT))
        full_true = add_vectors((true_mse, OUTER_COEFFICIENT), (sigreg, APPLIED_SIGREG_COEFFICIENT))
        full_shuffle = add_vectors((shuffled_mse, OUTER_COEFFICIENT), (sigreg, APPLIED_SIGREG_COEFFICIENT))
        objectives = {
            "sigreg_raw": sigreg,
            "sigreg_active_weighted": weighted_sigreg,
            "full_active_true": full_true,
            "full_active_shuffled": full_shuffle,
        }
        effects = {
            metric: {
                objective: gradient_descent_effect(metric_gradient, objective_gradient)
                for objective, objective_gradient in {
                    "ntp": ntp,
                    "mse_raw": true_mse,
                    "mse_active_weighted": base_objectives["mse_active_weighted"],
                    **objectives,
                }.items()
            }
            for metric, metric_gradient in metric_gradient_names.items()
        }
        draw_results.append({
            **draw_info,
            "objectives": {
                name: summarize_vector(vector, ntp, pair_discrimination)
                for name, vector in objectives.items()
            },
            "full_true_vs_shuffled": {
                "relation": relation(full_true, full_shuffle),
                "difference_over_true": vector_difference_ratio(full_true, full_shuffle),
                "pair_specific_residual_norm": vector_norm(add_vectors((full_true, 1.0), (full_shuffle, -1.0))),
                "pair_specific_fraction_of_full_true": vector_difference_ratio(full_true, full_shuffle),
            },
            "first_order_descent_effects": effects,
        })
    return {
        "checkpoint": label,
        "ntp_loss": ntp_loss,
        "ntp_target_tokens": ntp_tokens,
        "geometry": frozen_geometry(source, target),
        "frozen_pair_metrics": metric_values,
        "base_objectives": base_summaries,
        "draws": draw_results,
        "runtime_seconds": time.perf_counter() - started,
        "peak_allocated_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
    }


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


def aggregate_checkpoint(batches: list[dict[str, Any]]) -> dict[str, Any]:
    draws = [draw for batch in batches for draw in batch["draws"]]
    return {
        "batches": len(batches),
        "sigreg_draws": len(draws),
        "ntp_loss": numeric_summary(batch["ntp_loss"] for batch in batches),
        "source_variance": numeric_summary(batch["geometry"]["source"]["variance"] for batch in batches),
        "target_variance": numeric_summary(batch["geometry"]["target"]["variance"] for batch in batches),
        "source_effective_rank": numeric_summary(batch["geometry"]["source"]["effective_rank"] for batch in batches),
        "target_effective_rank": numeric_summary(batch["geometry"]["target"]["effective_rank"] for batch in batches),
        "squared_distance_pair_margin": numeric_summary(batch["frozen_pair_metrics"]["squared_distance_pair_margin"] for batch in batches),
        "cosine_pair_margin": numeric_summary(batch["frozen_pair_metrics"]["cosine_pair_margin"] for batch in batches),
        "mse_pair_residual_over_ntp": numeric_summary(batch["base_objectives"]["mse_true_vs_shuffled"]["pair_residual_over_ntp"] for batch in batches),
        "mse_difference_over_true": numeric_summary(batch["base_objectives"]["mse_true_vs_shuffled"]["difference_over_true"] for batch in batches),
        "sigreg_weighted_norm_over_ntp": numeric_summary(draw["objectives"]["sigreg_active_weighted"]["relative_to_ntp"] for draw in draws),
        "full_pair_specific_fraction": numeric_summary(draw["full_true_vs_shuffled"]["pair_specific_fraction_of_full_true"] for draw in draws),
        "sigreg_vs_ntp_cosine": numeric_summary(draw["objectives"]["sigreg_raw"]["relation_to_ntp"]["cosine_similarity"] for draw in draws),
        "sigreg_vs_pair_discrimination_cosine": numeric_summary(draw["objectives"]["sigreg_raw"]["relation_to_pair_discrimination_gradient"]["cosine_similarity"] for draw in draws),
        "sigreg_descent_effect_on_pair_discrimination": numeric_summary(draw["first_order_descent_effects"]["pair_discrimination"]["sigreg_active_weighted"]["descent_effect_per_unit_objective_gradient_norm"] for draw in draws),
        "full_descent_effect_on_pair_discrimination": numeric_summary(draw["first_order_descent_effects"]["pair_discrimination"]["full_active_true"]["descent_effect_per_unit_objective_gradient_norm"] for draw in draws),
        "sigreg_descent_effect_on_center_separation": numeric_summary(draw["first_order_descent_effects"]["center_separation"]["sigreg_active_weighted"]["descent_effect_per_unit_objective_gradient_norm"] for draw in draws),
        "sigreg_descent_effect_on_cosine_margin": numeric_summary(draw["first_order_descent_effects"]["cosine_retrieval_margin"]["sigreg_active_weighted"]["descent_effect_per_unit_objective_gradient_norm"] for draw in draws),
        "sigreg_descent_effect_on_joint_variance": numeric_summary(draw["first_order_descent_effects"]["joint_variance"]["sigreg_active_weighted"]["descent_effect_per_unit_objective_gradient_norm"] for draw in draws),
    }


def fixed_batches(rows, collator, count: int, physical_batch: int):
    permutation = torch.randperm(len(rows), generator=torch.Generator().manual_seed(SEED)).tolist()
    selected_indices = permutation[: count * BATCH_SIZE]
    batches = []
    for batch_index in range(count):
        indices = selected_indices[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE]
        selected = [rows[index] for index in indices]
        whole = collator(selected)
        _, targets = extract_source_and_target(whole)
        derangement = matched_derangement(targets, SHUFFLE_SEED + batch_index)
        chunks = [
            collator(selected[start:start + physical_batch])
            for start in range(0, BATCH_SIZE, physical_batch)
        ]
        batches.append({
            "batch_index": batch_index,
            "indices": indices,
            "rows": selected,
            "chunks": chunks,
            "derangement": derangement,
            "target_lengths": [len(target) for target in targets],
        })
    return selected_indices, batches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--physical-batch", type=int, default=PHYSICAL_BATCH)
    parser.add_argument("--batches", type=int, default=BATCHES)
    parser.add_argument("--draws", type=int, default=SIGREG_DRAWS)
    parser.add_argument("--only", choices=("all", "epoch1", "native"), default="all")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise EnvironmentError("the ChemFM audit requires CUDA")
    if BATCH_SIZE % args.physical_batch:
        raise ValueError("physical batch must divide 16")
    if not 1 <= args.batches <= BATCHES or not 1 <= args.draws <= SIGREG_DRAWS:
        raise ValueError("audit supports 1-4 batches and 1-4 draws")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    set_seed(SEED)
    rows = read_rows("uspto_mit_synthesis", path=TRAIN_MANIFEST)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab_size = len(tokenizer)
    predictor_ids = add_predictor_tokens(tokenizer)
    collator = ReactionCollator(tokenizer, task="forward")
    validate_serialization_endings(collator, rows, tokenizer.eos_token_id)
    selected_indices, fixed = fixed_batches(rows, collator, args.batches, args.physical_batch)
    model = load_lora_model(
        MODEL_DIR, tokenizer, attention_dropout=0.0, chemfm_vocab_size=chemfm_vocab_size
    ).cuda()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.eval()
    controls = disable_stochastic_behavior(model)
    method = CLMJEPA(predictor_ids, tokenizer.eos_token_id, tokenizer.pad_token_id, sigreg_seed=SEED)
    named_lora = lora_parameters(model)
    checkpoints = {
        "mse_sigreg_epoch1": CHECKPOINT_ROOT / "epoch_1",
        "mse_sigreg_epoch2": CHECKPOINT_ROOT / "epoch_2",
        "mse_sigreg_epoch4": CHECKPOINT_ROOT / "epoch_4",
        "matched_native_epoch4": NATIVE_CHECKPOINT,
    }
    if args.only == "epoch1":
        checkpoints = {"mse_sigreg_epoch1": checkpoints["mse_sigreg_epoch1"]}
    elif args.only == "native":
        checkpoints = {"matched_native_epoch4": checkpoints["matched_native_epoch4"]}
    for checkpoint in checkpoints.values():
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

    # Verify once at the first frozen state. The optimization changes neither
    # serialization nor endpoint definition and is checkpoint-independent.
    first_checkpoint = next(iter(checkpoints.values()))
    load_adapter_checkpoint(model, first_checkpoint)
    fast_path_equivalence = verify_endpoint_fast_path(
        method, model, fixed[0]["chunks"][0], named_lora
    )
    if not fast_path_equivalence["passed"]:
        raise RuntimeError(f"optimized endpoint path failed parity: {fast_path_equivalence}")

    started = time.perf_counter()
    results = {}
    no_update = {}
    for label, checkpoint in checkpoints.items():
        load_adapter_checkpoint(model, checkpoint)
        before = parameter_fingerprint(model)
        checkpoint_batches = []
        for batch in fixed:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            draw_seeds = [
                DRAW_SEED + batch["batch_index"] * 1009 + draw
                for draw in range(args.draws)
            ]
            result = audit_batch(
                label, model, method, batch["chunks"], batch["derangement"],
                draw_seeds, named_lora,
            )
            result["batch_index"] = batch["batch_index"]
            result["manifest_indices_zero_based"] = batch["indices"]
            result["derangement"] = batch["derangement"]
            result["target_token_lengths"] = batch["target_lengths"]
            checkpoint_batches.append(result)
            print(json.dumps({
                "checkpoint": label,
                "batch": batch["batch_index"] + 1,
                "of": args.batches,
                "seconds": result["runtime_seconds"],
            }), flush=True)
        after = parameter_fingerprint(model)
        no_update[label] = {"before": before, "after": after, "unchanged": before == after}
        if before != after:
            raise RuntimeError(f"parameters changed during frozen audit: {label}")
        results[label] = {
            "batches": checkpoint_batches,
            "aggregate": aggregate_checkpoint(checkpoint_batches),
        }

    output = {
        "scope": "frozen SIGReg pair-specificity audit; no optimizer, updates, training, or generation",
        "configuration": {
            "seed": SEED,
            "fixed_batches": args.batches,
            "examples_per_batch": BATCH_SIZE,
            "physical_batch": args.physical_batch,
            "sigreg_draws_per_batch": args.draws,
            "readout": "k=0 final source EOS and final target EOS",
            "jepa_loss": "raw symmetric mean squared endpoint distance",
            "sigreg": "exact Epps-Pulley; 17 knots [0,3], 1024 fresh random unit slices per call; source/target statistics independently averaged",
            "training_active_formula": "2*MSE + 0.0808080808*SIGReg",
            "sigreg_relative_coefficient": SIGREG_RELATIVE_COEFFICIENT,
            "active_sigreg_coefficient": APPLIED_SIGREG_COEFFICIENT,
            "training_auxiliary_activity_probability": 0.5,
            "audit_activity": "conditioned on an active auxiliary step; no Bernoulli dropout sampling",
            "projection_seed_scheme": "104729 + 1009*batch_index + draw_index; fresh per batch/draw and common across checkpoints",
            "stochastic_model_behavior": "train mode for gradient semantics; all Dropout and attention_dropout set to zero",
            "gradient_scope": "trainable LoRA A/B parameters only",
            "lora_parameter_tensors": len(named_lora),
            "lora_parameters": sum(parameter.numel() for _, parameter in named_lora),
            "optimizer_constructed": False,
            "gradient_checkpointing": args.gradient_checkpointing,
            "endpoint_execution": "verified source/target-only LlamaModel path; independent native row and unused vocabulary logits omitted",
            "endpoint_fast_path_equivalence": fast_path_equivalence,
            **controls,
        },
        "provenance": {
            "train_manifest": str(TRAIN_MANIFEST.resolve()),
            "train_manifest_sha256": sha256_file(TRAIN_MANIFEST),
            "selected_indices_zero_based": selected_indices,
            "checkpoints": {label: str(path.resolve()) for label, path in checkpoints.items()},
            "local_sigreg_implementation": str((ROOT / "src" / "jepa.py").resolve()),
            "upstream_lejepa_commit": "c293d291ca87cd4fddee9d3fffe4e914c7272052",
        },
        "sign_convention": (
            "For theta' = theta - eta*g_objective, the first-order metric change is "
            "Delta metric = -eta*(g_metric dot g_objective). Positive reported descent "
            "effect therefore means the objective step increases that metric."
        ),
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
        "aggregates": {label: value["aggregate"] for label, value in results.items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
