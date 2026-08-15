"""Frozen-gradient and LoRA block-swap audit for the ChemFM MSE+SIGReg endpoint.

This diagnostic never constructs an optimizer and never updates parameters.  It
uses the fixed seed-533 training examples established by the earlier SIGReg
gradient-response assay and the frozen 256-reaction validation panel used by
the MSE+SIGReg experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
from transformers import set_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemfm import (  # noqa: E402
    IGNORE_INDEX,
    MODEL_DIR,
    TOKENIZER_DIR,
    ReactionCollator,
    load_lora_model,
    load_reaction_tokenizer,
)
from geometry_diagnosis import effective_rank  # noqa: E402
from jepa import CLMJEPA, SIGReg, add_predictor_tokens, extract_source_and_target, matched_derangement  # noqa: E402
from train import ADAPTER_NAME, load_adapter_checkpoint, read_rows, validate_serialization_endings  # noqa: E402


SEED = 533
SHUFFLE_SEED = 1907
GRADIENT_BATCH = 16
PHYSICAL_BATCH = 2
LEJEPA_TRADEOFF = 0.01
SIGREG_RELATIVE_SCALE = 4.0
SIGREG_RELATIVE_COEFFICIENT = SIGREG_RELATIVE_SCALE * LEJEPA_TRADEOFF / (1.0 - LEJEPA_TRADEOFF)
ACTIVE_OUTER_COEFFICIENT = 2.0
AUXILIARY_ACTIVITY_PROBABILITY = 0.5
ACTIVE_SIGREG_COEFFICIENT = ACTIVE_OUTER_COEFFICIENT * SIGREG_RELATIVE_COEFFICIENT

TRAIN_MANIFEST = ROOT / "data" / "clm_jepa_uspto_mit_pilot_1280" / "uspto_mit_train.csv"
PANEL = ROOT / "data" / "clm_jepa_uspto_mit_validation_256" / "uspto_mit_validation_length_stratified_256.csv"
PANEL_IDENTITIES = ROOT / "data" / "clm_jepa_uspto_mit_validation_256" / "uspto_mit_validation_length_stratified_256.jsonl"
PARENT_VALIDATION_PANEL = ROOT / "data" / "clm_jepa_uspto_mit_validation_1024" / "uspto_mit_validation_1024.csv"
PRIOR_GRADIENT_ASSAY = ROOT / "runs" / "diagnostics" / "sigreg_gradient_response.json"
NATIVE_CHECKPOINT = ROOT / "runs" / "sigreg_batch16_pilot" / "matched_b4" / "native_checkpoints" / "epoch_4"
CLM_CHECKPOINT = ROOT / "runs" / "mse_ablation" / "stage1" / "mse_sigreg_checkpoints" / "epoch_4"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "diagnostics" / "mse_sigreg_mechanistic_audit"

LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
MODULE_FAMILIES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_fingerprint(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def device_batch(raw: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in raw.items()
    }


def disable_stochastic_behavior(model) -> dict[str, int]:
    dropout_modules = 0
    attention_fields = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
            dropout_modules += 1
        if hasattr(module, "attention_dropout"):
            module.attention_dropout = 0.0
            attention_fields += 1
    return {
        "dropout_modules_zeroed": dropout_modules,
        "attention_dropout_fields_zeroed": attention_fields,
    }


def capture_gradients(model) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        result[name] = (
            torch.zeros(parameter.shape, dtype=torch.float32)
            if parameter.grad is None
            else parameter.grad.detach().cpu().float().clone()
        )
    return result


def add_vectors(*terms: tuple[Mapping[str, torch.Tensor], float]) -> dict[str, torch.Tensor]:
    names = set().union(*(vector.keys() for vector, _ in terms))
    return {
        name: sum(
            (coefficient * vector[name] for vector, coefficient in terms if name in vector),
            torch.tensor(0.0),
        )
        for name in names
    }


def _selected_names(
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
    names: Iterable[str] | None,
) -> list[str]:
    available = set(first) & set(second)
    return sorted(available if names is None else available & set(names))


def vector_relation(
    auxiliary: Mapping[str, torch.Tensor],
    native: Mapping[str, torch.Tensor],
    names: Iterable[str] | None = None,
) -> dict[str, float | str | int]:
    chosen = _selected_names(auxiliary, native, names)
    dot = sum(
        float(torch.dot(auxiliary[name].flatten(), native[name].flatten()))
        for name in chosen
    )
    aux_sq = sum(float(value.square().sum()) for name, value in auxiliary.items() if name in chosen)
    native_sq = sum(float(value.square().sum()) for name, value in native.items() if name in chosen)
    aux_norm = math.sqrt(max(aux_sq, 0.0))
    native_norm = math.sqrt(max(native_sq, 0.0))
    denominator = aux_norm * native_norm
    cosine = dot / denominator if denominator else None
    cosine_sq = min(1.0, cosine * cosine) if cosine is not None else None
    parallel = cosine_sq if cosine is not None and cosine > 0 else 0.0
    opposed = cosine_sq if cosine is not None and cosine < 0 else 0.0
    orthogonal = 1.0 - cosine_sq if cosine_sq is not None else None
    return {
        "parameter_tensors": len(chosen),
        "auxiliary_norm": aux_norm,
        "native_norm": native_norm,
        "relative_norm_auxiliary_over_native": aux_norm / native_norm if native_norm else None,
        "dot_product": dot,
        "dot_sign": "positive" if dot > 0 else "negative" if dot < 0 else "zero",
        "cosine_similarity": cosine,
        "parallel_energy_fraction": parallel,
        "orthogonal_energy_fraction": orthogonal,
        "opposed_energy_fraction": opposed,
    }


def vector_difference_ratio(
    true: Mapping[str, torch.Tensor],
    shuffled: Mapping[str, torch.Tensor],
    names: Iterable[str] | None = None,
) -> float:
    chosen = _selected_names(true, shuffled, names)
    difference_sq = sum(
        float((true[name] - shuffled[name]).square().sum())
        for name in chosen
    )
    true_sq = sum(float(true[name].square().sum()) for name in chosen)
    return math.sqrt(difference_sq / true_sq) if true_sq else None


def parameter_groups(names: Sequence[str], layer_count: int) -> dict[str, dict[str, list[str]]]:
    layers = {f"layer_{index:02d}": [] for index in range(layer_count)}
    families: dict[str, list[str]] = {
        "attention_all": [],
        "mlp_all": [],
        "modules_to_save": [],
        **{family: [] for family in MODULE_FAMILIES},
    }
    globals_ = {"all_trainable": list(names), "lora_only": []}
    for name in names:
        match = LAYER_RE.search(name)
        if match:
            layers[f"layer_{int(match.group(1)):02d}"].append(name)
        if ".lora_" in name:
            globals_["lora_only"].append(name)
        if ".self_attn." in name:
            families["attention_all"].append(name)
        if ".mlp." in name:
            families["mlp_all"].append(name)
        if ".modules_to_save." in name:
            families["modules_to_save"].append(name)
        for family in MODULE_FAMILIES:
            if f".{family}." in name:
                families[family].append(name)
    return {"global": globals_, "layers": layers, "module_families": families}


def grouped_relations(
    auxiliary: Mapping[str, torch.Tensor],
    native: Mapping[str, torch.Tensor],
    groups: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, dict[str, dict[str, float | str | int]]]:
    membership: dict[str, list[tuple[str, str]]] = defaultdict(list)
    aggregates: dict[tuple[str, str], list[float]] = {}
    for category, members in groups.items():
        for label, names in members.items():
            if not names:
                continue
            aggregates[(category, label)] = [0.0, 0.0, 0.0, 0.0]
            for name in names:
                membership[name].append((category, label))
    for name in set(auxiliary) & set(native):
        if name not in membership:
            continue
        aux = auxiliary[name].flatten()
        ntp = native[name].flatten()
        values = (
            float(torch.dot(aux, ntp)),
            float(aux.square().sum()),
            float(ntp.square().sum()),
            1.0,
        )
        for key in membership[name]:
            accumulator = aggregates[key]
            for index, value in enumerate(values):
                accumulator[index] += value

    output: dict[str, dict[str, dict[str, float | str | int]]] = {
        category: {} for category in groups
    }
    for (category, label), (dot, aux_sq, native_sq, tensors) in aggregates.items():
        aux_norm = math.sqrt(max(aux_sq, 0.0))
        native_norm = math.sqrt(max(native_sq, 0.0))
        denominator = aux_norm * native_norm
        cosine = dot / denominator if denominator else None
        cosine_sq = min(1.0, cosine * cosine) if cosine is not None else None
        output[category][label] = {
            "parameter_tensors": int(tensors),
            "auxiliary_norm": aux_norm,
            "native_norm": native_norm,
            "relative_norm_auxiliary_over_native": aux_norm / native_norm if native_norm else None,
            "dot_product": dot,
            "dot_sign": "positive" if dot > 0 else "negative" if dot < 0 else "zero",
            "cosine_similarity": cosine,
            "parallel_energy_fraction": cosine_sq if cosine is not None and cosine > 0 else 0.0,
            "orthogonal_energy_fraction": 1.0 - cosine_sq if cosine_sq is not None else None,
            "opposed_energy_fraction": cosine_sq if cosine is not None and cosine < 0 else 0.0,
        }
    return output


def partition_target_regions(labels: torch.Tensor) -> dict[str, torch.Tensor]:
    """Mask target labels into per-example early/middle/late thirds.

    Partitioning occurs after the causal shift, includes the target EOS, and
    assigns token rank ``floor(3 * rank / target_length)``.
    """
    shifted = labels[:, 1:]
    result = {
        region: torch.full_like(shifted, IGNORE_INDEX)
        for region in ("early", "middle", "late")
    }
    region_names = ("early", "middle", "late")
    for row_index, row in enumerate(shifted):
        positions = row.ne(IGNORE_INDEX).nonzero(as_tuple=False).flatten()
        length = int(positions.numel())
        if not length:
            raise ValueError("each reaction must contain target tokens")
        for rank, position in enumerate(positions.tolist()):
            region = min(2, 3 * rank // length)
            result[region_names[region]][row_index, position] = row[position]
    return result


def geometry(states: torch.Tensor) -> dict[str, float]:
    values = states.detach().float().cpu()
    mean_energy = values.mean(0).square().sum() / values.square().sum(1).mean().clamp_min(1e-30)
    return {
        "variance": float(values.var(0, unbiased=False).mean()),
        "mean_direction_energy": float(mean_energy),
        "effective_rank": float(effective_rank(values)),
        "mean_embedding_norm": float(values.norm(dim=1).mean()),
        "mean_vector_norm": float(values.mean(0).norm()),
    }


def endpoint_output(method: CLMJEPA, model, batch: Mapping[str, Any]):
    return method(
        model,
        batch,
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


def collect_endpoint_statistics(method, model, chunks):
    accumulator = None
    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for raw in chunks:
            output = endpoint_output(method, model, device_batch(raw, model.device))
            source = output.source_states.detach().float()
            target = output.target_states.detach().float()
            sources.append(source)
            targets.append(target)
            states = torch.stack((source, target))
            if accumulator is None:
                accumulator = SIGReg(seed=SEED).start_streaming(
                    views=2,
                    dimensions=states.size(-1),
                    expected_samples=GRADIENT_BATCH,
                    device=states.device,
                )
            accumulator.update(states)
    if accumulator is None:
        raise RuntimeError("empty endpoint panel")
    prepared = accumulator.finalize()
    source = torch.cat(sources)
    target = torch.cat(targets)
    return source, target, prepared


def endpoint_objective_gradients(source, target, prepared, permutation):
    leaf_source = source.detach().clone().requires_grad_(True)
    leaf_target = target.detach().clone().requires_grad_(True)
    mse = F.mse_loss(leaf_source, leaf_target)
    mse_source, mse_target = torch.autograd.grad(mse, (leaf_source, leaf_target))

    shuffled_mse = F.mse_loss(leaf_source, leaf_target[permutation])
    shuffled_source, shuffled_target = torch.autograd.grad(
        shuffled_mse, (leaf_source, leaf_target)
    )
    sigreg_gradient = prepared.representation_gradients(
        torch.stack((leaf_source, leaf_target))
    ).to(leaf_source.dtype)
    gradients = {
        "mse_source": (mse_source, torch.zeros_like(mse_target)),
        "mse_target": (torch.zeros_like(mse_source), mse_target),
        "sigreg": (sigreg_gradient[0], sigreg_gradient[1]),
        "mse_shuffled": (shuffled_source, shuffled_target),
    }
    values = {
        "mse_true": float(mse.detach()),
        "mse_shuffled": float(shuffled_mse.detach()),
        "sigreg": float(prepared.loss.detach()),
        "active_weighted_auxiliary": float(
            ACTIVE_OUTER_COEFFICIENT
            * (mse.detach() + SIGREG_RELATIVE_COEFFICIENT * prepared.loss.detach())
        ),
    }
    return gradients, values


def accumulate_vjp(model, method, chunks, endpoint_gradients):
    names = list(endpoint_gradients)
    accumulators: dict[str, dict[str, torch.Tensor] | None] = {name: None for name in names}
    offset = 0
    for raw in chunks:
        batch_size = int(raw["input_ids"].size(0))
        output = endpoint_output(method, model, device_batch(raw, model.device))
        for objective_index, objective in enumerate(names):
            source_gradient, target_gradient = endpoint_gradients[objective]
            model.zero_grad(set_to_none=True)
            torch.autograd.backward(
                (output.source_states, output.target_states),
                (
                    source_gradient[offset:offset + batch_size].to(output.source_states.dtype),
                    target_gradient[offset:offset + batch_size].to(output.target_states.dtype),
                ),
                retain_graph=objective_index < len(names) - 1,
            )
            captured = capture_gradients(model)
            if accumulators[objective] is None:
                accumulators[objective] = captured
            else:
                accumulators[objective] = add_vectors((accumulators[objective], 1.0), (captured, 1.0))
        del output
        offset += batch_size
    model.zero_grad(set_to_none=True)
    return {name: value for name, value in accumulators.items() if value is not None}


def native_region_gradients(model, chunks):
    region_names = ("early", "middle", "late")
    counts = {region: 0 for region in region_names}
    nll_sums = {region: 0.0 for region in region_names}
    for raw in chunks:
        masks = partition_target_regions(raw["labels"])
        for region in region_names:
            counts[region] += int(masks[region].ne(IGNORE_INDEX).sum())
    accumulators: dict[str, dict[str, torch.Tensor] | None] = {region: None for region in region_names}
    for raw in chunks:
        batch = device_batch(raw, model.device)
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=False,
        )
        logits = output.logits[:, :-1].float()
        masks = partition_target_regions(batch["labels"])
        for region_index, region in enumerate(region_names):
            labels = masks[region]
            active = labels.ne(IGNORE_INDEX)
            nll_sum = F.cross_entropy(logits[active], labels[active], reduction="sum")
            nll_sums[region] += float(nll_sum.detach())
            model.zero_grad(set_to_none=True)
            (nll_sum / counts[region]).backward(retain_graph=region_index < 2)
            captured = capture_gradients(model)
            if accumulators[region] is None:
                accumulators[region] = captured
            else:
                accumulators[region] = add_vectors((accumulators[region], 1.0), (captured, 1.0))
        del output
    model.zero_grad(set_to_none=True)
    regional = {name: value for name, value in accumulators.items() if value is not None}
    total_count = sum(counts.values())
    total = add_vectors(*[
        (regional[region], counts[region] / total_count) for region in region_names
    ])
    losses = {
        region: nll_sums[region] / counts[region] for region in region_names
    }
    losses["total"] = sum(nll_sums.values()) / total_count
    counts["total"] = total_count
    return {"total": total, **regional}, losses, counts


def checkpoint_fixed_rows():
    prior = json.loads(PRIOR_GRADIENT_ASSAY.read_text(encoding="utf-8"))
    indices = [int(row["manifest_index_zero_based"]) for row in prior["fixed_examples"]]
    rows = read_rows("uspto_mit_synthesis", path=TRAIN_MANIFEST)
    fixed = [rows[index] for index in indices]
    for expected, actual in zip(prior["fixed_examples"], fixed):
        if hashlib.sha256(actual["src"].encode()).hexdigest() != expected["source_sha256"]:
            raise RuntimeError("fixed source panel no longer matches prior assay")
        if hashlib.sha256(actual["tgt"].encode()).hexdigest() != expected["target_sha256"]:
            raise RuntimeError("fixed target panel no longer matches prior assay")
    return indices, fixed


def summarize_checkpoint_gradients(label, model, method, chunks, permutation, groups):
    started = time.perf_counter()
    before = parameter_fingerprint(model)
    native_vectors, native_losses, native_counts = native_region_gradients(model, chunks)
    print(json.dumps({"checkpoint": label, "stage": "native_gradients", "elapsed": time.perf_counter() - started}), flush=True)
    source, target, prepared = collect_endpoint_statistics(method, model, chunks)
    endpoint_gradients, objective_values = endpoint_objective_gradients(
        source, target, prepared, permutation
    )
    components = accumulate_vjp(model, method, chunks, endpoint_gradients)
    print(json.dumps({"checkpoint": label, "stage": "auxiliary_gradients", "elapsed": time.perf_counter() - started}), flush=True)
    mse_true = add_vectors((components["mse_source"], 1.0), (components["mse_target"], 1.0))
    sigreg = components["sigreg"]
    active_aux = add_vectors(
        (mse_true, ACTIVE_OUTER_COEFFICIENT),
        (sigreg, ACTIVE_SIGREG_COEFFICIENT),
    )
    shuffled_aux = add_vectors(
        (components["mse_shuffled"], ACTIVE_OUTER_COEFFICIENT),
        (sigreg, ACTIVE_SIGREG_COEFFICIENT),
    )
    pair_residual = add_vectors((active_aux, 1.0), (shuffled_aux, -1.0))
    vectors = {
        "mse_true": mse_true,
        "sigreg": sigreg,
        "active_weighted_auxiliary": active_aux,
        "mse_source_only_target_detached": components["mse_source"],
        "mse_target_only_source_detached": components["mse_target"],
        "mse_shuffled": components["mse_shuffled"],
        "active_weighted_auxiliary_shuffled": shuffled_aux,
        "pair_specific_auxiliary_residual": pair_residual,
    }
    comparisons_to_total_ntp = {
        objective: grouped_relations(vector, native_vectors["total"], groups)
        for objective, vector in vectors.items()
    }
    position_groups = {
        "global": groups["global"],
        "layers": groups["layers"],
    }
    position_compatibility = {
        objective: {
            region: grouped_relations(vectors[objective], native_vectors[region], position_groups)
            for region in ("early", "middle", "late")
        }
        for objective in ("mse_true", "sigreg", "active_weighted_auxiliary")
    }
    true_shuffle = {
        "mse": {
            "relations": grouped_relations(mse_true, components["mse_shuffled"], groups),
            "difference_over_true": {
                scope: vector_difference_ratio(mse_true, components["mse_shuffled"], names)
                for scope, names in groups["global"].items()
            },
        },
        "active_weighted_auxiliary": {
            "relations": grouped_relations(active_aux, shuffled_aux, groups),
            "difference_over_true": {
                scope: vector_difference_ratio(active_aux, shuffled_aux, names)
                for scope, names in groups["global"].items()
            },
        },
        "pair_specific_residual_vs_ntp": grouped_relations(pair_residual, native_vectors["total"], groups),
    }
    after = parameter_fingerprint(model)
    if before != after:
        raise RuntimeError(f"parameters changed during frozen audit for {label}")
    return {
        "checkpoint": label,
        "trainable_parameter_sha256_before": before,
        "trainable_parameter_sha256_after": after,
        "no_parameter_change": before == after,
        "native_losses": native_losses,
        "native_token_counts": native_counts,
        "objective_values": objective_values,
        "geometry": {"source": geometry(source), "target": geometry(target)},
        "gradient_comparisons_to_total_ntp": comparisons_to_total_ntp,
        "autoregressive_position_compatibility": position_compatibility,
        "true_vs_shuffled": true_shuffle,
        "runtime_seconds": time.perf_counter() - started,
        "peak_allocated_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
    }


def trainable_state(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().float().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def state_delta_summary(native_state, clm_state, groups):
    delta = add_vectors((clm_state, 1.0), (native_state, -1.0))
    zero = {name: torch.zeros_like(value) for name, value in delta.items()}
    result = {}
    for category, members in groups.items():
        result[category] = {}
        for label, names in members.items():
            chosen = set(names)
            square = sum(float(value.square().sum()) for name, value in delta.items() if name in chosen)
            parameter_count = sum(value.numel() for name, value in delta.items() if name in chosen)
            result[category][label] = {
                "delta_l2_norm": math.sqrt(square),
                "parameter_count": parameter_count,
            }
    return result


def run_gradient_audit(output_dir: Path) -> Path:
    if not torch.cuda.is_available():
        raise EnvironmentError("the local ChemFM gradient audit requires CUDA")
    for path in (TRAIN_MANIFEST, PANEL, PRIOR_GRADIENT_ASSAY, NATIVE_CHECKPOINT, CLM_CHECKPOINT):
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    set_seed(SEED)
    indices, rows = checkpoint_fixed_rows()
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab_size = len(tokenizer)
    predictor_ids = add_predictor_tokens(tokenizer)
    collator = ReactionCollator(tokenizer, task="forward")
    validate_serialization_endings(collator, rows, tokenizer.eos_token_id)
    chunks = [
        collator(rows[start:start + PHYSICAL_BATCH])
        for start in range(0, len(rows), PHYSICAL_BATCH)
    ]
    all_targets = []
    for chunk in chunks:
        _, targets = extract_source_and_target(chunk)
        all_targets.extend(targets)
    permutation = matched_derangement(all_targets, SHUFFLE_SEED)

    model = load_lora_model(
        MODEL_DIR, tokenizer, attention_dropout=0.0, chemfm_vocab_size=chemfm_vocab_size
    ).cuda()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.train()
    controls = disable_stochastic_behavior(model)
    method = CLMJEPA(predictor_ids, tokenizer.eos_token_id, tokenizer.pad_token_id, sigreg_seed=SEED)
    base_state = trainable_state(model)
    parameter_names = list(base_state)
    groups = parameter_groups(parameter_names, model.config.num_hidden_layers)

    checkpoints = {
        "base_chemfm": None,
        "matched_native_epoch4": NATIVE_CHECKPOINT,
        "mse_sigreg_epoch4": CLM_CHECKPOINT,
    }
    results = {}
    frozen_states = {}
    total_started = time.perf_counter()
    for label, checkpoint in checkpoints.items():
        if checkpoint is None:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in base_state:
                        parameter.copy_(base_state[name].to(parameter.dtype))
        else:
            load_adapter_checkpoint(model, checkpoint)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        frozen_states[label] = trainable_state(model)
        results[label] = summarize_checkpoint_gradients(
            label, model, method, chunks, permutation, groups
        )
        print(json.dumps({"completed": label, "seconds": results[label]["runtime_seconds"]}), flush=True)

    output = {
        "scope": "frozen-checkpoint mechanistic gradient audit; no optimizer and no parameter updates",
        "seed": SEED,
        "checkpoints": {
            "base_chemfm": str(MODEL_DIR.resolve()),
            "matched_native_epoch4": str(NATIVE_CHECKPOINT.resolve()),
            "mse_sigreg_epoch4": str(CLM_CHECKPOINT.resolve()),
        },
        "manifests": {
            "gradient_training_panel": str(TRAIN_MANIFEST.resolve()),
            "gradient_training_panel_sha256": sha256_file(TRAIN_MANIFEST),
            "fixed_indices_zero_based": indices,
            "validation_panel": str(PANEL.resolve()),
            "validation_panel_sha256": sha256_file(PANEL),
        },
        "configuration": {
            "examples": GRADIENT_BATCH,
            "physical_batch": PHYSICAL_BATCH,
            "readout": "k=0 final source EOS and final target EOS",
            "mse": "mean((z_source-z_target)^2), symmetric unless branch-specific VJP is zeroed",
            "sigreg": "validated exact LeJEPA Epps-Pulley statistic; 17 knots, 1024 fixed seed-533 slices, source/target views independently averaged",
            "sigreg_tradeoff": LEJEPA_TRADEOFF,
            "sigreg_relative_scale": SIGREG_RELATIVE_SCALE,
            "sigreg_relative_coefficient": SIGREG_RELATIVE_COEFFICIENT,
            "active_outer_coefficient": ACTIVE_OUTER_COEFFICIENT,
            "active_sigreg_coefficient": ACTIVE_SIGREG_COEFFICIENT,
            "auxiliary_activity_probability": AUXILIARY_ACTIVITY_PROBABILITY,
            "active_auxiliary_formula": "2*MSE + 0.0808080808*SIGReg",
            "expected_auxiliary_formula": "MSE + 0.0404040404*SIGReg",
            "target_partition": "per-reaction causal target positions, including EOS, assigned by floor(3*rank/length)",
            "shuffle_seed": SHUFFLE_SEED,
            "shuffle_indices": permutation,
            "shuffle_total_absolute_target_token_length_cost": sum(
                abs(len(all_targets[i]) - len(all_targets[j])) for i, j in enumerate(permutation)
            ),
            "optimizer_constructed": False,
            "model_mode": "train for gradient checkpointing; all Dropout and attention_dropout disabled",
            **controls,
            "trainable_parameters": sum(value.numel() for value in base_state.values()),
            "trainable_tensors": len(base_state),
            "transformer_layers": model.config.num_hidden_layers,
        },
        "results": results,
        "native_to_clm_parameter_delta": state_delta_summary(
            frozen_states["matched_native_epoch4"], frozen_states["mse_sigreg_epoch4"], groups
        ),
        "total_runtime_seconds": time.perf_counter() - total_started,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    path = output_dir / "gradient_audit.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def adapter_weights_dir(checkpoint: Path) -> Path:
    nested = checkpoint / ADAPTER_NAME
    return nested if nested.exists() else checkpoint


def load_adapter_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    return load_peft_weights(str(adapter_weights_dir(checkpoint)), device="cpu")


def swap_group_for_key(key: str, groups: Sequence[Mapping[str, Any]]) -> str | None:
    layer_match = LAYER_RE.search(key)
    layer = int(layer_match.group(1)) if layer_match else None
    for group in groups:
        if group["kind"] == "layers" and layer is not None and group["start"] <= layer < group["stop"]:
            return group["name"]
        if group["kind"] == "token_io" and layer is None and (
            ".embed_tokens." in key or ".lm_head." in key
        ):
            return group["name"]
    return None


def hybrid_adapter_state(background, donor, selected_group: str, groups):
    if set(background) != set(donor):
        raise ValueError("adapter checkpoints have different keys")
    hybrid = {}
    changed = []
    for key in background:
        use_donor = swap_group_for_key(key, groups) == selected_group
        hybrid[key] = donor[key].clone() if use_donor else background[key].clone()
        if use_donor:
            changed.append(key)
    if not changed:
        raise ValueError(f"swap group {selected_group!r} matched no adapter keys")
    return hybrid, changed


def validate_hybrid(hybrid, background, donor, changed):
    changed_set = set(changed)
    for key, value in hybrid.items():
        expected = donor[key] if key in changed_set else background[key]
        if not torch.equal(value, expected):
            raise RuntimeError(f"hybrid state mismatch at {key}")
    return {
        "changed_tensors": len(changed),
        "changed_parameters": sum(hybrid[key].numel() for key in changed),
        "unchanged_tensors": len(hybrid) - len(changed),
        "exact_state_validation": True,
    }


def read_validation_panel(path: Path):
    # Match decoder_coupling.read_analysis_panel exactly: the 256 artifact
    # freezes identities/order, while source/target views come from the frozen
    # 1,024 parent panel.  The selection CSV contains length-stratification
    # metadata but not necessarily the evaluation R-SMILES view.
    identity_records = [
        json.loads(line)
        for line in PANEL_IDENTITIES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    identity_records.sort(key=lambda row: row["panel_index"])
    identities = [row["reaction_identity"] for row in identity_records]
    with PARENT_VALIDATION_PANEL.open(newline="", encoding="utf-8") as handle:
        parent = list(csv.DictReader(handle))
    by_identity = {row["reaction_identity"]: row for row in parent}
    rows = [by_identity[identity] for identity in identities]
    if len(rows) != 256 or len(set(identities)) != len(rows):
        raise ValueError("expected 256 unique frozen validation reactions")
    return rows


@torch.inference_mode()
def evaluate_target_ce(model, collator, panel, batch_size: int):
    model.eval()
    reactions = []
    total_nll = 0.0
    model_label_tokens = 0
    protocol_denominator_tokens = sum(
        len(tokens)
        for tokens in collator.tokenizer(
            [row["target"] for row in panel], add_special_tokens=False
        )["input_ids"]
    )
    started = time.perf_counter()
    for start in range(0, len(panel), batch_size):
        subset = panel[start:start + batch_size]
        batch = collator([{"src": row["source"], "tgt": row["target"]} for row in subset])
        batch = device_batch(batch, model.device)
        output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = output.logits[:, :-1].float()
        labels = batch["labels"][:, 1:]
        for row, row_logits, row_labels in zip(subset, logits, labels):
            active = row_labels.ne(IGNORE_INDEX)
            nll = F.cross_entropy(row_logits[active], row_labels[active], reduction="sum")
            token_count = int(active.sum())
            nll_value = float(nll)
            total_nll += nll_value
            model_label_tokens += token_count
            raw_target_tokens = len(
                collator.tokenizer(row["target"], add_special_tokens=False)["input_ids"]
            )
            reactions.append({
                "reaction_identity": row["reaction_identity"],
                "example_id": row["example_id"],
                "protocol_denominator_tokens": raw_target_tokens,
                "model_label_tokens": token_count,
                "target_nll_sum": nll_value,
                "target_ce": nll_value / token_count,
            })
    return {
        # Preserve the established report's aggregate denominator so the full
        # endpoints reproduce.  The correctly normalized model-label CE is
        # recorded separately; paired reaction CE already used this divisor.
        "aggregate_target_token_ce": total_nll / protocol_denominator_tokens,
        "model_label_aggregate_target_ce": total_nll / model_label_tokens,
        "mean_reaction_target_ce": sum(row["target_ce"] for row in reactions) / len(reactions),
        "protocol_denominator_tokens": protocol_denominator_tokens,
        "model_label_tokens": model_label_tokens,
        "special_tokens_excluded_by_protocol_denominator": model_label_tokens - protocol_denominator_tokens,
        "reactions": reactions,
        "runtime_seconds": time.perf_counter() - started,
        "peak_allocated_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
    }


def _bootstrap_paired_ci(candidate_rows, reference_rows, *, seed: int = SEED, draws: int = 10000):
    candidate = {row["reaction_identity"]: row for row in candidate_rows}
    reference = {row["reaction_identity"]: row for row in reference_rows}
    identities = list(candidate)
    reaction_deltas = torch.tensor(
        [candidate[key]["target_ce"] - reference[key]["target_ce"] for key in identities],
        dtype=torch.float64,
    )
    nll_deltas = torch.tensor(
        [candidate[key]["target_nll_sum"] - reference[key]["target_nll_sum"] for key in identities],
        dtype=torch.float64,
    )
    denominators = torch.tensor(
        [candidate[key]["protocol_denominator_tokens"] for key in identities],
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(identities), (draws, len(identities)), generator=generator)
    reaction_means = reaction_deltas[indices].mean(1)
    aggregate_differences = nll_deltas[indices].sum(1) / denominators[indices].sum(1)
    return {
        "draws": draws,
        "seed": seed,
        "mean_reaction_ce_change_95_ci": [
            float(torch.quantile(reaction_means, 0.025)),
            float(torch.quantile(reaction_means, 0.975)),
        ],
        "aggregate_protocol_ce_change_95_ci": [
            float(torch.quantile(aggregate_differences, 0.025)),
            float(torch.quantile(aggregate_differences, 0.975)),
        ],
    }


def paired_ce_summary(candidate, native, clm):
    by_id = {
        label: {row["reaction_identity"]: row["target_ce"] for row in result["reactions"]}
        for label, result in (("candidate", candidate), ("native", native), ("clm", clm))
    }
    ids = list(by_id["candidate"])
    candidate_minus_native = [by_id["candidate"][key] - by_id["native"][key] for key in ids]
    candidate_minus_clm = [by_id["candidate"][key] - by_id["clm"][key] for key in ids]
    full_gap = clm["aggregate_target_token_ce"] - native["aggregate_target_token_ce"]
    return {
        "mean_paired_reaction_ce_change_vs_native": sum(candidate_minus_native) / len(ids),
        "median_paired_reaction_ce_change_vs_native": float(torch.tensor(candidate_minus_native).median()),
        "reactions_improved_vs_native": sum(value < 0 for value in candidate_minus_native),
        "reactions_worsened_vs_native": sum(value > 0 for value in candidate_minus_native),
        "mean_paired_reaction_ce_change_vs_clm": sum(candidate_minus_clm) / len(ids),
        "fraction_of_full_aggregate_ce_gap_from_native": (
            (candidate["aggregate_target_token_ce"] - native["aggregate_target_token_ce"]) / full_gap
            if full_gap else None
        ),
        "fraction_of_full_aggregate_ce_gap_removed_from_clm": (
            (clm["aggregate_target_token_ce"] - candidate["aggregate_target_token_ce"]) / full_gap
            if full_gap else None
        ),
        "uncertainty_vs_native": _bootstrap_paired_ci(
            candidate["reactions"], native["reactions"], seed=SEED
        ),
        "uncertainty_vs_clm": _bootstrap_paired_ci(
            candidate["reactions"], clm["reactions"], seed=SEED + 1
        ),
    }


def refresh_swap_summaries(path: Path) -> Path:
    output = json.loads(path.read_text(encoding="utf-8"))
    native = output["results"]["full_native"]
    clm = output["results"]["full_mse_sigreg"]
    output["paired_summaries"] = {
        label: paired_ce_summary(result, native, clm)
        for label, result in output["results"].items()
    }
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def default_swap_groups(layer_count: int):
    boundaries = [round(index * layer_count / 4) for index in range(5)]
    groups = [
        {"name": f"depth_q{index + 1}", "kind": "layers", "start": boundaries[index], "stop": boundaries[index + 1]}
        for index in range(4)
    ]
    groups.append({"name": "token_io", "kind": "token_io"})
    return groups


def run_swap_audit(output_dir: Path, groups_path: Path | None, batch_size: int) -> Path:
    if not torch.cuda.is_available():
        raise EnvironmentError("the local ChemFM swap audit requires CUDA")
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = read_validation_panel(PANEL)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab_size = len(tokenizer)
    add_predictor_tokens(tokenizer)
    collator = ReactionCollator(tokenizer, task="forward")
    model = load_lora_model(
        MODEL_DIR, tokenizer, attention_dropout=0.0, chemfm_vocab_size=chemfm_vocab_size
    ).cuda().eval()
    disable_stochastic_behavior(model)
    native_state = load_adapter_state(NATIVE_CHECKPOINT)
    clm_state = load_adapter_state(CLM_CHECKPOINT)
    if set(native_state) != set(clm_state):
        raise RuntimeError("native and cLM adapter keys differ")
    groups = (
        json.loads(groups_path.read_text(encoding="utf-8"))["groups"]
        if groups_path is not None
        else default_swap_groups(model.config.num_hidden_layers)
    )
    results = {}
    validations = {}

    def apply_and_evaluate(label, state):
        set_result = set_peft_model_state_dict(model, state, adapter_name=ADAPTER_NAME)
        if getattr(set_result, "unexpected_keys", None):
            raise RuntimeError(f"unexpected adapter keys for {label}: {set_result.unexpected_keys}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        results[label] = evaluate_target_ce(model, collator, panel, batch_size)
        print(json.dumps({"completed": label, "ce": results[label]["aggregate_target_token_ce"], "seconds": results[label]["runtime_seconds"]}), flush=True)

    apply_and_evaluate("full_native", native_state)
    apply_and_evaluate("full_mse_sigreg", clm_state)
    for group in groups:
        group_name = group["name"]
        for background_label, background, donor_label, donor in (
            ("native", native_state, "mse_sigreg", clm_state),
            ("mse_sigreg", clm_state, "native", native_state),
        ):
            label = f"{background_label}_with_{donor_label}_{group_name}"
            state, changed = hybrid_adapter_state(background, donor, group_name, groups)
            validations[label] = validate_hybrid(state, background, donor, changed)
            validations[label]["changed_keys"] = changed
            apply_and_evaluate(label, state)

    native = results["full_native"]
    clm = results["full_mse_sigreg"]
    expected_native = 0.24068289650178812
    expected_clm = 0.24877883745660193
    # The frozen references were produced on an A6000.  Identical batch-4
    # semantics on the RTX 4050 can differ slightly in BF16 SDPA reductions;
    # this bound is 0.2% of the reported CE and far below the endpoint gap.
    tolerance = 5e-4
    reproduction = {
        "expected_native_ce": expected_native,
        "observed_native_ce": native["aggregate_target_token_ce"],
        "native_abs_error": abs(native["aggregate_target_token_ce"] - expected_native),
        "expected_mse_sigreg_ce": expected_clm,
        "observed_mse_sigreg_ce": clm["aggregate_target_token_ce"],
        "mse_sigreg_abs_error": abs(clm["aggregate_target_token_ce"] - expected_clm),
        "tolerance": tolerance,
    }
    reproduction["passed"] = (
        reproduction["native_abs_error"] <= tolerance
        and reproduction["mse_sigreg_abs_error"] <= tolerance
    )
    if not reproduction["passed"]:
        raise RuntimeError(f"endpoint CE reproduction failed: {reproduction}")
    paired = {
        label: paired_ce_summary(result, native, clm)
        for label, result in results.items()
    }
    output = {
        "scope": "frozen LoRA block-swap causal CE audit; no training and no optimizer",
        "checkpoints": {
            "native": str(NATIVE_CHECKPOINT.resolve()),
            "mse_sigreg": str(CLM_CHECKPOINT.resolve()),
        },
        "identity_panel": str(PANEL_IDENTITIES.resolve()),
        "identity_panel_sha256": sha256_file(PANEL_IDENTITIES),
        "parent_evaluation_panel": str(PARENT_VALIDATION_PANEL.resolve()),
        "parent_evaluation_panel_sha256": sha256_file(PARENT_VALIDATION_PANEL),
        "identities": len(panel),
        "batch_size": batch_size,
        "groups": groups,
        "swap_validation": validations,
        "endpoint_reproduction": reproduction,
        "results": results,
        "paired_summaries": paired,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }
    path = output_dir / "block_swap_audit.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("gradient", "swap", "summarize-swap", "all"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--groups", type=Path, help="optional JSON containing a groups list informed by the gradient audit")
    parser.add_argument("--ce-batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.command in {"gradient", "all"}:
        print(run_gradient_audit(args.output_dir))
    if args.command in {"swap", "all"}:
        print(run_swap_audit(args.output_dir, args.groups, args.ce_batch_size))
    if args.command == "summarize-swap":
        print(refresh_swap_summaries(args.output_dir / "block_swap_audit.json"))


if __name__ == "__main__":
    main()
