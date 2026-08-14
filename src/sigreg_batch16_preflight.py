from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler, set_seed

from chemfm import (
    MODEL_DIR,
    TOKENIZER_DIR,
    ReactionCollator,
    load_lora_model,
    load_reaction_tokenizer,
)
from jepa import CLMJEPA, SIGReg, add_predictor_tokens
from train import (
    ADAM_BETAS,
    ADAM_EPSILON,
    MIN_LEARNING_RATE,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    _restore_rng,
    _rng_snapshot,
    read_rows,
    validate_serialization_endings,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "data" / "gate45_v2" / "uspto_mit_train.csv"
DEFAULT_OUTPUT = ROOT / "runs" / "diagnostics" / "sigreg_batch16_preflight.json"

SEED = 533
PHYSICAL_BATCH = 2
SIGREG_BATCH = 16
ESTABLISHED_EFFECTIVE_BATCH = 8
TRAIN_ROWS = 1280
FULL_EPOCHS = 2
LEJEPA_TRADEOFF = 0.01
JEPA_RATIO = 0.5
LAMBDA_EFF = 1.0
ACTUAL_LAMBDA = LAMBDA_EFF / JEPA_RATIO


def relative_sigreg_coefficient(tradeoff: float) -> float:
    if not 0.0 <= tradeoff < 1.0:
        raise ValueError("LeJEPA trade-off must be in [0, 1)")
    return tradeoff / (1.0 - tradeoff)


def _device_batch(raw: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device)
        for name, value in raw.items()
        if torch.is_tensor(value)
    }


def _trainable(model) -> list[tuple[str, torch.nn.Parameter]]:
    return [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]


def _capture_gradients(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in _trainable(model)
        if parameter.grad is not None
    }


def _vector_stats(
    vectors: dict[str, dict[str, torch.Tensor]],
    coefficients: dict[str, float],
) -> dict[str, Any]:
    names = sorted({name for vector in vectors.values() for name in vector})
    squared_norm = 0.0
    max_abs = 0.0
    for name in names:
        combined = None
        for key, coefficient in coefficients.items():
            value = vectors[key].get(name)
            if value is not None:
                term = value.double().mul(coefficient)
                combined = term if combined is None else combined.add(term)
        if combined is None:
            continue
        squared_norm += float(combined.square().sum())
        max_abs = max(max_abs, float(combined.abs().max()))
    return {"norm": math.sqrt(squared_norm), "max_abs": max_abs}


def _dot(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> float:
    return sum(
        float(left[name].double().mul(right[name].double()).sum())
        for name in left.keys() & right.keys()
    )


def _cosine(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> float | None:
    left_norm = math.sqrt(max(_dot(left, left), 0.0))
    right_norm = math.sqrt(max(_dot(right, right), 0.0))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return _dot(left, right) / (left_norm * right_norm)


def _combine_vectors(
    vectors: dict[str, dict[str, torch.Tensor]],
    coefficients: dict[str, float],
) -> dict[str, torch.Tensor]:
    names = sorted({name for vector in vectors.values() for name in vector})
    result: dict[str, torch.Tensor] = {}
    for name in names:
        combined = None
        for key, coefficient in coefficients.items():
            value = vectors[key].get(name)
            if value is not None:
                term = value.mul(coefficient)
                combined = term if combined is None else combined.add(term)
        if combined is not None:
            result[name] = combined
    return result


def _global_relative_error(
    actual: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]
) -> tuple[float, float, float | None]:
    names = sorted(actual.keys() | reference.keys())
    difference_sq = 0.0
    reference_sq = 0.0
    max_abs = 0.0
    for name in names:
        a = actual.get(name)
        b = reference.get(name)
        if a is None:
            a = torch.zeros_like(b)
        if b is None:
            b = torch.zeros_like(a)
        difference = a.double() - b.double()
        difference_sq += float(difference.square().sum())
        reference_sq += float(b.double().square().sum())
        max_abs = max(max_abs, float(difference.abs().max()))
    relative = math.sqrt(difference_sq) / max(math.sqrt(reference_sq), 1e-30)
    return relative, max_abs, _cosine(actual, reference)


def _stack_states(output) -> torch.Tensor:
    if output.source_states is None or output.target_states is None:
        raise RuntimeError("JEPA endpoint states were not returned")
    return torch.stack((output.source_states, output.target_states))


def _endpoint_output(method, model, batch):
    return method(
        model,
        batch,
        k=0,
        jepa_weight=0.0,
        native_weight=1.0,
        monitor_only=True,
        stop_gradient_target=False,
        sigreg_tradeoff=0.0,
        jepa_ratio=JEPA_RATIO,
        force_jepa_active=True,
    )


def _real_equivalence_test(
    *, model, method, collator, rows: list[dict[str, str]], indices: list[int]
) -> dict[str, Any]:
    """Compare direct and exact streamed SIGReg on identical ChemFM microbatches."""
    model.eval()
    device = model.device
    selected = [rows[index] for index in indices]
    chunks = [
        _device_batch(collator(selected[start:start + PHYSICAL_BATCH]), device)
        for start in range(0, len(selected), PHYSICAL_BATCH)
    ]

    direct_sigreg = SIGReg(seed=1907)
    model.zero_grad(set_to_none=True)
    # Materialize the small reference distribution while retaining each
    # ChemFM microbatch graph. This keeps padding and BF16 kernel shapes
    # identical to the recomputed streaming path.
    direct_chunk_states = [
        _stack_states(_endpoint_output(method, model, chunk)) for chunk in chunks
    ]
    direct_states = torch.cat(direct_chunk_states, dim=1)
    direct_loss = direct_sigreg(direct_states)
    representation_dtype = direct_states.dtype
    direct_rep_gradient = torch.autograd.grad(
        direct_loss, direct_states, retain_graph=True
    )[0].detach().float()
    direct_loss_value = float(direct_loss.detach())
    direct_loss.backward()
    direct_parameter_gradients = _capture_gradients(model)
    direct_states_detached = direct_states.detach().float()
    del direct_chunk_states, direct_states, direct_loss

    streamed_sigreg = SIGReg(seed=1907)
    accumulator = streamed_sigreg.start_streaming(
        views=2,
        dimensions=direct_states_detached.size(-1),
        expected_samples=len(selected),
        device=device,
    )
    streamed_states: list[torch.Tensor] = []
    with torch.no_grad():
        for chunk in chunks:
            states = _stack_states(_endpoint_output(method, model, chunk))
            accumulator.update(states)
            streamed_states.append(states.detach().float())
    prepared = accumulator.finalize()
    streamed_states_detached = torch.cat(streamed_states, dim=1)
    streamed_rep_gradient = prepared.representation_gradients(
        streamed_states_detached
    ).to(representation_dtype).detach().float()

    model.zero_grad(set_to_none=True)
    for chunk in chunks:
        states = _stack_states(_endpoint_output(method, model, chunk))
        prepared.surrogate(states).backward()
    streamed_parameter_gradients = _capture_gradients(model)

    representation_difference = streamed_states_detached - direct_states_detached
    representation_gradient_difference = streamed_rep_gradient - direct_rep_gradient
    parameter_relative, parameter_max_abs, parameter_cosine = _global_relative_error(
        streamed_parameter_gradients, direct_parameter_gradients
    )
    model.zero_grad(set_to_none=True)
    model.train()
    return {
        "sample_count_per_view": len(selected),
        "physical_chunk_size": PHYSICAL_BATCH,
        "direct_loss": direct_loss_value,
        "streamed_loss": float(prepared.loss.detach()),
        "loss_abs_error": abs(float(prepared.loss.detach()) - direct_loss_value),
        "endpoint_max_abs_error": float(representation_difference.abs().max()),
        "endpoint_relative_l2_error": float(
            representation_difference.double().norm()
            / direct_states_detached.double().norm().clamp_min(1e-30)
        ),
        "representation_gradient_max_abs_error": float(
            representation_gradient_difference.abs().max()
        ),
        "representation_gradient_relative_l2_error": float(
            representation_gradient_difference.double().norm()
            / direct_rep_gradient.double().norm().clamp_min(1e-30)
        ),
        "parameter_gradient_max_abs_error": parameter_max_abs,
        "parameter_gradient_relative_l2_error": parameter_relative,
        "parameter_gradient_cosine": parameter_cosine,
    }


def _prepare_streaming_statistics(method, model, chunks):
    accumulator = None
    replay_states = []
    with torch.no_grad():
        for raw in chunks:
            replay_states.append(_rng_snapshot())
            batch = _device_batch(raw, model.device)
            states = _stack_states(_endpoint_output(method, model, batch))
            if accumulator is None:
                accumulator = method.sigreg.start_streaming(
                    views=2,
                    dimensions=states.size(-1),
                    expected_samples=SIGREG_BATCH,
                    device=states.device,
                )
            accumulator.update(states)
    if accumulator is None:
        raise RuntimeError("empty SIGReg calibration group")
    final_rng = _rng_snapshot()
    return accumulator.finalize(), replay_states, final_rng


def _component_gradient(
    *, component: str, model, method, chunks, prepared, replay_states, final_rng
) -> tuple[dict[str, torch.Tensor], list[float]]:
    model.zero_grad(set_to_none=True)
    values: list[float] = []
    for chunk_index, raw in enumerate(chunks):
        _restore_rng(replay_states[chunk_index])
        batch = _device_batch(raw, model.device)
        output = _endpoint_output(method, model, batch)
        if component == "ntp":
            loss = output.native_loss / len(chunks)
            values.append(float(output.native_loss.detach()))
        elif component == "cosine":
            if output.jepa_loss is None:
                raise RuntimeError("missing cosine JEPA loss")
            loss = output.jepa_loss / len(chunks)
            values.append(float(output.jepa_loss.detach()))
        elif component == "sigreg":
            states = _stack_states(output)
            loss = prepared.surrogate(states)
        else:
            raise ValueError(component)
        loss.backward()
    _restore_rng(final_rng)
    gradients = _capture_gradients(model)
    model.zero_grad(set_to_none=True)
    return gradients, values


def _calibrate_group(*, model, method, chunks, group_index: int) -> dict[str, Any]:
    prepared, replay_states, final_rng = _prepare_streaming_statistics(
        method, model, chunks
    )
    vectors: dict[str, dict[str, torch.Tensor]] = {}
    component_values: dict[str, list[float]] = {}
    for component in ("ntp", "cosine", "sigreg"):
        vectors[component], component_values[component] = _component_gradient(
            component=component,
            model=model,
            method=method,
            chunks=chunks,
            prepared=prepared,
            replay_states=replay_states,
            final_rng=final_rng,
        )

    relative = relative_sigreg_coefficient(LEJEPA_TRADEOFF)
    raw_aux = _combine_vectors(vectors, {"cosine": 1.0, "sigreg": relative})
    applied_sigreg = _combine_vectors(vectors, {"sigreg": ACTUAL_LAMBDA * relative})
    applied_aux = _combine_vectors(
        vectors,
        {"cosine": ACTUAL_LAMBDA, "sigreg": ACTUAL_LAMBDA * relative},
    )
    total = _combine_vectors(
        vectors,
        {"ntp": 1.0, "cosine": ACTUAL_LAMBDA, "sigreg": ACTUAL_LAMBDA * relative},
    )
    token_count = sum(int(raw["attention_mask"].sum()) for raw in chunks)
    native_loss = sum(component_values["ntp"]) / len(chunks)
    cosine_loss = sum(component_values["cosine"]) / len(chunks)
    sigreg_loss = float(prepared.loss.detach())
    return {
        "group": group_index,
        "examples": SIGREG_BATCH,
        "tokens": token_count,
        "native_ntp_loss": native_loss,
        "cosine_jepa_loss": cosine_loss,
        "sigreg_loss": sigreg_loss,
        "raw_auxiliary_loss": cosine_loss + relative * sigreg_loss,
        "active_total_loss": native_loss + ACTUAL_LAMBDA * (
            cosine_loss + relative * sigreg_loss
        ),
        "gradient_norms": {
            "ntp_raw": _vector_stats(vectors, {"ntp": 1.0})["norm"],
            "cosine_raw": _vector_stats(vectors, {"cosine": 1.0})["norm"],
            "sigreg_raw": _vector_stats(vectors, {"sigreg": 1.0})["norm"],
            "sigreg_applied": _vector_stats(
                {"sigreg": applied_sigreg}, {"sigreg": 1.0}
            )["norm"],
            "auxiliary_raw": _vector_stats(
                {"aux": raw_aux}, {"aux": 1.0}
            )["norm"],
            "auxiliary_applied": _vector_stats(
                {"aux": applied_aux}, {"aux": 1.0}
            )["norm"],
            "combined_total_preclip": _vector_stats(
                {"total": total}, {"total": 1.0}
            )["norm"],
        },
        "gradient_cosines_with_ntp": {
            "cosine_jepa": _cosine(vectors["ntp"], vectors["cosine"]),
            "sigreg": _cosine(vectors["ntp"], vectors["sigreg"]),
            "combined_auxiliary": _cosine(vectors["ntp"], raw_aux),
        },
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values)


def _calibration_summary(groups: list[dict[str, Any]]) -> dict[str, Any]:
    norm_keys = groups[0]["gradient_norms"].keys()
    cosine_keys = groups[0]["gradient_cosines_with_ntp"].keys()
    means = {
        "native_ntp_loss": _mean(row["native_ntp_loss"] for row in groups),
        "cosine_jepa_loss": _mean(row["cosine_jepa_loss"] for row in groups),
        "sigreg_loss": _mean(row["sigreg_loss"] for row in groups),
        "gradient_norms": {
            key: _mean(row["gradient_norms"][key] for row in groups)
            for key in norm_keys
        },
        "gradient_cosines_with_ntp": {
            key: _mean(
                row["gradient_cosines_with_ntp"][key]
                for row in groups
                if row["gradient_cosines_with_ntp"][key] is not None
            )
            for key in cosine_keys
        },
    }
    norms = means["gradient_norms"]
    means["ratios"] = {
        "applied_sigreg_to_ntp": norms["sigreg_applied"] / max(norms["ntp_raw"], 1e-30),
        "applied_auxiliary_to_ntp": norms["auxiliary_applied"] / max(norms["ntp_raw"], 1e-30),
        "total_to_ntp": norms["combined_total_preclip"] / max(norms["ntp_raw"], 1e-30),
    }
    return means


def _calibration_passes(equivalence: dict[str, Any], summary: dict[str, Any]) -> tuple[bool, list[str]]:
    failures = []
    if equivalence["representation_gradient_relative_l2_error"] > 5e-4:
        failures.append("streamed representation-gradient relative error exceeds 5e-4")
    if equivalence["parameter_gradient_relative_l2_error"] > 5e-3:
        failures.append("streamed parameter-gradient relative error exceeds 5e-3")
    if equivalence["parameter_gradient_cosine"] is None or equivalence["parameter_gradient_cosine"] < 0.999:
        failures.append("streamed/direct parameter-gradient cosine is below 0.999")
    flattened = [
        summary["native_ntp_loss"],
        summary["cosine_jepa_loss"],
        summary["sigreg_loss"],
        *summary["gradient_norms"].values(),
        *summary["gradient_cosines_with_ntp"].values(),
    ]
    if not all(math.isfinite(value) for value in flattened):
        failures.append("non-finite calibration value")
    if summary["ratios"]["applied_sigreg_to_ntp"] > 2.0:
        failures.append("applied SIGReg gradient exceeds 2x the NTP gradient")
    if summary["ratios"]["total_to_ntp"] > 5.0:
        failures.append("combined total gradient exceeds 5x the NTP gradient")
    return not failures, failures


def _smoke_test(
    *, model, method, collator, rows, permutation, updates: int
) -> list[dict[str, Any]]:
    set_seed(SEED)
    method.jepa_dropout_generator = torch.Generator().manual_seed(SEED)
    method.sigreg.global_step = 0
    model.train()
    trainable = [parameter for _, parameter in _trainable(model)]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=1e-4,
        betas=ADAM_BETAS,
        eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY,
        fused=False,
    )
    planned_updates = FULL_EPOCHS * (TRAIN_ROWS // SIGREG_BATCH)
    scheduler = get_scheduler(
        "cosine_with_min_lr",
        optimizer,
        num_warmup_steps=int(planned_updates * WARMUP_RATIO),
        num_training_steps=planned_updates,
        scheduler_specific_kwargs={"min_lr": MIN_LEARNING_RATE},
    )
    relative = relative_sigreg_coefficient(LEJEPA_TRADEOFF)
    optimizer.zero_grad(set_to_none=True)
    records = []
    for update in range(updates):
        selected = [rows[index] for index in permutation[update * SIGREG_BATCH:(update + 1) * SIGREG_BATCH]]
        chunks = [
            collator(selected[start:start + PHYSICAL_BATCH])
            for start in range(0, SIGREG_BATCH, PHYSICAL_BATCH)
        ]
        active = method.sample_jepa_activity(JEPA_RATIO)
        prepared = None
        replay_states = []
        final_rng = None
        statistics_seconds = 0.0
        if active:
            started = time.perf_counter()
            prepared, replay_states, final_rng = _prepare_streaming_statistics(
                method, model, chunks
            )
            torch.cuda.synchronize()
            statistics_seconds = time.perf_counter() - started

        native_values = []
        cosine_values = []
        token_count = 0
        started = time.perf_counter()
        for chunk_index, raw in enumerate(chunks):
            if active:
                _restore_rng(replay_states[chunk_index])
            token_count += int(raw["attention_mask"].sum())
            batch = _device_batch(raw, model.device)
            output = method(
                model,
                batch,
                k=0,
                jepa_weight=ACTUAL_LAMBDA if active else 0.0,
                native_weight=1.0,
                monitor_only=False,
                stop_gradient_target=False,
                sigreg_tradeoff=0.0,
                jepa_ratio=JEPA_RATIO,
                force_jepa_active=active,
            )
            loss = output.loss / len(chunks)
            if active:
                loss = loss + ACTUAL_LAMBDA * relative * prepared.surrogate(
                    _stack_states(output)
                )
                cosine_values.append(float(output.jepa_loss.detach()))
            native_values.append(float(output.native_loss.detach()))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite smoke loss at update {update + 1}")
            loss.backward()
        if active:
            _restore_rng(final_rng)
        torch.cuda.synchronize()
        gradient_seconds = time.perf_counter() - started

        learning_rate = optimizer.param_groups[0]["lr"]
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        gradients_finite = math.isfinite(gradient_norm) and all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in trainable
        )
        started = time.perf_counter()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        optimizer_seconds = time.perf_counter() - started

        native_loss = _mean(native_values)
        cosine_loss = _mean(cosine_values) if cosine_values else None
        sigreg_loss = float(prepared.loss.detach()) if prepared is not None else None
        auxiliary_loss = (
            cosine_loss + relative * sigreg_loss if active else 0.0
        )
        records.append({
            "update": update + 1,
            "jepa_sigreg_active": active,
            "native_ntp_loss": native_loss,
            "cosine_jepa_loss": cosine_loss,
            "sigreg_loss": sigreg_loss,
            "auxiliary_loss_before_outer_weight": auxiliary_loss,
            "total_loss": native_loss + ACTUAL_LAMBDA * auxiliary_loss,
            "preclip_total_gradient_norm": gradient_norm,
            "gradients_finite": gradients_finite,
            "learning_rate": learning_rate,
            "examples": SIGREG_BATCH,
            "tokens": token_count,
            "sigreg_statistics_seconds": statistics_seconds,
            "gradient_forward_backward_seconds": gradient_seconds,
            "optimizer_seconds": optimizer_seconds,
            "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
        })
    return records


def _linear_slope(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    x_mean = (len(values) - 1) / 2.0
    y_mean = _mean(values)
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    return sum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(values)
    ) / denominator


def _smoke_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    active = [row for row in records if row["jepa_sigreg_active"]]
    native = [row["native_ntp_loss"] for row in records]
    sigreg = [row["sigreg_loss"] for row in active]
    cosine = [row["cosine_jepa_loss"] for row in active]
    gradients = [row["preclip_total_gradient_norm"] for row in records]
    width = min(4, len(records) // 2)
    active_width = min(3, len(active) // 2)
    return {
        "updates": len(records),
        "active_updates": len(active),
        "realized_auxiliary_activity_fraction": len(active) / len(records),
        "native_first_window_mean": _mean(native[:width]),
        "native_last_window_mean": _mean(native[-width:]),
        "native_linear_slope_per_update": _linear_slope(native),
        "sigreg_first_active_window_mean": _mean(sigreg[:active_width]),
        "sigreg_last_active_window_mean": _mean(sigreg[-active_width:]),
        "sigreg_linear_slope_per_active_update": _linear_slope(sigreg),
        "cosine_first_active_window_mean": _mean(cosine[:active_width]),
        "cosine_last_active_window_mean": _mean(cosine[-active_width:]),
        "cosine_linear_slope_per_active_update": _linear_slope(cosine),
        "gradient_norm_min": min(gradients),
        "gradient_norm_max": max(gradients),
        "gradient_norm_mean": _mean(gradients),
        "all_gradients_finite": all(row["gradients_finite"] for row in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-16 SIGReg calibration and short smoke test")
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--calibration-groups", type=int, default=3)
    parser.add_argument("--smoke-updates", type=int, default=16)
    parser.add_argument(
        "--reuse-calibration",
        action="store_true",
        help="Reuse no-step calibration groups already stored at --output; equivalence is always rerun",
    )
    args = parser.parse_args()
    if not 10 <= args.smoke_updates <= 20:
        raise ValueError("smoke test must contain 10-20 optimizer updates")
    if args.calibration_groups < 2:
        raise ValueError("at least two representative calibration groups are required")

    if not torch.cuda.is_available():
        raise EnvironmentError("real ChemFM preflight requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    set_seed(SEED)

    rows = read_rows("uspto_mit_synthesis", path=args.train_manifest.resolve())
    if len(rows) != TRAIN_ROWS:
        raise ValueError(f"expected the established {TRAIN_ROWS}-row manifest, got {len(rows)}")
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab_size = len(tokenizer)
    predictor_ids = add_predictor_tokens(tokenizer)
    collator = ReactionCollator(tokenizer, task="forward")
    validate_serialization_endings(collator, rows, tokenizer.eos_token_id)
    model = load_lora_model(
        MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab_size
    ).cuda()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    method = CLMJEPA(
        predictor_ids,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        sigreg_seed=SEED,
    )
    permutation = torch.randperm(
        len(rows), generator=torch.Generator().manual_seed(SEED)
    ).tolist()

    started = time.perf_counter()
    equivalence = _real_equivalence_test(
        model=model,
        method=method,
        collator=collator,
        rows=rows,
        indices=permutation[:4],
    )
    reused_calibration_from = None
    if args.reuse_calibration:
        if not args.output.exists():
            raise FileNotFoundError("--reuse-calibration requires an existing --output artifact")
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        previous_config = previous["configuration"]
        if (
            previous_config["sigreg_statistical_batch_size_per_view"] != SIGREG_BATCH
            or previous_config["lejepa_two_view_tradeoff"] != LEJEPA_TRADEOFF
            or previous["seed"] != SEED
        ):
            raise ValueError("existing calibration artifact does not match this preflight")
        calibration_groups = previous["calibration_groups"]
        calibration_summary = previous["calibration_summary"]
        reused_calibration_from = str(args.output.resolve())
    else:
        set_seed(SEED)
        method.sigreg.global_step = 0
        calibration_loader = DataLoader(
            [rows[index] for index in permutation[: args.calibration_groups * SIGREG_BATCH]],
            batch_size=PHYSICAL_BATCH,
            shuffle=False,
            collate_fn=collator,
        )
        iterator = iter(calibration_loader)
        calibration_groups = []
        chunks_per_group = SIGREG_BATCH // PHYSICAL_BATCH
        for group_index in range(args.calibration_groups):
            chunks = [next(iterator) for _ in range(chunks_per_group)]
            calibration_groups.append(
                _calibrate_group(
                    model=model,
                    method=method,
                    chunks=chunks,
                    group_index=group_index + 1,
                )
            )
        calibration_summary = _calibration_summary(calibration_groups)
    passed, failures = _calibration_passes(equivalence, calibration_summary)

    smoke_records = []
    smoke_summary = None
    if passed:
        model.zero_grad(set_to_none=True)
        smoke_records = _smoke_test(
            model=model,
            method=method,
            collator=collator,
            rows=rows,
            permutation=permutation,
            updates=args.smoke_updates,
        )
        smoke_summary = _smoke_summary(smoke_records)

    result = {
        "scope": "batch-16 SIGReg preflight only; no full training or generation evaluation",
        "checkpoint": str(MODEL_DIR.resolve()),
        "checkpoint_role": "base ChemFM-1B initialization used by the intended future run",
        "seed": SEED,
        "configuration": {
            "k": 0,
            "symmetric_jepa": True,
            "target_stop_gradient": False,
            "native_ntp": True,
            "physical_batch_size": PHYSICAL_BATCH,
            "sigreg_statistical_batch_size_per_view": SIGREG_BATCH,
            "source_representations": SIGREG_BATCH,
            "target_representations": SIGREG_BATCH,
            "lejepa_two_view_tradeoff": LEJEPA_TRADEOFF,
            "relative_sigreg_coefficient": relative_sigreg_coefficient(LEJEPA_TRADEOFF),
            "lambda_eff_expected_cosine": LAMBDA_EFF,
            "jepa_activity_probability": JEPA_RATIO,
            "active_outer_jepa_coefficient": ACTUAL_LAMBDA,
            "active_applied_sigreg_coefficient": ACTUAL_LAMBDA
            * relative_sigreg_coefficient(LEJEPA_TRADEOFF),
            "expected_applied_sigreg_coefficient": JEPA_RATIO
            * ACTUAL_LAMBDA
            * relative_sigreg_coefficient(LEJEPA_TRADEOFF),
            "learning_rate": 1e-4,
            "optimizer": "AdamW",
            "planned_full_epochs_for_scheduler_only": FULL_EPOCHS,
            "planned_updates_per_epoch": TRAIN_ROWS // SIGREG_BATCH,
            "planned_total_updates": FULL_EPOCHS * (TRAIN_ROWS // SIGREG_BATCH),
            "smoke_optimizer_updates": args.smoke_updates,
        },
        "coefficient_mapping": (
            "LeJEPA uses (1-lambda)*L_cos + lambda*L_SIGReg. Dividing by "
            "(1-lambda) preserves this project's existing cosine strength, giving "
            "L_native + active*2*[L_cos + (0.01/0.99)*L_SIGReg]."
        ),
        "optimizer_semantics": {
            "established_examples_per_update": ESTABLISHED_EFFECTIVE_BATCH,
            "exact_sigreg16_examples_per_update": SIGREG_BATCH,
            "established_updates_per_epoch": TRAIN_ROWS // ESTABLISHED_EFFECTIVE_BATCH,
            "exact_sigreg16_updates_per_epoch": TRAIN_ROWS // SIGREG_BATCH,
            "update_cadence_fold_change": (TRAIN_ROWS // SIGREG_BATCH)
            / (TRAIN_ROWS // ESTABLISHED_EFFECTIVE_BATCH),
            "why_unavoidable": (
                "All 16 endpoint representations and their exact SIGReg VJP must be "
                "evaluated at one fixed parameter snapshot. Stepping after eight would "
                "make the other endpoints stale or mix parameter states."
            ),
            "later_matched_native_control_required": True,
        },
        "equivalence": equivalence,
        "calibration_groups": calibration_groups,
        "calibration_summary": calibration_summary,
        "reused_calibration_from": reused_calibration_from,
        "calibration_passed": passed,
        "calibration_failures": failures,
        "smoke_records": smoke_records,
        "smoke_summary": smoke_summary,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "calibration_passed": passed,
        "calibration_failures": failures,
        "calibration_summary": calibration_summary,
        "equivalence": equivalence,
        "smoke_summary": smoke_summary,
    }, indent=2))


if __name__ == "__main__":
    main()
