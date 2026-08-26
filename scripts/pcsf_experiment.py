"""Reference-cache, calibration, and frozen diagnostics for the PCSF experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import set_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from chemfm import (  # noqa: E402
    MODEL_DIR, TOKENIZER_DIR, ReactionCollator,
    load_lora_model, load_reaction_tokenizer,
)
from jepa import CLMJEPA, add_predictor_tokens  # noqa: E402
from historical_pcsf import (  # noqa: E402
    PairCenterSpreadFloor,
    pair_center_standard_deviation, pair_center_variance, pair_centers,
)
from metrics import effective_rank  # noqa: E402
from train import (  # noqa: E402
    file_sha256, load_adapter_checkpoint, reaction_row_fingerprint, read_rows,
    validate_serialization_endings,
)

SEED = 533
TRAIN_MANIFEST = (
    ROOT / "data" / "clm_jepa_uspto_mit_pilot_1280" / "uspto_mit_train.csv"
)
NATIVE_E4 = (
    ROOT / "runs" / "sigreg_batch16_pilot" / "matched_b4"
    / "native_checkpoints" / "epoch_4"
)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_runtime(checkpoint: Path | None):
    set_seed(SEED)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab_size = len(tokenizer)
    predictor_ids = add_predictor_tokens(tokenizer)
    collator = ReactionCollator(tokenizer, task="forward")
    model = load_lora_model(
        MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab_size,
        attn_implementation="sdpa",
    ).cuda()
    if checkpoint is not None:
        load_adapter_checkpoint(model, checkpoint.resolve())
    model.eval()
    method = CLMJEPA(
        predictor_ids, tokenizer.eos_token_id, tokenizer.pad_token_id,
        sigreg_seed=SEED, optimized_native_logits=True,
    )
    return tokenizer, collator, model, method


def tensor_batch(raw: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in raw.items() if torch.is_tensor(value)
    }


def endpoint_forward(method, model, raw):
    return method(
        model, tensor_batch(raw, model.device), k=0, jepa_weight=0.0,
        native_weight=1.0, monitor_only=True, stop_gradient_target=False,
        jepa_loss_type="mse", sigreg_tradeoff=0.0, jepa_ratio=1.0,
        force_jepa_active=True, endpoint_only=True,
    )


def collect_endpoints(
    rows: list[dict[str, str]], checkpoint: Path | None, batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    tokenizer, collator, model, method = load_runtime(checkpoint)
    validate_serialization_endings(collator, rows, tokenizer.eos_token_id)
    loader = DataLoader(rows, batch_size=batch_size, shuffle=False, collate_fn=collator)
    source_states, target_states = [], []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for raw in loader:
            output = endpoint_forward(method, model, raw)
            source_states.append(output.source_states.float().cpu())
            target_states.append(output.target_states.float().cpu())
    result = torch.cat(source_states), torch.cat(target_states)
    provenance = {
        "checkpoint": None if checkpoint is None else str(checkpoint.resolve()),
        "examples": len(rows),
        "batch_size": batch_size,
        "seconds": time.perf_counter() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "readout": "k=0 source EOS and target EOS",
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
    }
    del model
    torch.cuda.empty_cache()
    return result[0], result[1], provenance


def geometry(source: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    center = pair_centers(source, target)

    def side(values: torch.Tensor, prefix: str) -> dict[str, float]:
        values = values.float()
        mean = values.mean(0)
        return {
            f"{prefix}_variance": float(values.var(0, unbiased=False).mean()),
            f"{prefix}_effective_rank": float(effective_rank(values)),
            f"{prefix}_mean_direction_energy": float(
                mean.square().sum() / values.square().sum(1).mean().clamp_min(1e-30)
            ),
            f"{prefix}_embedding_norm_mean": float(values.norm(dim=1).mean()),
            f"{prefix}_mean_vector_norm": float(mean.norm()),
        }

    return {
        **side(source, "source"),
        **side(target, "target"),
        "pair_center_variance_unbiased": float(pair_center_variance(center)),
        "pair_center_sigma_unbiased": float(pair_center_standard_deviation(center)),
        "pair_mse": float(F.mse_loss(source.float(), target.float())),
    }


def shuffled_logical_batches(
    rows: list[dict[str, str]], collator, *, epochs: int = 4,
) -> list[list[list[int]]]:
    indexed = [dict(row, pcsf_reference_index=str(index)) for index, row in enumerate(rows)]
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        indexed, batch_size=4, shuffle=True, generator=generator, collate_fn=collator,
    )
    result = []
    for _ in range(epochs):
        physical = [raw["pcsf_reference_indices"].tolist() for raw in loader]
        result.append([
            [index for chunk in physical[start:start + 4] for index in chunk]
            for start in range(0, len(physical), 4)
        ])
    return result


def command_reference(args) -> None:
    rows = read_rows("uspto_mit_synthesis", path=args.train_manifest)
    source, target, provenance = collect_endpoints(rows, args.checkpoint, args.batch_size)
    payload = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "reference": "matched native ChemFM endpoint",
        "train_manifest": str(args.train_manifest.resolve()),
        "train_manifest_sha256": file_sha256(args.train_manifest),
        "row_fingerprints": [reaction_row_fingerprint(row) for row in rows],
        "pair_centers": pair_centers(source, target).cpu(),
        "source_states": source,
        "target_states": target,
        "geometry": geometry(source, target),
        "provenance": provenance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "sha256": file_sha256(args.output),
        "geometry": payload["geometry"],
        "provenance": provenance,
    }), flush=True)


def parse_checkpoint(specification: str) -> tuple[str, Path | None]:
    if "=" not in specification:
        raise ValueError("checkpoint specifications must be LABEL=PATH or LABEL=base")
    label, raw = specification.split("=", 1)
    if not label:
        raise ValueError("checkpoint label cannot be empty")
    return label, None if raw == "base" else Path(raw)


def command_measure(args) -> None:
    rows = read_rows("uspto_mit_synthesis", path=args.train_manifest)
    reference = torch.load(args.reference_cache, map_location="cpu", weights_only=False)
    if reference["row_fingerprints"] != [reaction_row_fingerprint(row) for row in rows]:
        raise ValueError("reference cache does not match measurement rows")
    reference_centers = reference["pair_centers"].float()
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    add_predictor_tokens(tokenizer)
    collator = ReactionCollator(tokenizer, task="forward")
    logical_batches = shuffled_logical_batches(rows, collator)
    states = {}
    for specification in args.checkpoint:
        label, checkpoint = parse_checkpoint(specification)
        source, target, provenance = collect_endpoints(rows, checkpoint, args.batch_size)
        centers = pair_centers(source, target)
        epoch_batch_ratios = []
        for batches in logical_batches:
            current_sigmas, reference_sigmas = [], []
            for indices in batches:
                index = torch.tensor(indices)
                current_sigmas.append(float(pair_center_standard_deviation(centers[index])))
                reference_sigmas.append(float(pair_center_standard_deviation(reference_centers[index])))
            ratios = [a / b for a, b in zip(current_sigmas, reference_sigmas)]
            epoch_batch_ratios.append({
                "mean": statistics.fmean(ratios),
                "median": statistics.median(ratios),
                "min": min(ratios),
                "max": max(ratios),
                "fraction_below_rho": (
                    None if args.rho is None
                    else sum(value < args.rho for value in ratios) / len(ratios)
                ),
            })
        cache_path = args.cache_dir / f"{label}.pt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"source_states": source, "target_states": target}, cache_path)
        states[label] = {
            "checkpoint": None if checkpoint is None else str(checkpoint.resolve()),
            "geometry": geometry(source, target),
            "global_pair_center_sigma_ratio_to_reference": float(
                pair_center_standard_deviation(centers)
                / pair_center_standard_deviation(reference_centers)
            ),
            "training_order_logical_batch_ratio_by_epoch": epoch_batch_ratios,
            "cache": str(cache_path.resolve()),
            "provenance": provenance,
        }
        print(json.dumps({"completed": label, **states[label]}), flush=True)
    save_json(args.output, {
        "schema_version": 1,
        "train_manifest": str(args.train_manifest.resolve()),
        "train_manifest_sha256": file_sha256(args.train_manifest),
        "reference_cache": str(args.reference_cache.resolve()),
        "reference_cache_sha256": file_sha256(args.reference_cache),
        "rho_evaluated": args.rho,
        "states": states,
    })


def named_parameters(model, *, lora_only: bool) -> list[tuple[str, torch.nn.Parameter]]:
    result = [
        (name, parameter) for name, parameter in model.named_parameters()
        if parameter.requires_grad and (not lora_only or ".lora_" in name)
    ]
    if not result:
        raise RuntimeError("no parameters selected for PCSF gradient calibration")
    return result


def gradients(
    loss: torch.Tensor,
    parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    retain_graph: bool,
) -> dict[str, torch.Tensor]:
    selected = list(parameters)
    values = torch.autograd.grad(
        loss, [parameter for _, parameter in selected],
        retain_graph=retain_graph, allow_unused=True,
    )
    return {
        name: (
            torch.zeros_like(parameter, dtype=torch.float32, device="cpu")
            if value is None else value.detach().float().cpu()
        )
        for (name, parameter), value in zip(selected, values)
    }


def vector_norm(vector: dict[str, torch.Tensor]) -> float:
    return math.sqrt(sum(float(value.square().sum()) for value in vector.values()))


def vector_cosine(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> float | None:
    dot = sum(float((first[name] * second[name]).sum()) for name in first)
    norm = vector_norm(first) * vector_norm(second)
    return None if norm == 0 else dot / norm


def subset(vector: dict[str, torch.Tensor], suffix: str) -> dict[str, torch.Tensor]:
    return {name: value for name, value in vector.items() if suffix in name}


def command_gradients(args) -> None:
    rows = read_rows("uspto_mit_synthesis", path=args.train_manifest)
    reference = torch.load(args.reference_cache, map_location="cpu", weights_only=False)
    expected = [reaction_row_fingerprint(row) for row in rows]
    if reference["row_fingerprints"] != expected:
        raise ValueError("reference cache does not match calibration rows")
    reference_centers = reference["pair_centers"].float().cuda()
    indexed = [dict(row, pcsf_reference_index=str(index)) for index, row in enumerate(rows)]
    tokenizer, collator, model, method = load_runtime(args.checkpoint)
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        indexed, batch_size=16, shuffle=True, generator=generator, collate_fn=collator,
    )
    all_parameters = named_parameters(model, lora_only=False)
    lora_names = {name for name, _ in named_parameters(model, lora_only=True)}
    regularizer = PairCenterSpreadFloor(rho=args.rho, epsilon=args.epsilon)
    before = hashlib.sha256(b"".join(
        parameter.detach().cpu().float().numpy().tobytes()
        for _, parameter in all_parameters
    )).hexdigest()
    records = []
    for batch_index, raw in enumerate(loader):
        if batch_index == args.batches:
            break
        batch = tensor_batch(raw, model.device)
        output = method(
            model, batch, k=0, jepa_weight=0.0, native_weight=1.0,
            monitor_only=True, jepa_loss_type="mse", force_jepa_active=True,
            endpoint_only=True,
        )
        mse = F.mse_loss(output.source_states, output.target_states)
        refs = reference_centers.index_select(0, batch["pcsf_reference_indices"])
        pcsf_value, sigma, reference_sigma = regularizer(
            output.source_states, output.target_states, refs,
        )
        g_mse = gradients(mse, all_parameters, retain_graph=True)
        g_pcsf = gradients(pcsf_value, all_parameters, retain_graph=False)
        del output
        native = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        g_ntp = gradients(native.loss, all_parameters, retain_graph=False)

        scopes = {}
        for scope, names in (
            ("all_trainable", set(g_mse)),
            ("lora_ab", lora_names),
        ):
            mse_vector = {name: g_mse[name] for name in names}
            pcsf_vector = {name: g_pcsf[name] for name in names}
            ntp_vector = {name: g_ntp[name] for name in names}
            mse_norm = vector_norm(mse_vector)
            pcsf_norm = vector_norm(pcsf_vector)
            ntp_norm = vector_norm(ntp_vector)
            scopes[scope] = {
                "mse_norm": mse_norm,
                "pcsf_norm": pcsf_norm,
                "ntp_norm": ntp_norm,
                "raw_pcsf_to_mse_norm": pcsf_norm / max(mse_norm, 1e-30),
                "beta_for_equal_pcsf_mse_norm": mse_norm / max(pcsf_norm, 1e-30),
                "raw_pcsf_ntp_cosine": vector_cosine(pcsf_vector, ntp_vector),
                "mse_ntp_cosine": vector_cosine(mse_vector, ntp_vector),
                "weighted_pcsf_to_mse_norm": (
                    None if args.beta is None
                    else args.beta * pcsf_norm / max(mse_norm, 1e-30)
                ),
                "weighted_pcsf_ntp_norm": (
                    None if args.beta is None
                    else args.beta * pcsf_norm / max(ntp_norm, 1e-30)
                ),
            }
        records.append({
            "batch": batch_index,
            "indices": batch["pcsf_reference_indices"].cpu().tolist(),
            "mse_loss": float(mse.detach()),
            "pcsf_loss": float(pcsf_value.detach()),
            "pair_center_sigma": float(sigma.detach()),
            "reference_pair_center_sigma": float(reference_sigma.detach()),
            "spread_ratio": float((sigma / reference_sigma).detach()),
            "above_floor": bool(pcsf_value.detach() > 0),
            "ntp_loss": float(native.loss.detach()),
            "scopes": scopes,
        })
        print(json.dumps(records[-1]), flush=True)
    after = hashlib.sha256(b"".join(
        parameter.detach().cpu().float().numpy().tobytes()
        for _, parameter in all_parameters
    )).hexdigest()
    if before != after:
        raise RuntimeError("gradient calibration changed model parameters")
    recommendations = {}
    for scope in ("all_trainable", "lora_ab"):
        values = [
            row["scopes"][scope]["beta_for_equal_pcsf_mse_norm"] for row in records
            if row["above_floor"]
        ]
        recommendations[scope] = {
            "median_beta": statistics.median(values),
            "mean_beta": statistics.fmean(values),
            "min_beta": min(values),
            "max_beta": max(values),
        } if values else None
    save_json(args.output, {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "reference_cache": str(args.reference_cache.resolve()),
        "rho": args.rho,
        "epsilon": args.epsilon,
        "beta_evaluated": args.beta,
        "batches": records,
        "beta_recommendation": recommendations,
        "parameter_fingerprint_unchanged": before == after,
        "optimizer_constructed": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    reference = subparsers.add_parser("reference")
    reference.add_argument("--train-manifest", type=Path, default=TRAIN_MANIFEST)
    reference.add_argument("--checkpoint", type=Path, default=NATIVE_E4)
    reference.add_argument("--batch-size", type=int, default=16)
    reference.add_argument("--output", type=Path, required=True)

    measure = subparsers.add_parser("measure")
    measure.add_argument("--train-manifest", type=Path, default=TRAIN_MANIFEST)
    measure.add_argument("--reference-cache", type=Path, required=True)
    measure.add_argument("--checkpoint", action="append", required=True)
    measure.add_argument("--batch-size", type=int, default=16)
    measure.add_argument("--rho", type=float)
    measure.add_argument("--cache-dir", type=Path, required=True)
    measure.add_argument("--output", type=Path, required=True)

    gradient = subparsers.add_parser("gradients")
    gradient.add_argument("--train-manifest", type=Path, default=TRAIN_MANIFEST)
    gradient.add_argument("--reference-cache", type=Path, required=True)
    gradient.add_argument("--checkpoint", type=Path, required=True)
    gradient.add_argument("--rho", type=float, required=True)
    gradient.add_argument("--beta", type=float)
    gradient.add_argument("--epsilon", type=float, default=1e-8)
    gradient.add_argument("--batches", type=int, default=4)
    gradient.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "reference":
        command_reference(args)
    elif args.command == "measure":
        command_measure(args)
    else:
        command_gradients(args)


if __name__ == "__main__":
    main()
