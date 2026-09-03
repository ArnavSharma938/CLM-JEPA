#!/usr/bin/env python
"""Extract and analyze frozen ChemFM states for the Geodesic Mechanism Audit.

The expensive transformer pass is cached once per checkpoint.  All geometry is
then recomputable from compact BF16 state shards and FP32 LM-head weights.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.chemfm import (
    END, MODEL_DIR, PRODUCT_START, REACTANT_START, TOKENIZER_DIR,
    load_lora_model, load_reaction_tokenizer,
)
from src.frozen_geometry import DEFAULT_PANEL, annotate_example, read_panel
from src.geodesic_audit import (
    acceleration_decomposition,
    categorical_fisher_squared,
    chord_coordinates,
    cosine,
    curvature_removal_intervention,
    estimate_piecewise_change_point,
    fisher_rao_distance,
    fisher_rao_path_efficiency,
    fisher_rao_triangle_excess,
    gold_logprob_gradient,
    local_tangent_acceleration,
    matched_geodesic_displacement,
    multiscale_turning,
    optimal_ray_residual,
    predictive_sensitivity,
    released_objective_anatomy,
    tangent_autocorrelation,
    tube_scale_space,
)
from src.jepa import add_predictor_tokens
from src.stp_representation_analysis import CheckpointSpec, checkpoint_specs, validate_checkpoint_specs
from src.train import load_adapter_checkpoint


DEFAULT_OUTPUT = ROOT / "runs" / "geodesic_mechanism_audit"
PRIMARY_KEYS = {
    *(f"native_r8_s{s}" for s in (533, 917, 1301)),
    *(f"released_r8_l0.02_s{s}" for s in (533, 917, 1301)),
    *(f"paper_r8_l0.02_s{s}" for s in (533, 917)),
}
DEPTHS = (0, 6, 16, 21, 22)
DEPTH_LABELS = ("embedding", "layer_6", "layer_16", "layer_21", "final_post_norm")
BOOTSTRAP_SEED = 20260902


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=json_default) + "\n", encoding="utf-8")


def write_jsonl_gz(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for row in rows:
            handle.write(json.dumps(row, default=json_default, separators=(",", ":")) + "\n")
            count += 1
    return count


class JsonlGzipWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = gzip.open(path, "wt", encoding="utf-8", compresslevel=1)
        self.count = 0

    def write(self, row: dict) -> None:
        self.handle.write(json.dumps(row, default=json_default, separators=(",", ":")) + "\n")
        self.count += 1

    def close(self) -> None:
        self.handle.close()


def build_gold_examples(tokenizer, panel: Path) -> list:
    return [
        annotate_example(tokenizer, row, index, "prespecified_256")
        for index, row in enumerate(read_panel(panel))
    ]


def find_llama(model):
    candidates = [
        model,
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "model", None),
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers") and hasattr(candidate, "norm"):
            return candidate
    for module in model.modules():
        if hasattr(module, "layers") and hasattr(module, "norm") and hasattr(module, "embed_tokens"):
            return module
    raise RuntimeError("could not locate the underlying LlamaModel")


class SelectedStateCapture:
    """Hooks exact states without `output_hidden_states=True` materialization."""

    def __init__(self, model):
        llama = find_llama(model)
        self.values: dict[str, torch.Tensor] = {}
        self.handles = []

        def capture(name):
            def hook(_module, _inputs, output):
                value = output[0] if isinstance(output, tuple) else output
                self.values[name] = value
            return hook

        self.handles.append(llama.embed_tokens.register_forward_hook(capture("embedding")))
        self.handles.append(llama.layers[5].register_forward_hook(capture("layer_6")))
        self.handles.append(llama.layers[15].register_forward_hook(capture("layer_16")))
        self.handles.append(llama.layers[20].register_forward_hook(capture("layer_21")))
        self.handles.append(llama.layers[-1].register_forward_hook(capture("post_last_pre_norm")))
        self.handles.append(llama.norm.register_forward_hook(capture("final_post_norm")))

    def clear(self):
        self.values.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()


def spec_limit(spec: CheckpointSpec) -> int:
    return 256 if spec.key in PRIMARY_KEYS else 64


def load_model_for_spec(spec: CheckpointSpec, tokenizer, chemfm_vocab_size: int, device: str):
    model = load_lora_model(
        MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab_size,
        attention_dropout=0.0, attn_implementation="sdpa",
        lora_rank=spec.rank, lora_alpha=spec.alpha,
    ).to(device)
    load_adapter_checkpoint(model, ROOT / spec.checkpoint)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def extract_gold(args) -> None:
    output = args.output.resolve()
    cache = output / "cache" / "gold_states"
    cache.mkdir(parents=True, exist_ok=True)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab_size = len(tokenizer)
    add_predictor_tokens(tokenizer)
    examples = build_gold_examples(tokenizer, args.panel.resolve())
    specs = checkpoint_specs()
    validate_checkpoint_specs(specs)
    if args.keys:
        wanted = set(args.keys.split(","))
        specs = [spec for spec in specs if spec.key in wanted]
        missing = wanted - {spec.key for spec in specs}
        if missing:
            raise ValueError(f"unknown checkpoint keys: {sorted(missing)}")
    metadata = {
        "type": "geodesic_audit_gold_state_cache", "git_commit": git_commit(),
        "panel": str(args.panel.resolve()), "panel_sha256": sha256(args.panel.resolve()),
        "device": args.device, "torch": torch.__version__, "python": platform.python_version(),
        "depths": dict(zip(DEPTH_LABELS, DEPTHS)), "checkpoints": [],
    }
    specs = sorted(specs, key=lambda item: (item.rank, item.key))
    model = None
    capture = None
    current_rank = None
    for index, spec in enumerate(specs, 1):
        destination = cache / f"{spec.key}.pt"
        if destination.exists() and not args.overwrite:
            print(json.dumps({"stage": "cache_reused", "key": spec.key}), flush=True)
            continue
        if current_rank != spec.rank:
            if capture is not None:
                capture.close()
            del model, capture
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = load_lora_model(
                MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab_size,
                attention_dropout=0.0, attn_implementation="sdpa",
                lora_rank=spec.rank, lora_alpha=spec.alpha,
            ).to(args.device).eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            capture = SelectedStateCapture(model)
            current_rank = spec.rank
        load_adapter_checkpoint(model, ROOT / spec.checkpoint)
        lm_weight = model.get_output_embeddings().weight.detach().float().cpu()
        records = []
        limit = spec_limit(spec)
        ordered = sorted(examples[:limit], key=lambda example: len(example.input_ids))
        start = time.perf_counter()
        if str(args.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for batch_start in range(0, len(ordered), args.batch_size):
                batch = ordered[batch_start:batch_start + args.batch_size]
                maximum = max(len(example.input_ids) for example in batch)
                ids = torch.zeros((len(batch), maximum), dtype=torch.long, device=args.device)
                mask = torch.zeros_like(ids, dtype=torch.bool)
                for row, example in enumerate(batch):
                    length = len(example.input_ids)
                    ids[row, :length] = torch.tensor(example.input_ids, device=args.device)
                    mask[row, :length] = True
                capture.clear()
                model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
                required = (*DEPTH_LABELS, "post_last_pre_norm")
                if any(name not in capture.values for name in required):
                    raise RuntimeError(f"missing capture: {set(required) - set(capture.values)}")
                for row, example in enumerate(batch):
                    length = len(example.input_ids)
                    source = [token.index for token in example.tokens if token.segment == "source"]
                    target = [token.index for token in example.tokens if token.segment == "target"]
                    records.append({
                        "panel_index": example.panel_index,
                        "reaction_identity": example.reaction_identity,
                        "source": example.source, "target": example.target,
                        "input_ids": torch.tensor(example.input_ids, dtype=torch.int32),
                        "source_indices": torch.tensor(source, dtype=torch.int16),
                        "target_indices": torch.tensor(target, dtype=torch.int16),
                        "states": torch.stack([
                            capture.values[name][row, :length].detach().to(torch.bfloat16).cpu()
                            for name in DEPTH_LABELS
                        ]),
                        "post_last_pre_norm": capture.values["post_last_pre_norm"][row, :length].detach().to(torch.bfloat16).cpu(),
                    })
                del ids, mask
        seconds = time.perf_counter() - start
        records.sort(key=lambda record: record["panel_index"])
        payload = {
            "spec": spec.__dict__, "depth_labels": DEPTH_LABELS,
            "records": records, "lm_head": lm_weight,
            "checkpoint_sha256": sha256(ROOT / spec.checkpoint / "USPTO-MIT-Synthesis" / "adapter_model.safetensors"),
        }
        torch.save(payload, destination)
        record = {
            "key": spec.key, "examples": limit, "seconds": seconds,
            "examples_per_second": limit / seconds,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()) if str(args.device).startswith("cuda") else 0,
            "cache_bytes": destination.stat().st_size,
        }
        metadata["checkpoints"].append(record)
        print(json.dumps({"stage": "checkpoint_complete", "index": index, "total": len(specs), **record}), flush=True)
        del lm_weight, records, payload
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if capture is not None:
        capture.close()
    write_json(output / "gold_extraction_metadata.json", metadata)


def trajectory_from_record(record: dict, layer_index: int, segment: str, device: str):
    states = record["states"][layer_index].float().to(device)
    source = record["source_indices"].long().tolist()
    target = record["target_indices"].long().tolist()
    if segment == "source":
        positions = [source[0] - 1, *source]
        return states[positions], positions
    if segment == "product":
        positions = [target[0] - 1, *target]
        return states[positions], positions
    if segment == "cross":
        source_path = states[[source[0] - 1, *source]]
        target_path = states[target] - states[target[0] - 1] + states[source[-1]]
        return torch.cat((source_path, target_path)), [source[0] - 1, *source, *target]
    raise ValueError(segment)


def aggregate_tube_rows(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["checkpoint"], row["layer"], row["segment"], row["span_length"])].append(row)
    output = []
    for key, values in groups.items():
        checkpoint, layer, segment, length = key
        result = {"checkpoint": checkpoint, "layer": layer, "segment": segment, "span_length": length, "reactions": len(values)}
        for metric in ("mean", "rms", "maximum", "p90", "p95", "monotonicity_violation", "fraction_gt_0.05", "fraction_gt_0.1", "fraction_gt_0.2", "fraction_gt_0.5"):
            data = np.asarray([value[metric] for value in values], dtype=np.float64)
            result[metric] = float(np.mean(data))
            result[f"{metric}_median"] = float(np.median(data))
        output.append(result)
    return sorted(output, key=lambda x: (x["checkpoint"], x["layer"], x["segment"], x["span_length"]))


def fisher_inner(first: torch.Tensor, second: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    first_mean = (p * first).sum(-1)
    second_mean = (p * second).sum(-1)
    return (p * first * second).sum(-1) - first_mean * second_mean


def functional_metrics(path: torch.Tensor, weight: torch.Tensor) -> dict:
    logits = path @ weight.T
    return functional_metrics_from_logits(path, logits)


def functional_metrics_from_logits(path: torch.Tensor, logits: torch.Tensor) -> dict:
    """Function-space geometry from precomputed FP32 logits.

    Keeping this separate lets candidate analysis issue one dense batched LM-head
    multiplication and then perform the many small reductions on CPU.  The
    equations and FP32 evaluation are identical to ``functional_metrics``.
    """
    probabilities = logits.softmax(-1)
    euclidean_efficiency = float((path[-1] - path[0]).norm() / (path[1:] - path[:-1]).norm(dim=-1).sum().clamp_min(1e-12))
    fr_efficiency = float(fisher_rao_path_efficiency(probabilities))
    if len(path) > 2:
        before = logits[1:-1] - logits[:-2]
        after = logits[2:] - logits[1:-1]
        p = probabilities[1:-1]
        numerator = fisher_inner(before, after, p)
        denominator = (categorical_fisher_squared(before, p) * categorical_fisher_squared(after, p)).clamp_min(1e-20).sqrt()
        fisher_curvature = float((1 - numerator / denominator).mean())
    else:
        fisher_curvature = math.nan
    return {
        "euclidean_path_efficiency": euclidean_efficiency,
        "fisher_path_efficiency": fr_efficiency,
        "fisher_local_curvature": fisher_curvature,
        "probabilities": probabilities,
        "logits": logits,
    }


def compact_multiscale_turning(path: torch.Tensor, max_scale: int = 32) -> list[dict]:
    path = path.float().detach().cpu().numpy()
    output = []
    for scale in range(1, min(max_scale, (len(path) - 1) // 2) + 1):
        before = path[scale:-scale] - path[:-2 * scale]
        after = path[2 * scale:] - path[scale:-scale]
        numerator = np.sum(before * after, axis=-1)
        denominator = np.linalg.norm(before, axis=-1) * np.linalg.norm(after, axis=-1)
        angles = np.arccos(np.clip(numerator / np.maximum(denominator, 1e-8), -1, 1))
        output.append({"scale": scale, "start_position": scale, "angles": angles.tolist()})
    return output


def deterministic_starts(n: int, length: int, maximum: int, salt: int) -> list[int]:
    count = n - length
    if count <= maximum:
        return list(range(count))
    rng = np.random.default_rng(BOOTSTRAP_SEED + salt * 1_000_003 + length)
    return sorted(rng.choice(count, maximum, replace=False).tolist())


def write_intervention_batch(
    writer: JsonlGzipWriter, *, checkpoint: str, record: dict,
    segment: str, path_states: torch.Tensor, positions: list[int],
    ids: torch.Tensor, weight: torch.Tensor,
) -> None:
    triples = []
    for length in range(2, len(path_states)):
        for s in deterministic_starts(len(path_states), length, 8, record["panel_index"]):
            t = s + length
            r = (s + t) // 2
            if positions[r] + 1 < len(ids):
                triples.append((length, s, r, t, positions[r] + 1))
    if not triples:
        return
    index = torch.tensor([[row[1], row[2], row[3]] for row in triples], device=path_states.device)
    h_s, h_r, h_t = (path_states[index[:, column]] for column in range(3))
    alpha, q, rho = chord_coordinates(h_s, h_r, h_t)
    parallel = alpha[:, None] * (h_t - h_s)
    gold = ids[torch.tensor([row[4] for row in triples], device=ids.device)]
    logits = h_r @ weight.T
    probabilities = logits.softmax(-1)
    gradient = weight[gold] - probabilities @ weight
    para_signed = (gradient * parallel).sum(-1)
    perp_signed = (gradient * q).sum(-1)
    para_cosine = para_signed / (gradient.norm(dim=-1) * parallel.norm(dim=-1)).clamp_min(1e-12)
    perp_cosine = perp_signed / (gradient.norm(dim=-1) * q.norm(dim=-1)).clamp_min(1e-12)
    base_logp_all = logits.log_softmax(-1)
    base_logp = base_logp_all.gather(1, gold[:, None]).squeeze(1)
    base_rank = (logits > logits.gather(1, gold[:, None])).sum(1) + 1
    masked = logits.clone()
    masked.scatter_(1, gold[:, None], -torch.inf)
    base_margin = logits.gather(1, gold[:, None]).squeeze(1) - masked.max(1).values
    base_entropy = -(probabilities * base_logp_all).sum(-1)
    base_topk = torch.topk(logits, min(10, logits.shape[1]), dim=1).indices
    path_probabilities = (path_states @ weight.T).softmax(-1)
    fr_excess = fisher_rao_triangle_excess(
        path_probabilities[index[:, 0]], path_probabilities[index[:, 1]],
        path_probabilities[index[:, 2]],
    )
    ray = optimal_ray_residual(h_s, h_r, h_t)

    variants = []
    variant_labels = []
    for gamma in (.1, .25, .5):
        changed = h_r - gamma * q
        variants.append(changed)
        variant_labels.append((gamma, False))
        restored = changed * (h_r.norm(dim=-1) / changed.norm(dim=-1).clamp_min(1e-12))[:, None]
        variants.append(restored)
        variant_labels.append((gamma, True))
    changed_states = torch.cat(variants, dim=0)
    changed_logits = changed_states @ weight.T
    n = len(triples)
    changed_logits = changed_logits.reshape(len(variants), n, -1)
    changed_logp_all = changed_logits.log_softmax(-1)
    changed_probabilities = changed_logits.softmax(-1)
    changed_gold_logits = changed_logits.gather(2, gold[None, :, None].expand(len(variants), -1, 1)).squeeze(2)
    changed_gold_logp = changed_logp_all.gather(2, gold[None, :, None].expand(len(variants), -1, 1)).squeeze(2)
    changed_rank = (changed_logits > changed_gold_logits[:, :, None]).sum(-1) + 1
    changed_masked = changed_logits.clone()
    changed_masked.scatter_(2, gold[None, :, None].expand(len(variants), -1, 1), -torch.inf)
    changed_margin = changed_gold_logits - changed_masked.max(-1).values
    changed_entropy = -(changed_probabilities * changed_logp_all).sum(-1)
    changed_topk = torch.topk(changed_logits, min(10, logits.shape[1]), dim=-1).indices

    # One device synchronization for all scalar output avoids hundreds of
    # thousands of per-value CUDA synchronizations in the JSON loop.
    cpu = {
        "rho": rho.detach().cpu().numpy(), "alpha": alpha.detach().cpu().numpy(),
        "ray": ray.detach().cpu().numpy(), "fr": fr_excess.detach().cpu().numpy(),
        "para_signed": para_signed.detach().cpu().numpy(),
        "para_cosine": para_cosine.detach().cpu().numpy(),
        "perp_signed": perp_signed.detach().cpu().numpy(),
        "perp_cosine": perp_cosine.detach().cpu().numpy(),
        "q_norm": q.norm(dim=-1).detach().cpu().numpy(),
        "parallel_norm": parallel.norm(dim=-1).detach().cpu().numpy(),
        "base_logp": base_logp.detach().cpu().numpy(),
        "base_rank": base_rank.detach().cpu().numpy(),
        "base_margin": base_margin.detach().cpu().numpy(),
        "base_entropy": base_entropy.detach().cpu().numpy(),
        "delta_logp": (changed_gold_logp - base_logp[None]).detach().cpu().numpy(),
        "delta_rank": (changed_rank - base_rank[None]).detach().cpu().numpy(),
        "delta_margin": (changed_margin - base_margin[None]).detach().cpu().numpy(),
        "delta_entropy": (changed_entropy - base_entropy[None]).detach().cpu().numpy(),
        "topk_changed": (changed_topk != base_topk[None]).any(-1).detach().cpu().numpy(),
    }

    for row_index, (length, s, r, t, _) in enumerate(triples):
        effects = []
        for variant, (gamma, restore) in enumerate(variant_labels):
            effects.append({
                "gamma": gamma, "norm_restored": restore,
                "delta_gold_log_probability": float(cpu["delta_logp"][variant, row_index]),
                "delta_gold_rank": int(cpu["delta_rank"][variant, row_index]),
                "delta_gold_margin": float(cpu["delta_margin"][variant, row_index]),
                "delta_entropy": float(cpu["delta_entropy"][variant, row_index]),
                "topk_changed": bool(cpu["topk_changed"][variant, row_index]),
            })
        writer.write({
            "checkpoint": checkpoint, "panel_index": record["panel_index"],
            "reaction_identity": record["reaction_identity"], "segment": segment,
            "span_length": length, "s": s, "r": r, "t": t,
            "rho": float(cpu["rho"][row_index]), "alpha": float(cpu["alpha"][row_index]),
            "ray_residual": float(cpu["ray"][row_index]), "fisher_triangle_excess": float(cpu["fr"][row_index]),
            "parallel_signed_sensitivity": float(cpu["para_signed"][row_index]),
            "parallel_cosine_sensitivity": float(cpu["para_cosine"][row_index]),
            "perpendicular_signed_sensitivity": float(cpu["perp_signed"][row_index]),
            "perpendicular_cosine_sensitivity": float(cpu["perp_cosine"][row_index]),
            "perpendicular_norm": float(cpu["q_norm"][row_index]),
            "parallel_norm": float(cpu["parallel_norm"][row_index]),
            "base_gold_log_probability": float(cpu["base_logp"][row_index]),
            "base_gold_rank": int(cpu["base_rank"][row_index]),
            "base_gold_margin": float(cpu["base_margin"][row_index]),
            "base_entropy": float(cpu["base_entropy"][row_index]),
            "interventions": effects,
        })


def write_anatomy_batch(writer: JsonlGzipWriter, checkpoint: str, record: dict, path: torch.Tensor) -> None:
    spans = []
    for length in range(1, len(path)):
        for s in deterministic_starts(len(path), length, 8, record["panel_index"] + 17):
            t = s + length
            if s != 0 or t != len(path) - 1:
                spans.append((length, s, t))
    index = torch.tensor([[row[1], row[2]] for row in spans], device=path.device)
    before = path[index[:, 0]] - path[0]
    patch = path[index[:, 1]] - path[index[:, 0]]
    after = path[-1] - path[index[:, 1]]
    values = released_objective_anatomy(before, patch, after)
    values = {name: value.detach().cpu().numpy() for name, value in values.items()}
    for row_index, (length, _, _) in enumerate(spans):
        writer.write({
            "checkpoint": checkpoint, "panel_index": record["panel_index"],
            "span_length": length,
            **{name: float(value[row_index]) for name, value in values.items()},
        })


def analyze_gold(args) -> None:
    output = args.output.resolve()
    cache_paths = sorted((output / "cache" / "gold_states").glob("*.pt"))
    requested_keys = {value.strip() for value in args.keys.split(",") if value.strip()}
    if requested_keys:
        cache_paths = [path for path in cache_paths if path.stem in requested_keys]
    if not cache_paths:
        raise FileNotFoundError("no gold-state caches; run extract-gold first")
    raw = output / "raw"
    writers = {
        "tube_reaction_rows": JsonlGzipWriter(raw / "tube_scale_space_by_reaction.jsonl.gz"),
        "trajectory_rows": JsonlGzipWriter(raw / "trajectory_metrics.jsonl.gz"),
        "persistence_rows": JsonlGzipWriter(raw / "tangent_persistence.jsonl.gz"),
        "turning_rows": JsonlGzipWriter(raw / "multiscale_turning.jsonl.gz"),
        "intervention_rows": JsonlGzipWriter(raw / "signal_noise_interventions.jsonl.gz"),
        "anatomy_rows": JsonlGzipWriter(raw / "released_objective_anatomy.jsonl.gz"),
    }
    tube_accumulators = defaultdict(lambda: defaultdict(float))
    start = time.perf_counter()
    for cache_index, path in enumerate(cache_paths, 1):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        spec = payload["spec"]
        key = spec["key"]
        weight = payload["lm_head"].float().to(args.device)
        analysis_records = payload["records"][:args.analysis_limit or None]
        for record in analysis_records:
            for layer_index, label in enumerate(payload["depth_labels"]):
                for segment in ("source", "product", "cross"):
                    path_states, positions = trajectory_from_record(record, layer_index, segment, args.device)
                    tube = tube_scale_space(path_states)
                    for row in tube:
                        tube_row = {
                            "checkpoint": key, "rank": spec["rank"], "formulation": spec["formulation"],
                            "lambda": spec["stp_lambda"], "seed": spec["seed"],
                            "panel_index": record["panel_index"], "reaction_identity": record["reaction_identity"],
                            "layer": label, "segment": segment, **row,
                        }
                        writers["tube_reaction_rows"].write(tube_row)
                        group = tube_accumulators[(key, label, segment, row["span_length"])]
                        group["reactions"] += 1
                        for metric in ("mean", "rms", "maximum", "p90", "p95", "monotonicity_violation", "fraction_gt_0.05", "fraction_gt_0.1", "fraction_gt_0.2", "fraction_gt_0.5"):
                            group[metric] += row[metric]
                    acceleration = acceleration_decomposition(path_states)
                    trajectory = {
                        "checkpoint": key, "rank": spec["rank"], "formulation": spec["formulation"],
                        "lambda": spec["stp_lambda"], "seed": spec["seed"],
                        "panel_index": record["panel_index"], "reaction_identity": record["reaction_identity"],
                        "layer": label, "segment": segment, "tokens": len(path_states),
                        "speed": float(acceleration["speed"].mean()),
                        "tangential_acceleration": float(acceleration["acceleration_parallel"].mean()),
                        "normal_acceleration": float(acceleration["acceleration_normal"].mean()),
                        "normalized_normal_acceleration": float(acceleration["normalized_normal"].mean()),
                    }
                    if label == "final_post_norm":
                        functional = functional_metrics(path_states, weight)
                        trajectory.update({key_: value for key_, value in functional.items() if not torch.is_tensor(value)})
                    writers["trajectory_rows"].write(trajectory)
                    for row in tangent_autocorrelation(path_states):
                        writers["persistence_rows"].write({"checkpoint": key, "panel_index": record["panel_index"], "layer": label, "segment": segment, **row})
                    # Full (position,scale) heatmaps are retained for the first 64
                    # reactions; other rows contribute to reduced summaries.
                    if record["panel_index"] < 64:
                        writers["turning_rows"].write({
                            "checkpoint": key, "panel_index": record["panel_index"],
                            "layer": label, "segment": segment,
                            "scales": compact_multiscale_turning(path_states),
                        })

                    if label != "final_post_norm" or segment == "cross":
                        continue
                    functional = functional_metrics(path_states, weight)
                    ids = record["input_ids"].long().to(args.device)
                    write_intervention_batch(
                        writers["intervention_rows"], checkpoint=key,
                        record=record, segment=segment, path_states=path_states,
                        positions=positions, ids=ids, weight=weight,
                    )
                # Released-objective anatomy uses the exact framing-excluded path.
                if label == "final_post_norm":
                    cross, _ = trajectory_from_record(record, layer_index, "cross", args.device)
                    write_anatomy_batch(writers["anatomy_rows"], key, record, cross)
        print(json.dumps({"stage": "gold_analysis_checkpoint", "index": cache_index, "total": len(cache_paths), "key": key}), flush=True)
        del payload, weight
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    aggregate = []
    for (checkpoint, layer, segment, length), values in tube_accumulators.items():
        count = values["reactions"]
        row = {"checkpoint": checkpoint, "layer": layer, "segment": segment, "span_length": length, "reactions": int(count)}
        for metric, total in values.items():
            if metric != "reactions":
                row[metric] = total / count
        aggregate.append(row)
    aggregate.sort(key=lambda x: (x["checkpoint"], x["layer"], x["segment"], x["span_length"]))
    change_points = []
    groups = defaultdict(list)
    for row in aggregate:
        groups[(row["checkpoint"], row["layer"], row["segment"])].append(row)
    for (checkpoint, layer, segment), values in groups.items():
        change_points.append({"checkpoint": checkpoint, "layer": layer, "segment": segment, **estimate_piecewise_change_point(values)})
    for writer in writers.values():
        writer.close()
    counts = {name: writer.count for name, writer in writers.items()}
    counts["tube_aggregate_rows"] = write_jsonl_gz(raw / "tube_scale_space_aggregate.jsonl.gz", aggregate)
    write_json(output / "analysis" / "tube_change_points.json", change_points)
    write_json(output / "gold_analysis_metadata.json", {"git_commit": git_commit(), "seconds": time.perf_counter() - start, "counts": counts})


def paired_checkpoint_keys(specs: list[CheckpointSpec]) -> list[tuple[str, str]]:
    available = {spec.key for spec in specs}
    pairs = []
    for treatment in specs:
        if treatment.formulation == "native":
            continue
        native = f"native_r{treatment.rank}_s{treatment.seed}"
        if native in available:
            pairs.append((native, treatment.key))
    return pairs


def analyze_matched(args) -> None:
    cache = args.output.resolve() / "cache" / "gold_states"
    specs = checkpoint_specs()
    destination = args.output.resolve() / "raw" / "matched_native_stp_displacement.jsonl.gz"
    writer = JsonlGzipWriter(destination)
    for native_key, treatment_key in paired_checkpoint_keys(specs):
        native_path, treatment_path = cache / f"{native_key}.pt", cache / f"{treatment_key}.pt"
        if not native_path.exists() or not treatment_path.exists():
            continue
        native = torch.load(native_path, map_location="cpu", weights_only=False)
        treatment = torch.load(treatment_path, map_location="cpu", weights_only=False)
        native_records = {r["panel_index"]: r for r in native["records"]}
        for tr in treatment["records"]:
            nr = native_records[tr["panel_index"]]
            limit = min(len(nr["states"]), len(tr["states"]))
            for layer_index in range(limit):
                label = native["depth_labels"][layer_index]
                for segment in ("source", "product", "cross"):
                    n_path, _ = trajectory_from_record(nr, layer_index, segment, args.device)
                    t_path, _ = trajectory_from_record(tr, layer_index, segment, args.device)
                    n = min(len(n_path), len(t_path))
                    triples = []
                    for length in range(2, n):
                        for s in deterministic_starts(n, length, 8, nr["panel_index"]):
                            t = s + length
                            triples.append((length, s, (s + t) // 2, t))
                    index = torch.tensor([[row[1], row[2], row[3]] for row in triples], device=args.device)
                    result = matched_geodesic_displacement(
                        (n_path[index[:, 0]], n_path[index[:, 1]], n_path[index[:, 2]]),
                        (t_path[index[:, 0]], t_path[index[:, 1]], t_path[index[:, 2]]),
                    )
                    cpu = {name: value.detach().cpu().numpy() for name, value in result.items()}
                    lengths = np.asarray([row[0] for row in triples])
                    for length in np.unique(lengths):
                        mask = lengths == length
                        row = {
                            "native": native_key, "treatment": treatment_key,
                            "panel_index": nr["panel_index"], "reaction_identity": nr["reaction_identity"],
                            "layer": label, "segment": segment, "span_length": int(length),
                            "sampled_triples": int(mask.sum()),
                        }
                        for name, values in cpu.items():
                            row[name] = float(np.nanmean(values[mask]))
                            row[f"sd_{name}"] = float(np.nanstd(values[mask]))
                        writer.write(row)
        print(json.dumps({"stage": "matched_pair_complete", "native": native_key, "treatment": treatment_key}), flush=True)
    writer.close()
    write_json(args.output.resolve() / "matched_analysis_metadata.json", {"git_commit": git_commit(), "rows": writer.count})


INTRINSIC_KEYS = {
    "native_r8_s533", "released_r8_l0.02_s533", "paper_r8_l0.02_s533",
    "native_r128_s533", "released_r128_l0.02_s533", "paper_r128_l0.02_s533",
    "native_r8_s1301", "released_r8_l0.02_s1301",
}


def prediction_path_for_key(key: str) -> Path:
    parts = key.split("_")
    seed = int(parts[-1][1:])
    if key.startswith("native_r8_"):
        return ROOT / f"runs/stp_matrix/a6000/stage_a/r8_l0.02/native/seed_{seed}/evaluation/predictions.jsonl"
    if key.startswith("released_r8_l0.02_"):
        return ROOT / f"runs/stp_matrix/a6000/stage_a/r8_l0.02/released/seed_{seed}/evaluation/predictions.jsonl"
    if key.startswith("paper_r8_l0.02_"):
        return ROOT / f"runs/stp_matrix/a6000/stage_c/paper_r8_l0.02/seed_{seed}/evaluation/predictions.jsonl"
    raise KeyError(key)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _fast_reaction_tokenization(tokenizer, source: str, target: str) -> tuple[list[int], list[int]]:
    """Tokenize exact serialization and locate target content without RDKit."""
    text = f"{REACTANT_START}{source}{END}{PRODUCT_START}{target}{END}"
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    target_start = len(REACTANT_START) + len(source) + len(END) + len(PRODUCT_START)
    target_end = target_start + len(target)
    target_positions = [
        index for index, (start, end) in enumerate(encoded["offset_mapping"])
        if start >= target_start and end <= target_end
    ]
    if not target_positions:
        raise ValueError("candidate serialization produced no target content tokens")
    return encoded["input_ids"], target_positions


def _candidate_workload(tokenizer, key: str) -> list[dict]:
    """Build the locked per-view gold/wrong workload from archived beams."""
    rows = read_jsonl(prediction_path_for_key(key))
    special = {}
    if key in {"native_r8_s1301", "released_r8_l0.02_s1301"}:
        diagnostic = json.loads((
            ROOT / "runs/stp_completion/a6000/existing_diagnostics/released_r8_l0.02_seed1301_beams.json"
        ).read_text(encoding="utf-8"))
        losses = diagnostic["comparison"]["native_only_top1"]["rows"]
        special = {int(row["panel_index"]): row["treatment_winner"] for row in losses}
    workload = []
    for row in rows:
        panel_index = int(row["panel_index"])
        if panel_index >= 256 and panel_index not in special:
            continue
        for view, source in enumerate(row["sources"]):
            raw = row["raw_candidates_by_view"][view]
            canonical = row["canonical_candidates_by_view"][view]
            candidates = [(row["target"], row["target"], "gold", 0)]
            # The full panel retains the behaviorally decisive gold, view top-1,
            # and highest wrong path.  The first 64 reactions (and the seed-1301
            # natural experiment) retain all top-five candidates as a robustness
            # panel.  This avoids recomputing thousands of redundant additional
            # paths while preserving every preregistered primary comparison.
            retain_all = panel_index < 64 or panel_index in special
            for rank, (raw_value, canonical_value) in enumerate(zip(raw[:5], canonical[:5]), 1):
                role = "view_top1" if rank == 1 else "additional"
                if canonical_value and canonical_value != row["target"] and not any(c[2] == "highest_wrong" for c in candidates):
                    role = "highest_wrong"
                if retain_all or rank == 1 or role == "highest_wrong":
                    candidates.append((raw_value, canonical_value, role, rank))
            if panel_index in special:
                candidates.append((special[panel_index], special[panel_index], "seed1301_promoted_wrong", -1))
            seen = set()
            for candidate, canonical_candidate, role, rank in candidates:
                if not candidate or (candidate in seen and role != "seed1301_promoted_wrong"):
                    continue
                seen.add(candidate)
                input_ids, target_positions = _fast_reaction_tokenization(tokenizer, source, candidate)
                workload.append({
                    "panel_index": panel_index, "reaction_identity": row["reaction_identity"],
                    "view": view, "source": source, "gold": row["target"],
                    "candidate": candidate, "role": role, "beam_rank": rank,
                    "canonical_candidate": canonical_candidate,
                    "valid_candidate": bool(canonical_candidate),
                    "input_ids": input_ids, "target_positions": target_positions,
                    "aggregate_rank": (
                        row["ranked_candidates"].index(row["target"]) + 1
                        if row["target"] in row["ranked_candidates"] else None
                    ),
                    "model_aggregate_correct": bool(row["ranked_candidates"] and row["ranked_candidates"][0] == row["target"]),
                })
    return workload


def _candidate_geometry(path: torch.Tensor, logits: torch.Tensor | None) -> dict:
    tube = tube_scale_space(path)
    change = estimate_piecewise_change_point(tube)
    acceleration = acceleration_decomposition(path)
    paper_spans, released_spans = [], []
    for length in range(2, len(path)):
        for s in deterministic_starts(len(path), length, 8, len(path)):
            t = s + length
            r = (s + t) // 2
            paper_spans.append((s, r, t))
    for length in range(1, len(path)):
        for s in deterministic_starts(len(path), length, 8, len(path) + 31):
            t = s + length
            released_spans.append((s, t))
    paper = torch.tensor(paper_spans, device=path.device, dtype=torch.long).reshape(-1, 3)
    released = torch.tensor(released_spans, device=path.device)
    paper_values = (
        1 - cosine(
            path[paper[:, 1]] - path[paper[:, 0]],
            path[paper[:, 2]] - path[paper[:, 1]],
        ) if len(paper) else path.new_empty(0)
    )
    total = path[-1] - path[0]
    patch = path[released[:, 1]] - path[released[:, 0]]
    released_values = 1 - cosine(patch, total[None] - patch)
    tube_columns = {
        name: [row[name] for row in tube]
        for name in ("span_length", "triples", "zero_chords", "mean", "rms", "maximum", "p50", "p90", "p95", "monotonicity_violation", "fraction_gt_0.05", "fraction_gt_0.1", "fraction_gt_0.2", "fraction_gt_0.5")
    }
    persistence = tangent_autocorrelation(path)
    persistence_columns = {
        name: [row[name] for row in persistence]
        for name in ("lag", "n", "mean", "median")
    }
    result = {
        "tokens": len(path), "tube_scale": tube_columns, "tube_change_point": change,
        "paper_loss": float(paper_values.mean()) if len(paper_values) else math.nan,
        "released_loss": float(released_values.mean()) if len(released_values) else math.nan,
        "tangent_persistence": persistence_columns,
        "speed": float(acceleration["speed"].mean()),
        "normal_acceleration": float(acceleration["acceleration_normal"].mean()),
        "normalized_normal_acceleration": float(acceleration["normalized_normal"].mean()),
    }
    if logits is not None:
        functional = functional_metrics_from_logits(path, logits)
        result.update({key: value for key, value in functional.items() if not torch.is_tensor(value)})
        ray = []
        for scale in range(1, min(32, (len(path) - 1) // 2) + 1):
            values = optimal_ray_residual(path[:-2 * scale], path[scale:-scale], path[2 * scale:]).detach().cpu().numpy()
            ray.append({"horizon": scale, "mean": float(values.mean()), "median": float(np.median(values))})
        result["ray_residual"] = {
            name: [row[name] for row in ray] for name in ("horizon", "mean", "median")
        }
    return result


def analyze_candidates(args) -> None:
    output = args.output.resolve()
    writer = JsonlGzipWriter(output / "raw" / "gold_wrong_candidate_geometry.jsonl.gz")
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab_size = len(tokenizer)
    add_predictor_tokens(tokenizer)
    specs = [spec for spec in checkpoint_specs() if spec.key in PRIMARY_KEYS]
    requested_keys = {value.strip() for value in args.keys.split(",") if value.strip()}
    if requested_keys:
        specs = [spec for spec in specs if spec.key in requested_keys]
    started = time.perf_counter()
    model = load_lora_model(
        MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab_size,
        attention_dropout=0.0, attn_implementation="sdpa", lora_rank=8, lora_alpha=8,
    ).to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    capture = SelectedStateCapture(model)
    with ThreadPoolExecutor(max_workers=args.analysis_workers) as executor:
        for spec_index, spec in enumerate(specs, 1):
            workload = _candidate_workload(tokenizer, spec.key)
            workload.sort(key=lambda row: len(row["input_ids"]))
            load_adapter_checkpoint(model, ROOT / spec.checkpoint)
            weight = model.get_output_embeddings().weight.detach().float().to(args.device)
            with torch.inference_mode():
                for batch_start in range(0, len(workload), args.batch_size):
                    batch = workload[batch_start:batch_start + args.batch_size]
                    maximum = max(len(row["input_ids"]) for row in batch)
                    ids = torch.zeros((len(batch), maximum), dtype=torch.long, device=args.device)
                    mask = torch.zeros_like(ids, dtype=torch.bool)
                    for i, row in enumerate(batch):
                        length = len(row["input_ids"])
                        ids[i, :length] = torch.tensor(row["input_ids"], device=args.device)
                        mask[i, :length] = True
                    capture.clear()
                    model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
                    tasks = []
                    for i, row in enumerate(batch):
                        positions = [row["target_positions"][0] - 1, *row["target_positions"]]
                        labels = ["final_post_norm"]
                        if row["panel_index"] < 64 or row["role"] == "seed1301_promoted_wrong":
                            labels = list(DEPTH_LABELS)
                        for label in labels:
                            states = capture.values[label][i, positions].float()
                            tasks.append((row, label, states, label == "final_post_norm"))
                    final_tasks = [task for task in tasks if task[3]]
                    final_logits = []
                    if final_tasks:
                        lengths = [len(task[2]) for task in final_tasks]
                        joined = torch.cat([task[2] for task in final_tasks])
                        joined_logits = joined @ weight.T
                        final_logits = list(torch.split(joined_logits, lengths))
                    final_index = 0
                    cpu_tasks = []
                    for row, label, states, is_final in tasks:
                        logits = final_logits[final_index] if is_final else None
                        final_index += int(is_final)
                        cpu_tasks.append((row, label, states.cpu(), None if logits is None else logits.cpu()))
                    metrics_rows = executor.map(lambda task: _candidate_geometry(task[2], task[3]), cpu_tasks)
                    for (row, label, _, _), metrics in zip(cpu_tasks, metrics_rows):
                        writer.write({
                            **{k: v for k, v in row.items() if k not in {"input_ids", "target_positions"}},
                            "checkpoint": spec.key, "rank": spec.rank,
                            "formulation": spec.formulation, "lambda": spec.stp_lambda,
                            "seed": spec.seed, "layer": label, **metrics,
                        })
                    del ids, mask, tasks, cpu_tasks, final_tasks, final_logits
            del weight, workload
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(json.dumps({"stage": "candidate_checkpoint_complete", "index": spec_index, "total": len(specs), "key": spec.key, "rows": writer.count}), flush=True)
    capture.close()
    del model, capture
    writer.close()
    write_json(output / "candidate_analysis_metadata.json", {
        "git_commit": git_commit(), "rows": writer.count, "seconds": time.perf_counter() - started,
        "prediction_hashes": {spec.key: sha256(prediction_path_for_key(spec.key)) for spec in specs},
    })


def _seed1301_loss_panels() -> set[int]:
    diagnostic = json.loads((
        ROOT / "runs/stp_completion/a6000/existing_diagnostics/released_r8_l0.02_seed1301_beams.json"
    ).read_text(encoding="utf-8"))
    return {int(row["panel_index"]) for row in diagnostic["comparison"]["native_only_top1"]["rows"]}


def analyze_candidate_intrinsic(args) -> None:
    """Bounded intrinsic estimate for gold/wrong beam trajectories."""
    output = args.output.resolve()
    writer = JsonlGzipWriter(output / "raw" / "candidate_intrinsic_geometry.jsonl.gz")
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab_size = len(tokenizer)
    add_predictor_tokens(tokenizer)
    natural_panels = _seed1301_loss_panels()
    specs = [spec for spec in checkpoint_specs() if spec.key in PRIMARY_KEYS]
    model = load_lora_model(
        MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab_size,
        attention_dropout=0.0, attn_implementation="sdpa", lora_rank=8, lora_alpha=8,
    ).to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    capture = SelectedStateCapture(model)
    started = time.perf_counter()
    for spec in specs:
        payload = torch.load(
            output / "cache" / "gold_states" / f"{spec.key}.pt",
            map_location="cpu", weights_only=False,
        )
        reference_rows = []
        for record in payload["records"]:
            states = record["states"][-1].float()
            for local, position in enumerate(record["target_indices"].long().tolist()):
                reference_rows.append((
                    _stable_priority("candidate-ref", record["reaction_identity"], local),
                    states[position], record["reaction_identity"],
                ))
        reference_rows.sort(key=lambda row: row[0])
        reference_rows = reference_rows[:min(12_000, len(reference_rows))]
        references = torch.stack([row[1] for row in reference_rows]).to(args.device)
        reference_reactions = np.asarray([row[2] for row in reference_rows])
        mean = references.mean(0)
        centered = references - mean
        whiten_rank = min(128, len(references) - 1, references.shape[1])
        torch.manual_seed(_stable_priority("candidate-whitening") % (2**63 - 1))
        _, singular, principal = torch.pca_lowrank(centered, q=whiten_rank, center=False, niter=2)
        scale = (singular / math.sqrt(max(1, len(references) - 1))).clamp_min(1e-5)
        whitened_references = (centered @ principal) / scale

        workload = [
            row for row in _candidate_workload(tokenizer, spec.key)
            if row["role"] in {"gold", "highest_wrong", "seed1301_promoted_wrong"}
            and (row["panel_index"] < 64 or row["panel_index"] in natural_panels)
        ]
        workload.sort(key=lambda row: len(row["input_ids"]))
        load_adapter_checkpoint(model, ROOT / spec.checkpoint)
        query_rows = []
        with torch.inference_mode():
            for batch_start in range(0, len(workload), args.batch_size):
                batch = workload[batch_start:batch_start + args.batch_size]
                maximum = max(len(row["input_ids"]) for row in batch)
                ids = torch.zeros((len(batch), maximum), dtype=torch.long, device=args.device)
                mask = torch.zeros_like(ids, dtype=torch.bool)
                for index, row in enumerate(batch):
                    ids[index, :len(row["input_ids"])] = torch.tensor(row["input_ids"], device=args.device)
                    mask[index, :len(row["input_ids"])] = True
                capture.clear()
                model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
                for index, row in enumerate(batch):
                    positions = [row["target_positions"][0] - 1, *row["target_positions"]]
                    path = capture.values["final_post_norm"][index, positions].float()
                    if len(path) < 3:
                        continue
                    possible = list(range(1, len(path) - 1))
                    possible.sort(key=lambda position: _stable_priority(
                        "candidate-query", row["reaction_identity"], row["view"], row["role"], position,
                    ))
                    for position in possible[:3]:
                        query_rows.append({
                            **{name: value for name, value in row.items() if name not in {"input_ids", "target_positions"}},
                            "position": position, "state": path[position].cpu(),
                            "velocity": (path[position] - path[position - 1]).cpu(),
                            "acceleration": (path[position + 1] - 2 * path[position] + path[position - 1]).cpu(),
                            "robust": row["panel_index"] in natural_panels and row["role"] in {"gold", "seed1301_promoted_wrong"},
                        })
                del ids, mask

        queries = torch.stack([row["state"] for row in query_rows]).to(args.device)
        velocities = torch.stack([row["velocity"] for row in query_rows]).to(args.device)
        accelerations = torch.stack([row["acceleration"] for row in query_rows]).to(args.device)
        query_reactions = np.asarray([row["reaction_identity"] for row in query_rows])
        whitened_queries = ((queries - mean) @ principal) / scale
        central = np.arange(len(query_rows))
        robust = np.asarray([index for index, row in enumerate(query_rows) if row["robust"]], dtype=np.int64)
        settings = [(central, "euclidean", 64, (16,))]
        for metric in ("euclidean", "pca_whitened_128"):
            for k in (32, 64, 128):
                dimensions = tuple(
                    dimension for dimension in (8, 16, 32)
                    if not (metric == "euclidean" and k == 64 and dimension == 16)
                )
                settings.append((robust, metric, k, dimensions))
        for selected, metric, k, dimensions in settings:
            if not len(selected) or not dimensions:
                continue
            selected_tensor = torch.as_tensor(selected, device=args.device, dtype=torch.long)
            search_references = references if metric == "euclidean" else whitened_references
            search_queries = queries[selected_tensor] if metric == "euclidean" else whitened_queries[selected_tensor]
            eligible_np = query_reactions[selected, None] != reference_reactions[None, :]
            eligible = torch.from_numpy(eligible_np).to(args.device)
            neighbor_indices = _topk_neighbors(
                search_queries, search_references, eligible, k, batch_size=32,
            )
            for batch_start in range(0, len(selected), 16):
                batch_end = min(batch_start + 16, len(selected))
                original_indices = selected[batch_start:batch_end]
                original_tensor = torch.as_tensor(original_indices, device=args.device, dtype=torch.long)
                neighbors = references[neighbor_indices[batch_start:batch_end]]
                decompositions = _batched_local_pca_decomposition(
                    neighbors, velocities[original_tensor], accelerations[original_tensor], dimensions,
                )
                cpu = {
                    dimension: {name: values.detach().cpu().numpy() for name, values in metrics.items()}
                    for dimension, metrics in decompositions.items()
                }
                for offset, original_index in enumerate(original_indices):
                    row = query_rows[int(original_index)]
                    for dimension in dimensions:
                        writer.write({
                            **{name: value for name, value in row.items() if name not in {"state", "velocity", "acceleration", "robust"}},
                            "checkpoint": spec.key, "search_metric": metric,
                            "neighbors": k, "tangent_dim": dimension,
                            **{name: float(values[offset]) for name, values in cpu[dimension].items()},
                        })
            del eligible, neighbor_indices
        print(json.dumps({"stage": "candidate_intrinsic_checkpoint", "checkpoint": spec.key, "rows": writer.count}), flush=True)
        del payload, references, whitened_references, queries, velocities, accelerations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    capture.close(); del capture, model
    writer.close()
    write_json(output / "candidate_intrinsic_metadata.json", {
        "git_commit": git_commit(), "rows": writer.count,
        "seconds": time.perf_counter() - started,
    })


CONE_KEYS = {
    "native_r8_s1301", "released_r8_l0.02_s1301",
    "native_r8_s533", "paper_r8_l0.02_s533",
}


def _cone_prefixes(tokenizer, panel: Path) -> list[dict]:
    candidates = {"event": [], "ordinary": []}
    for example in build_gold_examples(tokenizer, panel):
        targets = [token for token in example.tokens if token.segment == "target"]
        for local, token in enumerate(targets[:-5]):
            kind = "event" if token.events else "ordinary"
            candidates[kind].append((
                _stable_priority("cone", example.reaction_identity, local),
                {
                    "panel_index": example.panel_index,
                    "reaction_identity": example.reaction_identity,
                    "position": token.index, "target_local_position": local,
                    "kind": kind, "events": sorted(token.events),
                    "input_ids": example.input_ids,
                },
            ))
    selected = []
    for kind in ("event", "ordinary"):
        candidates[kind].sort(key=lambda x: x[0])
        selected.extend(row for _, row in candidates[kind][:32])
    return sorted(selected, key=lambda row: (row["panel_index"], row["position"]))


def _forward_final_states(model, capture, sequences: list[list[int]], device: str):
    maximum = max(map(len, sequences))
    ids = torch.zeros((len(sequences), maximum), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    lengths = []
    for row, sequence in enumerate(sequences):
        lengths.append(len(sequence))
        ids[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[row, :len(sequence)] = True
    capture.clear()
    model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
    states = torch.stack([capture.values["final_post_norm"][row, length - 1].float() for row, length in enumerate(lengths)])
    return states


def _weighted_cone_metrics(
    base: torch.Tensor, branches: torch.Tensor, gold: torch.Tensor,
    probabilities: torch.Tensor, weight: torch.Tensor,
) -> dict:
    directions = branches - base
    gold_direction = gold - base
    probabilities = probabilities / probabilities.sum().clamp_min(1e-12)
    mean_direction = (probabilities[:, None] * directions).sum(0)
    cos_mean = cosine(directions, mean_direction.expand_as(directions))
    cos_gold = cosine(directions, gold_direction.expand_as(directions))
    coefficient = (directions * mean_direction).sum(-1) / mean_direction.square().sum().clamp_min(1e-12)
    perpendicular = directions - coefficient[:, None] * mean_direction
    branch_output = (branches @ weight.T).softmax(-1)
    mixture = (probabilities[:, None] * branch_output).sum(0)
    fr_dispersion = (probabilities * fisher_rao_distance(branch_output, mixture[None])).sum()
    return {
        "weighted_angle_from_mean": float((probabilities * torch.acos(cos_mean.clamp(-1, 1))).sum()),
        "weighted_angle_from_gold": float((probabilities * torch.acos(cos_gold.clamp(-1, 1))).sum()),
        "perpendicular_variance": float((probabilities * perpendicular.square().sum(-1)).sum()),
        "mean_axis_gold_cosine": float(cosine(mean_direction, gold_direction)),
        "fisher_dispersion": float(fr_dispersion),
        "mean_direction_norm": float(mean_direction.norm()),
        "gold_direction_norm": float(gold_direction.norm()),
    }


def analyze_cones(args) -> None:
    output = args.output.resolve()
    writer = JsonlGzipWriter(output / "raw" / "inference_cones.jsonl.gz")
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab_size = len(tokenizer)
    add_predictor_tokens(tokenizer)
    prefixes = _cone_prefixes(tokenizer, args.panel.resolve())
    specs = [spec for spec in checkpoint_specs() if spec.key in CONE_KEYS]
    started = time.perf_counter()
    model = load_lora_model(
        MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab_size,
        attention_dropout=0.0, attn_implementation="sdpa", lora_rank=8, lora_alpha=8,
    ).to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    capture = SelectedStateCapture(model)
    for spec in specs:
        load_adapter_checkpoint(model, ROOT / spec.checkpoint)
        weight = model.get_output_embeddings().weight.detach().float().to(args.device)
        with torch.inference_mode():
            for item in prefixes:
                full = item["input_ids"]
                prefix = full[:item["position"] + 1]
                base = _forward_final_states(model, capture, [prefix], args.device)[0]
                base_logits = base @ weight.T
                top = torch.topk(base_logits, 10)
                branch_probabilities = top.values.softmax(-1)
                branch_sequences = [prefix + [int(token)] for token in top.indices]
                for horizon in range(1, 6):
                    branch_states = _forward_final_states(model, capture, branch_sequences, args.device)
                    gold_sequence = full[:item["position"] + 1 + horizon]
                    gold_state = _forward_final_states(model, capture, [gold_sequence], args.device)[0]
                    metrics = _weighted_cone_metrics(base, branch_states, gold_state, branch_probabilities, weight)
                    writer.write({
                        "checkpoint": spec.key, "seed": spec.seed,
                        "formulation": spec.formulation, "panel_index": item["panel_index"],
                        "reaction_identity": item["reaction_identity"], "position": item["position"],
                        "target_local_position": item["target_local_position"], "kind": item["kind"],
                        "events": item["events"], "horizon": horizon, **metrics,
                    })
                    if horizon < 5:
                        next_logits = branch_states @ weight.T
                        next_tokens = next_logits.argmax(-1).tolist()
                        branch_sequences = [sequence + [int(token)] for sequence, token in zip(branch_sequences, next_tokens)]
        del weight
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(json.dumps({"stage": "cone_checkpoint_complete", "checkpoint": spec.key, "rows": writer.count}), flush=True)
    capture.close()
    del model, capture
    writer.close()
    write_json(output / "cone_analysis_metadata.json", {
        "git_commit": git_commit(), "rows": writer.count, "prefixes": len(prefixes),
        "seconds": time.perf_counter() - started,
    })


def _stable_priority(*values) -> int:
    text = "|".join(map(str, values)).encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little")


def _topk_neighbors(
    queries: torch.Tensor, references: torch.Tensor, eligible: torch.Tensor,
    k: int, batch_size: int = 32,
) -> torch.Tensor:
    """Exact squared-Euclidean top-k with per-query eligibility masks."""
    result = []
    reference_norm = references.square().sum(-1)
    for start in range(0, len(queries), batch_size):
        q = queries[start:start + batch_size]
        distances = q.square().sum(-1, keepdim=True) + reference_norm - 2 * q @ references.T
        distances.masked_fill_(~eligible[start:start + batch_size], float("inf"))
        result.append(torch.topk(distances, k, dim=1, largest=False).indices)
    return torch.cat(result)


def _local_pca_decomposition(
    neighbors: torch.Tensor, velocity: torch.Tensor, acceleration: torch.Tensor,
    dimensions: tuple[int, ...],
) -> list[dict]:
    """Local PCA through the smaller neighbor Gram matrix."""
    centered = neighbors.float() - neighbors.float().mean(0, keepdim=True)
    gram = centered @ centered.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(1e-10)
    eigenvectors = eigenvectors[:, order]
    result = []
    for dimension in dimensions:
        dim = min(dimension, len(neighbors) - 1)
        # V = U^T X / singular_value, stored as rows.
        basis = (eigenvectors[:, :dim].T @ centered) / eigenvalues[:dim].sqrt().unsqueeze(1)
        tangent_a = (acceleration @ basis.T) @ basis
        normal_a = acceleration - tangent_a
        tangent_v = (velocity @ basis.T) @ basis
        coefficient = (tangent_a * tangent_v).sum() / tangent_v.square().sum().clamp_min(1e-12)
        geo = tangent_a - coefficient * tangent_v
        result.append({
            "tangent_dim": dimension,
            "tangent_acceleration": float(tangent_a.norm()),
            "normal_acceleration": float(normal_a.norm()),
            "geodesic_violation": float(geo.norm()),
            "projected_velocity": float(tangent_v.norm()),
            "geodesic_over_acceleration": float(geo.norm() / acceleration.norm().clamp_min(1e-12)),
            "normal_over_acceleration": float(normal_a.norm() / acceleration.norm().clamp_min(1e-12)),
        })
    return result


def _batched_local_pca_decomposition(
    neighbors: torch.Tensor, velocities: torch.Tensor, accelerations: torch.Tensor,
    dimensions: tuple[int, ...],
) -> dict[int, dict[str, torch.Tensor]]:
    """Exact local-PCA decomposition for a batch of query neighborhoods.

    This is algebraically the same smaller-Gram construction as
    ``_local_pca_decomposition`` but amortizes eigensolver and projection kernel
    launches across queries.  No approximation or neighbor-order change is
    introduced.
    """
    centered = neighbors.float() - neighbors.float().mean(1, keepdim=True)
    gram = centered @ centered.transpose(1, 2)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    maximum = min(max(dimensions), neighbors.shape[1] - 1)
    eigenvalues = eigenvalues[:, -maximum:].flip(1).clamp_min(1e-10)
    eigenvectors = eigenvectors[:, :, -maximum:].flip(2)
    basis = torch.einsum("bkm,bkd->bmd", eigenvectors, centered)
    basis = basis / eigenvalues.sqrt().unsqueeze(2)
    acceleration_coefficients = torch.einsum("bd,bmd->bm", accelerations.float(), basis)
    velocity_coefficients = torch.einsum("bd,bmd->bm", velocities.float(), basis)
    acceleration_squared = accelerations.float().square().sum(1)
    output = {}
    for requested_dimension in dimensions:
        dimension = min(requested_dimension, neighbors.shape[1] - 1, maximum)
        a_coeff = acceleration_coefficients[:, :dimension]
        v_coeff = velocity_coefficients[:, :dimension]
        tangent_a_squared = a_coeff.square().sum(1).clamp_min(0)
        tangent_v_squared = v_coeff.square().sum(1).clamp_min(0)
        av = (a_coeff * v_coeff).sum(1)
        geodesic_squared = (
            tangent_a_squared - av.square() / tangent_v_squared.clamp_min(1e-12)
        ).clamp_min(0)
        normal_squared = (acceleration_squared - tangent_a_squared).clamp_min(0)
        acceleration_norm = acceleration_squared.sqrt().clamp_min(1e-12)
        output[requested_dimension] = {
            "tangent_acceleration": tangent_a_squared.sqrt(),
            "normal_acceleration": normal_squared.sqrt(),
            "geodesic_violation": geodesic_squared.sqrt(),
            "projected_velocity": tangent_v_squared.sqrt(),
            "geodesic_over_acceleration": geodesic_squared.sqrt() / acceleration_norm,
            "normal_over_acceleration": normal_squared.sqrt() / acceleration_norm,
        }
    return output


def analyze_intrinsic(args) -> None:
    output = args.output.resolve()
    cache = output / "cache" / "gold_states"
    writer = JsonlGzipWriter(output / "raw" / "intrinsic_manifold_decomposition.jsonl.gz")
    started = time.perf_counter()
    for cache_path in sorted(cache.glob("*.pt")):
        key = cache_path.stem
        if key not in INTRINSIC_KEYS:
            continue
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        for layer_index, layer in enumerate(payload["depth_labels"]):
            reference_rows = []
            query_rows = []
            for record in payload["records"]:
                states = record["states"][layer_index].float()
                for segment, indices in (("source", record["source_indices"]), ("product", record["target_indices"])):
                    positions = indices.long().tolist()
                    for local, position in enumerate(positions):
                        reference_rows.append((
                            _stable_priority("intrinsic-reference", layer, record["reaction_identity"], segment, local),
                            states[position], record["reaction_identity"], segment,
                        ))
                    for local in range(1, len(positions) - 1):
                        p0, p1, p2 = positions[local - 1:local + 2]
                        velocity = states[p1] - states[p0]
                        acceleration = states[p2] - 2 * states[p1] + states[p0]
                        query_rows.append((
                            _stable_priority("intrinsic-query", layer, record["reaction_identity"], segment, local),
                            states[p1], velocity, acceleration, record["reaction_identity"], segment,
                            record["panel_index"], local,
                        ))
            reference_rows.sort(key=lambda item: item[0])
            query_rows.sort(key=lambda item: item[0])
            # Fixed hash subsamples bound exact neighbor search and are independent
            # of all geometry/performance values.
            reference_rows = reference_rows[: min(12_000, len(reference_rows))]
            query_rows = query_rows[: min(args.intrinsic_queries, len(query_rows))]
            references = torch.stack([row[1] for row in reference_rows]).to(args.device)
            reference_reactions = np.asarray([row[2] for row in reference_rows])
            reference_segments = np.asarray([row[3] for row in reference_rows])
            queries = torch.stack([row[1] for row in query_rows]).to(args.device)
            velocities = torch.stack([row[2] for row in query_rows]).to(args.device)
            accelerations = torch.stack([row[3] for row in query_rows]).to(args.device)
            query_reactions = np.asarray([row[4] for row in query_rows])
            query_segments = np.asarray([row[5] for row in query_rows])
            mean = references.mean(0)
            centered = references - mean
            whiten_rank = min(128, len(references) - 1, references.shape[1])
            # Low-rank covariance whitening retains the well-estimated global
            # directions and avoids an unstable 2048-D covariance inverse.
            torch.manual_seed(_stable_priority("intrinsic-whitening", layer) % (2**63 - 1))
            _, singular, principal = torch.pca_lowrank(
                centered, q=whiten_rank, center=False, niter=2,
            )
            scale = (singular / math.sqrt(max(1, len(references) - 1))).clamp_min(1e-5)
            whitened_references = (centered @ principal) / scale
            whitened_queries = ((queries - mean) @ principal) / scale
            for search_metric in ("euclidean", "pca_whitened_128"):
                search_references = references if search_metric == "euclidean" else whitened_references
                search_queries = queries if search_metric == "euclidean" else whitened_queries
                for same_segment in (False, True):
                    eligible_np = query_reactions[:, None] != reference_reactions[None, :]
                    if same_segment:
                        eligible_np &= query_segments[:, None] == reference_segments[None, :]
                    eligible = torch.from_numpy(eligible_np).to(args.device)
                    neighbor_indices = _topk_neighbors(search_queries, search_references, eligible, 128)
                    for k in (32, 64, 128):
                        for batch_start in range(0, len(query_rows), 16):
                            batch_end = min(batch_start + 16, len(query_rows))
                            local_indices = neighbor_indices[batch_start:batch_end, :k]
                            neighbors = references[local_indices]
                            decompositions = _batched_local_pca_decomposition(
                                neighbors, velocities[batch_start:batch_end],
                                accelerations[batch_start:batch_end], (8, 16, 32),
                            )
                            cpu = {
                                dimension: {name: values.detach().cpu().numpy() for name, values in metrics.items()}
                                for dimension, metrics in decompositions.items()
                            }
                            for offset, row in enumerate(query_rows[batch_start:batch_end]):
                                for dimension in (8, 16, 32):
                                    writer.write({
                                        "checkpoint": key, "layer": layer,
                                        "reaction_identity": row[4], "segment": row[5],
                                        "panel_index": row[6], "position": row[7],
                                        "search_metric": search_metric, "same_segment": same_segment,
                                        "neighbors": k, "tangent_dim": dimension,
                                        **{name: float(values[offset]) for name, values in cpu[dimension].items()},
                                    })
                    del eligible, neighbor_indices
            print(json.dumps({"stage": "intrinsic_layer_complete", "checkpoint": key, "layer": layer, "queries": len(query_rows), "references": len(reference_rows)}), flush=True)
            del references, queries, velocities, accelerations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del payload
    writer.close()
    write_json(output / "intrinsic_analysis_metadata.json", {
        "git_commit": git_commit(), "rows": writer.count,
        "seconds": time.perf_counter() - started, "query_cap": args.intrinsic_queries,
    })


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=(
        "extract-gold", "analyze-gold", "analyze-matched",
        "analyze-intrinsic", "analyze-candidates", "analyze-candidate-intrinsic",
        "analyze-cones",
    ))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--keys", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--intrinsic-queries", type=int, default=256)
    parser.add_argument("--analysis-limit", type=int, default=0)
    parser.add_argument("--analysis-workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "extract-gold":
        extract_gold(arguments)
    elif arguments.command == "analyze-gold":
        analyze_gold(arguments)
    elif arguments.command == "analyze-matched":
        analyze_matched(arguments)
    elif arguments.command == "analyze-intrinsic":
        analyze_intrinsic(arguments)
    elif arguments.command == "analyze-candidates":
        analyze_candidates(arguments)
    elif arguments.command == "analyze-candidate-intrinsic":
        analyze_candidate_intrinsic(arguments)
    else:
        analyze_cones(arguments)
