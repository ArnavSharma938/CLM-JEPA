"""Frozen all-checkpoint representation analysis for the STP experiment family.

The extractor reuses the exact chemical-event annotation and matched-control
geometry from :mod:`frozen_geometry`.  It adds whole-trajectory, spectral,
source/product relationship, fixed-objective, and same-seed Native-drift
measurements.  It never constructs an optimizer or calls backward.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import transformers

from chemfm import MODEL_DIR, TOKENIZER_DIR, load_lora_model, load_reaction_tokenizer
from frozen_geometry import (
    DEFAULT_PANEL,
    DEFAULT_STEREO_SUPPLEMENT,
    EVENT_TYPES,
    SPAN_BINS,
    Example,
    _compute_batch_geometry,
    _sampled_parameter_fingerprint,
    annotate_example,
    match_controls_and_anchors,
    one_minus_cosine,
    read_panel,
    read_stereo_supplement,
)
from jepa import add_predictor_tokens
from train import load_adapter_checkpoint


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "runs" / "stp_representation" / "frozen_all_checkpoints"
PROTOCOL_COMMIT = "e1c5011"
DIAGNOSTIC_SEED = 20260829


@dataclass(frozen=True)
class CheckpointSpec:
    key: str
    rank: int
    alpha: int
    formulation: str
    stp_lambda: float
    seed: int
    checkpoint: str

    @property
    def is_native(self) -> bool:
        return self.formulation == "native"


def checkpoint_specs() -> list[CheckpointSpec]:
    specs: list[CheckpointSpec] = []

    def add(key: str, rank: int, formulation: str, weight: float, seed: int, path: str):
        specs.append(CheckpointSpec(key, rank, rank, formulation, weight, seed, path))

    for seed in (533, 917, 1301):
        add(
            f"native_r8_s{seed}", 8, "native", 0.0, seed,
            f"runs/pair_residual/a6000/results/seed_{seed}/native/training/checkpoints/epoch_4",
        )
        add(
            f"released_r8_l0.02_s{seed}", 8, "released", .02, seed,
            f"runs/stp/a6000/results/seed_{seed}/stp/training/checkpoints/epoch_4",
        )
    for seed in (533, 917):
        add(
            f"native_r128_s{seed}", 128, "native", 0.0, seed,
            f"runs/stp_matrix/a6000/stage_b/native_r128/seed_{seed}/training/checkpoints/epoch_4",
        )
        add(
            f"released_r128_l0.02_s{seed}", 128, "released", .02, seed,
            f"runs/stp_matrix/a6000/stage_b/released_r128_l0.02/seed_{seed}/training/checkpoints/epoch_4",
        )
        add(
            f"paper_r128_l0.02_s{seed}", 128, "paper", .02, seed,
            f"runs/stp_completion/a6000/trajectories/paper_r128_l0.02/seed_{seed}/training/checkpoints/epoch_4",
        )
        add(
            f"paper_r8_l0.02_s{seed}", 8, "paper", .02, seed,
            f"runs/stp_matrix/a6000/stage_c/paper_r8_l0.02/seed_{seed}/training/checkpoints/epoch_4",
        )
        for weight, label in ((.005, "0.005"), (.08, "0.08")):
            add(
                f"released_r8_l{label}_s{seed}", 8, "released", weight, seed,
                f"runs/stp_matrix/a6000/stage_d/released_r8_l{label}/seed_{seed}/training/checkpoints/epoch_4",
            )
        for weight, label in ((.08, "0.08"), (.12, "0.12")):
            add(
                f"paper_r8_l{label}_s{seed}", 8, "paper", weight, seed,
                f"runs/stp_completion/a6000/trajectories/paper_r8_l{label}/seed_{seed}/training/checkpoints/epoch_4",
            )
    # Native checkpoints precede treatments so aligned references are available
    # without retaining multiple transformer instances.
    return sorted(specs, key=lambda item: (not item.is_native, item.rank, item.seed, item.key))


def validate_checkpoint_specs(specs: Sequence[CheckpointSpec]) -> None:
    if len(specs) != 22 or len({spec.key for spec in specs}) != 22:
        raise RuntimeError("the locked STP checkpoint census must contain 22 unique entries")
    for spec in specs:
        checkpoint = ROOT / spec.checkpoint
        config_path = checkpoint / "USPTO-MIT-Synthesis" / "adapter_config.json"
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if int(config["r"]) != spec.rank or int(config["lora_alpha"]) != spec.alpha:
            raise RuntimeError(f"adapter metadata mismatch for {spec.key}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_examples(
    tokenizer, panel: Path, stereo_supplement: Path, stereo_reactions: int,
    seed: int, anchors_per_event: int,
) -> tuple[list[Example], int]:
    rows = read_panel(panel)
    examples: list[Example] = []
    for panel_index, row in enumerate(rows):
        example = annotate_example(tokenizer, row, panel_index, "prespecified_256")
        match_controls_and_anchors(
            example, seed, anchors_per_event,
            categories=("ring_closure", "branch", "motif", "reaction_center"),
        )
        examples.append(example)
    stereo_rows = read_stereo_supplement(stereo_supplement, stereo_reactions, seed)
    for panel_index, row in enumerate(stereo_rows, start=len(rows)):
        example = annotate_example(
            tokenizer, row, panel_index, "uspto50k_stereo_supplement",
            infer_reaction_center=False,
        )
        match_controls_and_anchors(
            example, seed, anchors_per_event, categories=("stereochemistry",),
        )
        examples.append(example)
    return examples, len(rows)


def allocate_event_arrays(pair_count: int, layers: int) -> dict[str, np.ndarray]:
    return {
        "local_event": np.full((pair_count, layers), np.nan, dtype=np.float32),
        "local_control": np.full((pair_count, layers), np.nan, dtype=np.float32),
        "semi_event": np.full((pair_count, layers), np.nan, dtype=np.float32),
        "semi_control": np.full((pair_count, layers), np.nan, dtype=np.float32),
        "semi_valid_counts": np.zeros((pair_count, layers), dtype=np.int16),
        "semi_event_bins": np.full((pair_count, layers, len(SPAN_BINS)), np.nan, dtype=np.float32),
        "semi_control_bins": np.full((pair_count, layers, len(SPAN_BINS)), np.nan, dtype=np.float32),
        "semi_bin_valid_counts": np.zeros((pair_count, layers, len(SPAN_BINS)), dtype=np.int16),
    }


def fixed_spans(example: Example, count: int, seed: int) -> dict[str, np.ndarray]:
    source_n = sum(token.segment == "source" for token in example.tokens)
    target_n = sum(token.segment == "target" for token in example.tokens)
    full_length = source_n + target_n
    if full_length < 3:
        raise ValueError("fixed STP diagnostics require at least three content tokens")
    rng = np.random.default_rng(seed + example.panel_index * 1_000_003 + 7717)
    released = []
    paper = []
    while len(released) < count:
        start = int(rng.integers(0, full_length))
        end = int(rng.integers(start + 1, full_length + 1))
        if start == 0 and end == full_length:
            continue
        released.append((start, end))
    while len(paper) < count:
        start = int(rng.integers(0, full_length - 1))
        end = int(rng.integers(start + 2, full_length + 1))
        interior = int(rng.integers(start + 1, end))
        paper.append((start, interior, end))
    return {
        "released": np.asarray(released, dtype=np.int16),
        "paper": np.asarray(paper, dtype=np.int16),
        "full_length": np.asarray(full_length, dtype=np.int16),
    }


def semantic_path(state: torch.Tensor, example: Example) -> torch.Tensor:
    """Return the exact framing-excluded path used by paper STP.

    Rows are offsets 0..L.  This is a vectorized form of
    ``PaperSemanticTubePrediction.semantic_path_embedding``.
    """
    source = [token.index for token in example.tokens if token.segment == "source"]
    target = [token.index for token in example.tokens if token.segment == "target"]
    if not source or not target:
        raise ValueError("reaction must contain source and target content")
    source_marker = source[0] - 1
    target_marker = target[0] - 1
    source_path = state[[source_marker, *source]]
    target_path = state[target] - state[target_marker] + state[source[-1]]
    return torch.cat((source_path, target_path), dim=0)


def objective_values(
    path: torch.Tensor, spans: dict[str, np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    released = torch.as_tensor(spans["released"], device=path.device, dtype=torch.long)
    patch = path[released[:, 1]] - path[released[:, 0]]
    total = path[-1] - path[0]
    complement = total.unsqueeze(0) - patch
    released_values = one_minus_cosine(complement, patch)
    paper = torch.as_tensor(spans["paper"], device=path.device, dtype=torch.long)
    first = path[paper[:, 1]] - path[paper[:, 0]]
    second = path[paper[:, 2]] - path[paper[:, 1]]
    paper_values = one_minus_cosine(first, second)
    return released_values, paper_values


def trajectory_metrics(state: torch.Tensor, indices: list[int]) -> dict[str, torch.Tensor]:
    values = state[indices].float()
    transitions = values[1:] - values[:-1]
    curvature = one_minus_cosine(transitions[:-1], transitions[1:]) if len(transitions) > 1 else values.new_tensor([math.nan])
    path_length = transitions.norm(dim=-1).sum()
    return {
        "activation_norm": values.norm(dim=-1).mean(),
        "transition_norm": transitions.norm(dim=-1).mean(),
        "curvature": torch.nanmean(curvature),
        "path_efficiency": (values[-1] - values[0]).norm() / path_length.clamp_min(1e-12),
        "pooled": values.mean(dim=0),
        "transition_sample": transitions[len(transitions) // 2],
    }


def _eigen_summary(values: torch.Tensor) -> dict[str, float]:
    values = values.float()
    centered = values - values.mean(dim=0, keepdim=True)
    gram = centered @ centered.T
    energy = torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
    probabilities = energy / energy.sum().clamp_min(1e-30)
    nonzero = probabilities.clamp_min(1e-30)
    effective_rank = torch.exp(-(probabilities * nonzero.log()).sum())
    participation = energy.sum().square() / energy.square().sum().clamp_min(1e-30)
    raw_energy = values.square().sum(dim=1).mean().clamp_min(1e-30)
    return {
        "variance": float(centered.square().mean()),
        "effective_rank": float(effective_rank),
        "participation_ratio": float(participation),
        "top1_energy": float(probabilities[0]),
        "top8_energy": float(probabilities[:8].sum()),
        "mean_direction_energy": float(values.mean(0).square().sum() / raw_energy),
    }


def _centered_cka(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.float() - first.float().mean(0, keepdim=True)
    second = second.float() - second.float().mean(0, keepdim=True)
    first_gram = first @ first.T
    second_gram = second @ second.T
    numerator = (first_gram * second_gram).sum()
    denominator = first_gram.square().sum().sqrt() * second_gram.square().sum().sqrt()
    return float(numerator / denominator.clamp_min(1e-30))


def relationship_summary(sources: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    sources = F.normalize(sources.float(), dim=-1)
    targets = F.normalize(targets.float(), dim=-1)
    scores = sources @ targets.T
    count = len(scores)
    true = scores.diag()
    shuffle = scores[torch.arange(count), torch.roll(torch.arange(count), 1)]
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    ranks = (order == torch.arange(count).unsqueeze(1)).nonzero()[:, 1] + 1
    return {
        "true_pair_cosine": float(true.mean()),
        "matched_shuffle_cosine": float(shuffle.mean()),
        "pairing_gap": float((true - shuffle).mean()),
        "retrieval_top1": float((ranks == 1).float().mean()),
        "retrieval_mrr": float((1.0 / ranks.float()).mean()),
        "source_target_cka": _centered_cka(sources, targets),
    }


def drift_summary(values: torch.Tensor, native: torch.Tensor) -> dict[str, float]:
    values = values.float()
    native = native.float()
    delta = values - native
    native_centered = native - native.mean(0, keepdim=True)
    denominator = native_centered.square().sum(dim=1).mean().sqrt().clamp_min(1e-30)
    return {
        "centered_linear_cka": _centered_cka(values, native),
        "aligned_cosine": float(F.cosine_similarity(values, native, dim=-1).mean()),
        "relative_rms_displacement": float(delta.square().sum(dim=1).mean().sqrt() / denominator),
        "displacement_effective_rank": _eigen_summary(delta)["effective_rank"],
    }


def summarize_checkpoint(
    pooled: dict[str, torch.Tensor], transitions: dict[str, torch.Tensor],
    native_reference: dict[str, torch.Tensor] | None,
) -> tuple[dict, dict | None]:
    layer_count = pooled["source"].shape[0]
    result = {"layers": []}
    drift = {"layers": []} if native_reference is not None else None
    for layer in range(layer_count):
        source = pooled["source"][layer]
        target = pooled["target"][layer]
        result["layers"].append({
            "layer": layer,
            "source_pooled_spectrum": _eigen_summary(source),
            "target_pooled_spectrum": _eigen_summary(target),
            "source_transition_spectrum": _eigen_summary(transitions["source"][layer]),
            "target_transition_spectrum": _eigen_summary(transitions["target"][layer]),
            "relationship": relationship_summary(source, target),
        })
        if drift is not None:
            drift["layers"].append({
                "layer": layer,
                "source": drift_summary(source, native_reference["source"][layer]),
                "target": drift_summary(target, native_reference["target"][layer]),
            })
    return result, drift


def extract_checkpoint(
    model, spec: CheckpointSpec, examples: Sequence[Example], main_count: int,
    fixed_span_sets: Sequence[dict[str, np.ndarray]], output_dir: Path,
    batch_size: int,
) -> tuple[dict, dict[str, torch.Tensor]]:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = _sampled_parameter_fingerprint(model)
    layer_count = int(model.config.num_hidden_layers) + 1
    hidden_size = int(model.config.hidden_size)
    pairs = [pair for example in examples for pair in example.pairs]
    event_arrays = allocate_event_arrays(len(pairs), layer_count)
    pair_offsets = {}
    offset = 0
    for example in examples:
        pair_offsets[example.panel_index] = offset
        offset += len(example.pairs)

    scalar_names = (
        "source_activation_norm", "target_activation_norm",
        "source_transition_norm", "target_transition_norm",
        "source_curvature", "target_curvature",
        "source_path_efficiency", "target_path_efficiency",
        "released_fixed_loss", "paper_fixed_loss",
    )
    scalar = {name: np.full((main_count, layer_count), np.nan, dtype=np.float32) for name in scalar_names}
    pooled = {
        segment: torch.empty((layer_count, main_count, hidden_size), dtype=torch.float32)
        for segment in ("source", "target")
    }
    transitions = {
        segment: torch.empty((layer_count, main_count, hidden_size), dtype=torch.float32)
        for segment in ("source", "target")
    }
    device = next(model.parameters()).device
    ordered = sorted(examples, key=lambda example: len(example.input_ids))
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for batch_start in range(0, len(ordered), batch_size):
            batch = ordered[batch_start:batch_start + batch_size]
            maximum = max(len(example.input_ids) for example in batch)
            input_ids = torch.full((len(batch), maximum), 0, dtype=torch.long, device=device)
            attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for row, example in enumerate(batch):
                length = len(example.input_ids)
                input_ids[row, :length] = torch.tensor(example.input_ids, device=device)
                attention_mask[row, :length] = True
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, use_cache=False, return_dict=True,
            )
            _compute_batch_geometry(outputs.hidden_states, batch, event_arrays, pair_offsets)
            for batch_index, example in enumerate(batch):
                if example.panel_index >= main_count:
                    continue
                source_indices = [token.index for token in example.tokens if token.segment == "source"]
                target_indices = [token.index for token in example.tokens if token.segment == "target"]
                for layer, states in enumerate(outputs.hidden_states):
                    state = states[batch_index]
                    source_metrics = trajectory_metrics(state, source_indices)
                    target_metrics = trajectory_metrics(state, target_indices)
                    for segment, metrics in (("source", source_metrics), ("target", target_metrics)):
                        scalar[f"{segment}_activation_norm"][example.panel_index, layer] = float(metrics["activation_norm"])
                        scalar[f"{segment}_transition_norm"][example.panel_index, layer] = float(metrics["transition_norm"])
                        scalar[f"{segment}_curvature"][example.panel_index, layer] = float(metrics["curvature"])
                        scalar[f"{segment}_path_efficiency"][example.panel_index, layer] = float(metrics["path_efficiency"])
                        pooled[segment][layer, example.panel_index] = metrics["pooled"].cpu()
                        transitions[segment][layer, example.panel_index] = metrics["transition_sample"].cpu()
                    path = semantic_path(state, example)
                    released, paper = objective_values(path, fixed_span_sets[example.panel_index])
                    scalar["released_fixed_loss"][example.panel_index, layer] = float(torch.nanmean(released))
                    scalar["paper_fixed_loss"][example.panel_index, layer] = float(torch.nanmean(paper))
            del outputs, input_ids, attention_mask
    elapsed = time.perf_counter() - started
    after = _sampled_parameter_fingerprint(model)
    if before != after:
        raise RuntimeError(f"frozen parameters changed for {spec.key}")
    checkpoint_dir = output_dir / "checkpoints" / spec.key
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pair_metadata = {
        "category": np.asarray([pair["category"] for pair in pairs]),
        "reaction_identity": np.asarray([pair["reaction_identity"] for pair in pairs]),
        "sample_origin": np.asarray([pair["sample_origin"] for pair in pairs]),
        "segment": np.asarray([pair["segment"] for pair in pairs]),
    }
    np.savez_compressed(checkpoint_dir / "event_geometry.npz", **event_arrays, **pair_metadata)
    np.savez_compressed(checkpoint_dir / "reaction_geometry.npz", **scalar)
    checkpoint_meta = {
        "spec": asdict(spec),
        "checkpoint_sha256": _sha256(ROOT / spec.checkpoint / "USPTO-MIT-Synthesis" / "adapter_model.safetensors"),
        "parameter_fingerprint": before,
        "inference_seconds": elapsed,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "all_parameters_frozen": True,
        "optimizer_constructed": False,
        "backward_called": False,
    }
    (checkpoint_dir / "metadata.json").write_text(json.dumps(checkpoint_meta, indent=2) + "\n", encoding="utf-8")
    return checkpoint_meta, {**pooled, "source_transition": transitions["source"], "target_transition": transitions["target"]}


def _event_summary(checkpoint_dir: Path) -> list[dict]:
    arrays = np.load(checkpoint_dir / "event_geometry.npz")
    categories = arrays["category"]
    reactions = arrays["reaction_identity"]
    result = []
    for metric, event_key, control_key in (
        ("local", "local_event", "local_control"),
        ("semi", "semi_event", "semi_control"),
    ):
        delta = arrays[event_key] - arrays[control_key]
        for category in EVENT_TYPES:
            mask = categories == category
            for layer in range(delta.shape[1]):
                reaction_means = []
                for identity in np.unique(reactions[mask]):
                    values = delta[mask & (reactions == identity), layer]
                    if np.isfinite(values).any():
                        reaction_means.append(float(np.nanmean(values)))
                values = np.asarray(reaction_means)
                result.append({
                    "metric": metric, "category": category, "layer": layer,
                    "reaction_n": len(values), "mean_delta": float(values.mean()),
                    "sd_delta": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
                })
    return result


def run(args: argparse.Namespace) -> dict:
    start = time.perf_counter()
    specs = checkpoint_specs()
    validate_checkpoint_specs(specs)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    # Every Report 07--09 trajectory used the shared trainer, which retained
    # the ten historical predictor tokens even though STP itself does not use
    # them.  Reconstruct that 402-row modules_to_save architecture exactly.
    chemfm_vocab_size = len(tokenizer)
    add_predictor_tokens(tokenizer)
    examples, main_count = build_examples(
        tokenizer, args.panel.resolve(), args.stereo_supplement.resolve(),
        args.stereo_reactions, args.seed, args.anchors_per_event,
    )
    fixed_span_sets = [fixed_spans(example, args.objective_spans, args.seed) for example in examples]
    native_references: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    records = []
    model = None
    current_rank = None
    for index, spec in enumerate(specs, start=1):
        if current_rank != spec.rank:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = load_lora_model(
                MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab_size,
                attention_dropout=0.0, attn_implementation=args.attn_implementation,
                lora_rank=spec.rank, lora_alpha=spec.alpha,
            ).to(args.device)
            current_rank = spec.rank
        load_adapter_checkpoint(model, ROOT / spec.checkpoint)
        metadata, states = extract_checkpoint(
            model, spec, examples, main_count, fixed_span_sets, output, args.batch_size,
        )
        pooled = {"source": states["source"], "target": states["target"]}
        transitions = {"source": states["source_transition"], "target": states["target_transition"]}
        reference = native_references.get((spec.rank, spec.seed))
        representation, drift = summarize_checkpoint(pooled, transitions, reference)
        checkpoint_dir = output / "checkpoints" / spec.key
        (checkpoint_dir / "representation_summary.json").write_text(
            json.dumps({"representation": representation, "native_drift": drift}, indent=2) + "\n",
            encoding="utf-8",
        )
        event_summary = _event_summary(checkpoint_dir)
        (checkpoint_dir / "event_summary.json").write_text(json.dumps(event_summary, indent=2) + "\n", encoding="utf-8")
        if spec.is_native:
            native_references[(spec.rank, spec.seed)] = {
                "source": states["source"].clone(), "target": states["target"].clone(),
            }
        records.append({**metadata, "has_native_drift": drift is not None})
        print(json.dumps({
            "stage": "checkpoint_complete", "index": index, "total": len(specs),
            "key": spec.key, "seconds": round(metadata["inference_seconds"], 1),
        }), flush=True)
        del states, pooled, transitions
        gc.collect()
    del model
    metadata = {
        "type": "frozen_stp_all_checkpoint_representation_analysis",
        "protocol_commit": PROTOCOL_COMMIT,
        "git_commit_at_start": os.popen(f'git -C "{ROOT}" rev-parse HEAD').read().strip(),
        "panel": str(args.panel.resolve()), "panel_sha256": _sha256(args.panel.resolve()),
        "stereo_supplement": str(args.stereo_supplement.resolve()),
        "stereo_supplement_sha256": _sha256(args.stereo_supplement.resolve()),
        "main_reactions": main_count, "stereo_reactions": args.stereo_reactions,
        "total_reactions": len(examples), "pair_count": sum(len(example.pairs) for example in examples),
        "layers": 23, "objective_spans_per_reaction": args.objective_spans,
        "anchors_per_event": args.anchors_per_event, "seed": args.seed,
        "checkpoint_count": len(specs), "checkpoints": records,
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(), "total_seconds": time.perf_counter() - start,
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "complete", "output": str(output), "seconds": round(metadata["total_seconds"], 1)}), flush=True)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--stereo-supplement", type=Path, default=DEFAULT_STEREO_SUPPLEMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stereo-reactions", type=int, default=64)
    parser.add_argument("--anchors-per-event", type=int, default=64)
    parser.add_argument("--objective-spans", type=int, default=32)
    parser.add_argument("--seed", type=int, default=DIAGNOSTIC_SEED)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="sdpa")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
