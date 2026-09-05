#!/usr/bin/env python
"""Execute the preregistered frozen latent-predictability audit.

Stages are intentionally resumable.  Nothing in this runner trains or changes
ChemFM; ``fit-probes`` trains only diagnostic predictors over frozen tensors.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chemfm import MODEL_DIR, TOKENIZER_DIR, load_lora_model, load_reaction_tokenizer  # noqa: E402
from frozen_geometry import (  # noqa: E402
    MOTIF_QUERIES, annotate_example, _atom_spans,
    _sampled_parameter_fingerprint,
)
from jepa import add_predictor_tokens  # noqa: E402
from latent_predictability import (  # noqa: E402
    HORIZONS, LAYERS, ResidualMLPProbe, RidgeProbe, Standardizer, TargetBasis,
    assert_disjoint_confirmation, canonical_atom_correspondence,
    build_suffix_cache, replay_suffix_from_cache,
    chemical_pair_id,
    decoder_distribution_metrics, deterministic_random_smiles, fit_probe,
    forecast_matrices, forecast_plan, materialize_forecast_plan,
    invariance_metrics, latent_metrics, locked_reaction_split,
    reaction_balanced_indices, sha256_file, suffix_replay_one_position,
    shuffled_reaction_targets,
)
from stp_representation_analysis import checkpoint_specs  # noqa: E402
from train import load_adapter_checkpoint  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from run_geodesic_audit import (  # noqa: E402
    SelectedStateCapture, _candidate_workload, find_llama, prediction_path_for_key,
)


DEFAULT_PANEL = ROOT / "data/clm_jepa_uspto_mit_validation_1024/uspto_mit_validation_1024.csv"
DEFAULT_OUTPUT = ROOT / "runs/latent_predictability_audit"
DEFAULT_SPLIT = ROOT / "data/clm_jepa_uspto_mit_latent_audit/splits.json"
PRIMARY_KEYS = {
    *(f"native_r8_s{seed}" for seed in (533, 917, 1301)),
    *(f"released_r8_l0.02_s{seed}" for seed in (533, 917, 1301)),
    *(f"paper_r8_l0.02_s{seed}" for seed in (533, 917)),
}
VIEW_SEEDS = (202609041, 202609042, 202609043, 202609044)
DEVELOPMENT_PANEL = ROOT / "data/clm_jepa_uspto_mit_official_endpoint/prespecified_stage1_512.jsonl"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1024:
        raise RuntimeError(f"locked probe panel must contain 1024 reactions, got {len(rows)}")
    identities = [row["reaction_identity"] for row in rows]
    if len(set(identities)) != len(rows):
        raise RuntimeError("probe panel contains duplicate reactions")
    return rows


def load_split_manifest(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("type") != "latent_predictability_reaction_splits":
        raise ValueError("wrong split-manifest type")
    return value


def recheck_confirmation(args) -> None:
    split = load_split_manifest(args.split_manifest)
    assert_disjoint_confirmation(
        [row["chemical_pair_id"] for row in split["records"]],
        args.confirmation_manifest,
    )


def lock_splits(args) -> None:
    rows = read_rows(args.panel)
    assignment = locked_reaction_split([row["reaction_identity"] for row in rows])
    pair_by_identity = {
        row["reaction_identity"]: chemical_pair_id(row["source_identity"], row["target_identity"])
        for row in rows
    }
    exclusion = assert_disjoint_confirmation(pair_by_identity.values(), args.confirmation_manifest)
    records = [
        {"reaction_identity": key, "chemical_pair_id": pair_by_identity[key], "split": assignment[key]}
        for key in sorted(assignment)
    ]
    value = {
        "type": "latent_predictability_reaction_splits",
        "salt": "latent-decoder-audit-v1",
        "panel": str(args.panel.resolve()),
        "panel_sha256": sha256_file(args.panel),
        "counts": {name: sum(row["split"] == name for row in records) for name in ("train", "validation", "test")},
        "confirmation_exclusion": exclusion,
        "records": records,
    }
    if args.split_manifest.exists():
        existing = json.loads(args.split_manifest.read_text(encoding="utf-8"))
        if existing != value:
            raise ValueError("locked latent-audit split changed")
    write_json(args.split_manifest, value)
    print(json.dumps({"stage": "splits_locked", "counts": value["counts"]}), flush=True)


def selected_specs(keys: str | None = None):
    specs = [spec for spec in checkpoint_specs() if spec.key in PRIMARY_KEYS]
    if keys:
        wanted = {item.strip() for item in keys.split(",") if item.strip()}
        specs = [spec for spec in specs if spec.key in wanted]
        if wanted != {spec.key for spec in specs}:
            raise ValueError(f"unknown/non-primary keys: {sorted(wanted - {spec.key for spec in specs})}")
    for spec in specs:
        config_path = ROOT / spec.checkpoint / "USPTO-MIT-Synthesis/adapter_config.json"
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if int(config["r"]) != 8 or int(config["lora_alpha"]) != 8:
            raise ValueError(f"rank/alpha mismatch for {spec.key}")
    return sorted(specs, key=lambda spec: spec.key)


def token_metadata(example) -> dict[str, dict]:
    return {
        str(token.index): {
            "token_class": token.token_class,
            "component": token.component,
            "segment_rank": token.segment_rank,
            "events": sorted(token.events),
            "event_details": dict(token.details),
            "around_reaction_center": "reaction_center" in token.events,
        }
        for token in example.tokens if token.segment != "marker"
    }


def build_examples(tokenizer, rows: list[dict], split_assignment: dict[str, str]) -> list[tuple[object, str]]:
    result = []
    for index, row in enumerate(rows):
        adapted = {
            "reaction_identity": row["reaction_identity"],
            "canonical_source": row["source"],
            "canonical_target": row["target"],
        }
        result.append((annotate_example(tokenizer, adapted, index, "latent_probe_1024"), split_assignment[row["reaction_identity"]]))
    return result


def extract_records(model, capture, examples, device: str, batch_size: int) -> list[dict]:
    records = []
    ordered = sorted(examples, key=lambda item: len(item[0].input_ids))
    with torch.inference_mode():
        for start in range(0, len(ordered), batch_size):
            batch = ordered[start:start + batch_size]
            maximum = max(len(example.input_ids) for example, _ in batch)
            ids = torch.zeros((len(batch), maximum), dtype=torch.long, device=device)
            mask = torch.zeros_like(ids, dtype=torch.bool)
            for row_index, (example, _) in enumerate(batch):
                length = len(example.input_ids)
                ids[row_index, :length] = torch.tensor(example.input_ids, device=device)
                mask[row_index, :length] = True
            capture.clear()
            model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
            for row_index, (example, split) in enumerate(batch):
                length = len(example.input_ids)
                records.append({
                    "panel_index": example.panel_index,
                    "reaction_identity": example.reaction_identity,
                    "split": split,
                    "source": example.source,
                    "target": example.target,
                    "input_ids": torch.tensor(example.input_ids, dtype=torch.int32),
                    "source_indices": [token.index for token in example.tokens if token.segment == "source"],
                    "product_indices": [token.index for token in example.tokens if token.segment == "target"],
                    "token_metadata": token_metadata(example),
                    "reaction_center_metadata": example.reaction_center_metadata,
                    "states": {
                        name: capture.values[name][row_index, :length].to(torch.bfloat16).cpu()
                        for name in LAYERS
                    },
                })
    return sorted(records, key=lambda row: row["panel_index"])


def extract(args) -> None:
    split_manifest = load_split_manifest(args.split_manifest)
    assignment = {row["reaction_identity"]: row["split"] for row in split_manifest["records"]}
    # Recheck at every expensive entrypoint, not only when the split file is made.
    assert_disjoint_confirmation(
        [row["chemical_pair_id"] for row in split_manifest["records"]],
        args.confirmation_manifest,
    )
    rows = read_rows(args.panel)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab = len(tokenizer)
    add_predictor_tokens(tokenizer)
    examples = build_examples(tokenizer, rows, assignment)
    specs = selected_specs(args.keys)
    model = load_lora_model(
        MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab,
        attention_dropout=0.0, attn_implementation="sdpa", lora_rank=8, lora_alpha=8,
    ).to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    capture = SelectedStateCapture(model)
    metadata = []
    for index, spec in enumerate(specs, 1):
        destination = args.output / "cache/canonical" / f"{spec.key}.pt"
        if destination.exists() and not args.overwrite:
            print(json.dumps({"stage": "cache_reused", "key": spec.key}), flush=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        load_adapter_checkpoint(model, ROOT / spec.checkpoint)
        parameter_fingerprint = _sampled_parameter_fingerprint(model)
        before = time.perf_counter()
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        records = extract_records(model, capture, examples, args.device, args.batch_size)
        payload = {
            "type": "latent_predictability_state_cache",
            "spec": spec.__dict__, "layers": LAYERS, "records": records,
            "checkpoint_sha256": sha256_file(ROOT / spec.checkpoint / "USPTO-MIT-Synthesis/adapter_model.safetensors"),
            "panel_sha256": sha256_file(args.panel),
        }
        torch.save(payload, destination)
        after_fingerprint = _sampled_parameter_fingerprint(model)
        if after_fingerprint != parameter_fingerprint:
            raise RuntimeError("model parameters changed during frozen extraction")
        row = {
            "key": spec.key, "seconds": time.perf_counter() - before,
            "bytes": destination.stat().st_size,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else 0,
            "parameter_fingerprint": parameter_fingerprint,
        }
        metadata.append(row)
        print(json.dumps({"stage": "checkpoint_extracted", "index": index, "total": len(specs), **row}), flush=True)
        del records, payload
        gc.collect()
    capture.close()
    write_json(args.output / "extraction.json", {
        "type": "latent_predictability_extraction", "torch": torch.__version__,
        "python": platform.python_version(), "checkpoints": metadata,
    })


def predict_batches(model, values: torch.Tensor, device: str, batch_size: int) -> torch.Tensor:
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            output.append(model(values[start:start + batch_size].to(device)).float().cpu())
    return torch.cat(output) if output else torch.empty((0, 0))


def load_probe_predictions(path: Path, values: torch.Tensor, kind: str, device: str, batch_size: int):
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    predictions = probe_predictions_from_artifact(
        artifact, values, device, batch_size, kinds=(kind,),
    )
    return predictions[kind], artifact


def probe_predictions_from_artifact(
    artifact: dict, values: torch.Tensor, device: str, batch_size: int,
    kinds=("constant", "ridge", "residual_mlp"),
) -> dict[str, torch.Tensor]:
    """Evaluate requested probes after one artifact load/standardization."""
    standardizer, basis = artifact["standardizer"], artifact["basis"]
    standardized = standardizer(values)
    output = {}
    if "constant" in kinds:
        output["constant"] = basis.mean.expand(len(values), -1).clone()
    learned = set(kinds) & {"ridge", "residual_mlp"}
    if not learned:
        return output
    ridge = RidgeProbe(artifact["input_size"], artifact["output_size"])
    ridge.load_state_dict(artifact["ridge_state"])
    if "ridge" in learned:
        output["ridge"] = basis.decode(
            predict_batches(ridge.to(device), standardized, device, batch_size)
        )
        ridge = ridge.cpu()
    if "residual_mlp" in learned:
        model = ResidualMLPProbe(
            ridge, artifact["input_size"], artifact["output_size"], artifact["mlp_width"],
        )
        model.load_state_dict(artifact["mlp_state"])
        model = model.to(device)
        output["residual_mlp"] = basis.decode(
            predict_batches(model, standardized, device, batch_size)
        )
    unknown = set(kinds) - {"constant", "ridge", "residual_mlp"}
    if unknown:
        raise ValueError(sorted(unknown))
    return output


def support_flags(metadata: list[dict]) -> dict[str, np.ndarray]:
    return {
        "arbitrary": np.ones(len(metadata), dtype=bool),
        "common_k8_eligible": np.asarray([
            int(row.get("current_segment_rank", -1)) <= int(row.get("segment_length", 0)) - 9
            for row in metadata
        ]),
        "atom_to_atom": np.asarray([bool(row.get("atom_to_atom")) for row in metadata]),
        "event_to_next_event": np.asarray([bool(row.get("event_to_next_event")) for row in metadata]),
        "component_boundary": np.asarray([bool(row.get("component_boundary")) for row in metadata]),
        "reaction_center_window": np.asarray([bool(row.get("around_reaction_center")) for row in metadata]),
    }


def reaction_metric_map(target, prediction, training_mean, metadata, mask_np=None):
    if mask_np is None:
        mask_np = np.ones(len(metadata), dtype=bool)
    grouped = {}
    for index in np.flatnonzero(mask_np):
        grouped.setdefault(metadata[index]["reaction_identity"], []).append(int(index))
    return {
        reaction: latent_metrics(
            target[torch.tensor(indices, dtype=torch.long)],
            prediction[torch.tensor(indices, dtype=torch.long)], training_mean,
        )
        for reaction, indices in sorted(grouped.items())
    }


def fit_probes(args) -> None:
    assert_disjoint_confirmation(
        [row["chemical_pair_id"] for row in load_split_manifest(args.split_manifest)["records"]],
        args.confirmation_manifest,
    )
    result_path = args.output / "raw/probe_metrics.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.resume and result_path.exists():
        completed = {
            json.loads(line)["key"] for line in result_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    handle = result_path.open("a" if args.resume else "w", encoding="utf-8")
    device = args.device
    for spec in selected_specs(args.keys):
        payload = torch.load(args.output / "cache/canonical" / f"{spec.key}.pt", map_location="cpu", weights_only=False)
        records_by_split = {
            split: [row for row in payload["records"] if row["split"] == split]
            for split in ("train", "validation", "test")
        }
        basis_cache = {}
        for layer in LAYERS:
            for segment in ("source", "product"):
                for horizon in HORIZONS:
                    for mode in ("current", "history"):
                        key = f"{spec.key}__{layer}__{segment}__k{horizon}__{mode}"
                        if key in completed:
                            print(json.dumps({"stage": "probe_reused", "key": key}), flush=True)
                            continue
                        matrices = {
                            split: forecast_matrices(records_by_split[split], layer, segment, horizon, mode)
                            for split in records_by_split
                        }
                        if min(len(matrices[name][0]) for name in matrices) == 0:
                            continue
                        tx, ty, tm = matrices["train"]
                        chosen = reaction_balanced_indices(tm, args.train_positions, args.probe_seed)
                        tx, ty = tx[chosen], ty[chosen]
                        vx, vy, vm = matrices["validation"]
                        ex, ey, em = matrices["test"]
                        standardizer = Standardizer.fit(tx)
                        txs, vxs, exs = standardizer(tx), standardizer(vx), standardizer(ex)
                        basis_key = (layer, segment, horizon)
                        if basis_key not in basis_cache:
                            basis_cache[basis_key] = TargetBasis.fit(
                                ty, args.pca_rank, seed=args.probe_seed + horizon
                            )
                        basis = basis_cache[basis_key]
                        tys, vys = basis.encode(ty), basis.encode(vy)
                        ridge = RidgeProbe(txs.shape[1], tys.shape[1]).to(device)
                        ridge, ridge_fit = fit_probe(
                            ridge, txs, tys, vxs, vys, weight_decay=args.ridge,
                            epochs=args.epochs, batch_size=args.probe_batch_size, seed=args.probe_seed,
                        )
                        ridge_scores = predict_batches(ridge, exs, device, args.probe_batch_size)
                        ridge_prediction = basis.decode(ridge_scores)
                        mlp = ResidualMLPProbe(ridge, txs.shape[1], tys.shape[1], args.mlp_width).to(device)
                        mlp, mlp_fit = fit_probe(
                            mlp, txs, tys, vxs, vys, weight_decay=args.mlp_decay,
                            epochs=args.epochs, batch_size=args.probe_batch_size, seed=args.probe_seed,
                        )
                        mlp_prediction = basis.decode(predict_batches(mlp, exs, device, args.probe_batch_size))
                        mlp_seed_robustness = []
                        if layer == "final_post_norm" and segment == "product":
                            for extra_seed in (args.probe_seed + 1, args.probe_seed + 2):
                                replicate = ResidualMLPProbe(
                                    ridge, txs.shape[1], tys.shape[1], args.mlp_width
                                ).to(device)
                                replicate, replicate_fit = fit_probe(
                                    replicate, txs, tys, vxs, vys,
                                    weight_decay=args.mlp_decay, epochs=args.epochs,
                                    batch_size=args.probe_batch_size, seed=extra_seed,
                                )
                                replicate_prediction = basis.decode(
                                    predict_batches(replicate, exs, device, args.probe_batch_size)
                                )
                                mlp_seed_robustness.append({
                                    "seed": extra_seed,
                                    "fit": replicate_fit,
                                    "metrics": latent_metrics(ey, replicate_prediction, basis.mean),
                                })
                                del replicate, replicate_prediction
                        constant = basis.mean.expand_as(ey)
                        predictions = {"constant": constant, "ridge": ridge_prediction, "residual_mlp": mlp_prediction}
                        validation_predictions = {
                            "constant": basis.mean.expand_as(vy),
                            "ridge": basis.decode(predict_batches(ridge, vxs, device, args.probe_batch_size)),
                            "residual_mlp": basis.decode(predict_batches(mlp, vxs, device, args.probe_batch_size)),
                        }
                        flags = support_flags(em)
                        shuffled_y, shuffled_donors = shuffled_reaction_targets(
                            ey, em, args.probe_seed + horizon,
                        )
                        metrics = {}
                        reaction_metrics = {}
                        for name, prediction in predictions.items():
                            metrics[name] = {}
                            reaction_metrics[name] = {}
                            for support, mask_np in flags.items():
                                mask = torch.from_numpy(mask_np)
                                if mask.any():
                                    support_indices = np.flatnonzero(mask_np)
                                    chosen_ids = [em[index]["reaction_identity"] for index in support_indices]
                                    metrics[name][support] = latent_metrics(
                                        ey[mask], prediction[mask], basis.mean, chosen_ids
                                    )
                                    reaction_metrics[name][support] = reaction_metric_map(
                                        ey, prediction, basis.mean, em, mask_np
                                    )
                            metrics[name]["shuffled_reaction"] = latent_metrics(
                                shuffled_y, prediction, basis.mean,
                            )
                        metrics["nonlinear_r2_improvement"] = (
                            metrics["residual_mlp"]["arbitrary"]["r2"] - metrics["ridge"]["arbitrary"]["r2"]
                        )
                        artifact = {
                            "key": key, "spec": spec.__dict__, "layer": layer, "segment": segment,
                            "horizon": horizon, "mode": mode, "basis": basis,
                            "standardizer": standardizer, "ridge_state": {k: v.cpu() for k,v in ridge.state_dict().items()},
                            "mlp_state": {k: v.cpu() for k,v in mlp.state_dict().items()},
                            "input_size": txs.shape[1], "output_size": tys.shape[1], "mlp_width": args.mlp_width,
                        }
                        probe_path = args.output / "probes" / f"{key}.pt"
                        probe_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(artifact, probe_path)
                        record = {
                            "key": key, "checkpoint": spec.key, "layer": layer, "segment": segment,
                            "horizon": horizon, "mode": mode, "pca_variance_coverage": basis.variance_coverage,
                            "train_positions": len(tx), "validation_positions": len(vx), "test_positions": len(ex),
                            "ridge_fit": ridge_fit, "mlp_fit": mlp_fit, "metrics": metrics,
                            "reaction_metrics": reaction_metrics,
                            "validation_reaction_metrics": {
                                name: reaction_metric_map(vy, prediction, basis.mean, vm)
                                for name, prediction in validation_predictions.items()
                            },
                            "mlp_seed_robustness": mlp_seed_robustness,
                            "shuffled_self_matches": int(sum(
                                em[i]["reaction_identity"] == em[int(donor)]["reaction_identity"]
                                for i, donor in enumerate(shuffled_donors.tolist())
                            )),
                        }
                        handle.write(json.dumps(record) + "\n")
                        handle.flush()
                        print(json.dumps({"stage": "probe_complete", "key": key}), flush=True)
                        del tx, ty, vx, vy, ex, ey, ridge, mlp, predictions
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
    handle.close()


def graph_token_atoms(example, segment: str, canonical_smiles: str) -> dict[int, list[int]]:
    """Map serialized token indices to canonical graph atoms via RDKit correspondence."""
    value = example.source if segment == "source" else example.target
    correspondence = canonical_atom_correspondence(canonical_smiles, value)
    spans = _atom_spans(value)
    segment_tokens = [token for token in example.tokens if token.segment == ("source" if segment == "source" else "target")]
    segment_start = min(token.start for token in segment_tokens)
    result = {}
    for view_atom, (start, end) in enumerate(spans):
        for token in segment_tokens:
            relative_start, relative_end = token.start - segment_start, token.end - segment_start
            if relative_start < end and relative_end > start:
                result.setdefault(token.index, []).append(correspondence[view_atom])
    return result


def extract_views(args) -> None:
    """Extract deterministic cross-view states and graph-alignment metadata."""
    split_manifest = load_split_manifest(args.split_manifest)
    assignment = {row["reaction_identity"]: row["split"] for row in split_manifest["records"]}
    assert_disjoint_confirmation(
        [row["chemical_pair_id"] for row in split_manifest["records"]],
        args.confirmation_manifest,
    )
    rows = [row for row in read_rows(args.panel) if assignment[row["reaction_identity"]] in {"validation", "test"}]
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab = len(tokenizer)
    add_predictor_tokens(tokenizer)
    expanded = []
    for panel_index, row in enumerate(rows):
        sources = deterministic_random_smiles(row["source_identity"], VIEW_SEEDS)
        targets = deterministic_random_smiles(row["target_identity"], VIEW_SEEDS)
        for view, (source, target) in enumerate(zip(sources, targets)):
            adapted = {"reaction_identity": row["reaction_identity"], "canonical_source": source, "canonical_target": target}
            example = annotate_example(tokenizer, adapted, panel_index * 5 + view, "latent_cross_view")
            expanded.append((example, assignment[row["reaction_identity"]], panel_index, view, row["source_identity"], row["target_identity"]))
    specs = selected_specs(args.keys)
    model = load_lora_model(MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab, attention_dropout=0.0, attn_implementation="sdpa", lora_rank=8, lora_alpha=8).to(args.device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    capture = SelectedStateCapture(model)
    for spec in specs:
        load_adapter_checkpoint(model, ROOT / spec.checkpoint)
        simple = [(item[0], item[1]) for item in expanded]
        records = extract_records(model, capture, simple, args.device, args.batch_size)
        for record, item in zip(sorted(records, key=lambda r:r["panel_index"]), sorted(expanded, key=lambda x:x[0].panel_index)):
            example, _, identity_index, view, canonical_source, canonical_target = item
            record.update({
                "identity_index": identity_index, "view": view,
                "canonical_source": canonical_source, "canonical_product": canonical_target,
                "source_atom_map": graph_token_atoms(example, "source", canonical_source),
                "product_atom_map": graph_token_atoms(example, "product", canonical_target),
            })
        destination = args.output / "cache/views" / f"{spec.key}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"type":"latent_cross_view_cache", "spec":spec.__dict__, "records":records}, destination)
        print(json.dumps({"stage":"views_extracted", "key":spec.key, "records":len(records)}), flush=True)
    capture.close()


def extract_candidates(args) -> None:
    """Replay the already archived beams; never regenerate or rescore candidates."""
    split_manifest = load_split_manifest(args.split_manifest)
    assert_disjoint_confirmation([row["chemical_pair_id"] for row in split_manifest["records"]], args.confirmation_manifest)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab = len(tokenizer); add_predictor_tokens(tokenizer)
    model = load_lora_model(MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab, attention_dropout=0.0, attn_implementation="sdpa", lora_rank=8, lora_alpha=8).to(args.device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    capture = SelectedStateCapture(model)
    for spec in selected_specs(args.keys):
        workload = [
            row for row in _candidate_workload(tokenizer, spec.key)
            if row["role"] == "gold" or row["valid_candidate"]
        ]
        examples = []
        for index, row in enumerate(workload):
            adapted = {"reaction_identity": row["reaction_identity"], "canonical_source": row["source"], "canonical_target": row["candidate"]}
            examples.append((annotate_example(tokenizer, adapted, index, "archived_candidate"), "beam"))
        load_adapter_checkpoint(model, ROOT / spec.checkpoint)
        records = extract_records(model, capture, examples, args.device, args.batch_size)
        for record, beam in zip(records, workload):
            record["states"] = {"final_post_norm": record["states"]["final_post_norm"]}
            record["beam_metadata"] = {key:value for key,value in beam.items() if key not in {"input_ids","source_positions","target_positions"}}
        destination = args.output / "cache/candidates" / f"{spec.key}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"type":"latent_candidate_cache", "spec":spec.__dict__, "records":records}, destination)
        print(json.dumps({"stage":"candidates_extracted", "key":spec.key, "records":len(records)}), flush=True)
    capture.close()


def analyze_views(args) -> None:
    """Reduce graph-aligned atom states into within/between invariance metrics."""
    recheck_confirmation(args)
    output_path = args.output / "raw/invariance.jsonl"
    reaction_path = args.output / "raw/invariance_reaction.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle, reaction_path.open("w", encoding="utf-8", newline="\n") as reaction_handle:
        for spec in selected_specs(args.keys):
            payload = torch.load(args.output / "cache/views" / f"{spec.key}.pt", map_location="cpu", weights_only=False)
            records = payload["records"]
            for layer in LAYERS:
                for segment in ("source", "product"):
                    map_key = f"{segment}_atom_map"
                    canonical_key = "canonical_source" if segment == "source" else "canonical_product"
                    for object_type in ("atom", "motif", "component"):
                        objects: dict[tuple, dict[int, list[torch.Tensor]]] = {}
                        identity_metadata = {
                            record["identity_index"]: (record["reaction_identity"], record["split"])
                            for record in records
                        }
                        for record in records:
                            states = record["states"][layer].float()
                            token_atoms = {int(key): set(map(int, atoms)) for key, atoms in record[map_key].items()}
                            canonical = Chem.MolFromSmiles(record[canonical_key])
                            if object_type == "atom":
                                supports = [("atom", atom, {atom}) for atom in range(canonical.GetNumAtoms())]
                            elif object_type == "motif":
                                supports = [
                                    ("motif", name, tuple(match), set(match))
                                    for name, query in MOTIF_QUERIES.items()
                                    for match in canonical.GetSubstructMatches(query, uniquify=True)
                                ]
                            else:
                                supports = [("component", tuple(sorted(fragment)), set(fragment)) for fragment in Chem.GetMolFrags(canonical)]
                            for support in supports:
                                atom_set = support[-1]
                                state_rows = [states[token] for token, atoms in token_atoms.items() if atoms & atom_set]
                                if state_rows:
                                    key = (record["identity_index"], *support[:-1])
                                    objects.setdefault(key, {}).setdefault(record["view"], []).append(torch.stack(state_rows).mean(0))
                        for split in ("validation", "test"):
                            aligned_items = [
                                (key, torch.stack([torch.stack(views[view]).mean(0) for view in range(5)]))
                                for key, views in objects.items()
                                if set(views) == set(range(5)) and identity_metadata[key[0]][1] == split
                            ]
                            if not aligned_items:
                                continue
                            values = torch.stack([value for _, value in aligned_items])
                            result = {
                                "checkpoint": spec.key, "layer": layer, "segment": segment,
                                "object": object_type, "split": split, **invariance_metrics(values),
                            }
                            handle.write(json.dumps(result) + "\n")
                            by_reaction = {}
                            for key, value in aligned_items:
                                by_reaction.setdefault(identity_metadata[key[0]][0], []).append(value)
                            for reaction_identity, reaction_values in sorted(by_reaction.items()):
                                if len(reaction_values) < 2:
                                    continue
                                reaction_handle.write(json.dumps({
                                    "checkpoint": spec.key, "layer": layer, "segment": segment,
                                    "object": object_type, "split": split,
                                    "reaction_identity": reaction_identity,
                                    **invariance_metrics(torch.stack(reaction_values)),
                                }) + "\n")
                            print(json.dumps({"stage":"invariance_complete", "checkpoint":spec.key, "layer":layer, "segment":segment, "object":object_type, "split":split}), flush=True)


def beam_covariates(row: dict) -> dict:
    candidate_sets = [set(value for value in view if value) for view in row["canonical_candidates_by_view"]]
    jaccard = []
    for left in range(5):
        for right in range(left + 1, 5):
            union = candidate_sets[left] | candidate_sets[right]
            jaccard.append(len(candidate_sets[left] & candidate_sets[right]) / len(union) if union else 1.0)
    gold_ranks = []
    for view in row["canonical_candidates_by_view"]:
        gold_ranks.append(view.index(row["target"]) + 1 if row["target"] in view else None)
    aggregate_correct = bool(row["ranked_candidates"][0] == row["target"])
    view_gold_top1 = [rank == 1 for rank in gold_ranks]
    return {
        "candidate_jaccard": float(np.mean(jaccard)),
        "per_view_gold_rank": gold_ranks,
        "views_gold_top1": int(sum(view_gold_top1)),
        "aggregate_gold_rank": row["ranked_candidates"].index(row["target"]) + 1 if row["target"] in row["ranked_candidates"] else None,
        "aggregate_gold_score": float(row["rank_scores"].get(row["target"], 0.0)),
        "aggregate_correct": aggregate_correct,
        "cross_view_aggregation_failure": bool(any(view_gold_top1) and not aggregate_correct),
        "within_view_ranking_failure": bool(not any(view_gold_top1) and not aggregate_correct),
    }


def _pooled_graph_states(record: dict, example, layer: str, segment: str, canonical_smiles: str) -> torch.Tensor:
    mapping = graph_token_atoms(example, segment, canonical_smiles)
    states = record["states"][layer].float()
    molecule = Chem.MolFromSmiles(canonical_smiles)
    rows = []
    for atom in range(molecule.GetNumAtoms()):
        tokens = [token for token, atoms in mapping.items() if atom in atoms]
        if not tokens:
            raise ValueError("graph atom has no aligned token state")
        rows.append(states[tokens].mean(0))
    return torch.stack(rows)


def extract_development_view_invariance(args) -> None:
    """Join graph-aligned five-view states to the archived generation endpoint."""
    groups = [json.loads(line) for line in DEVELOPMENT_PANEL.read_text(encoding="utf-8").splitlines() if line]
    valid_groups = [row for row in groups if row.get("canonical_source")]
    assert_disjoint_confirmation(
        [chemical_pair_id(row["canonical_source"], row["canonical_target"]) for row in valid_groups],
        args.confirmation_manifest,
        [row["reaction_identity"] for row in valid_groups],
    )
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab = len(tokenizer); add_predictor_tokens(tokenizer)
    expanded = []
    for group_index, row in enumerate(valid_groups):
        for view in range(5):
            adapted = {
                "reaction_identity": row["reaction_identity"],
                "canonical_source": row["sources"][view],
                "canonical_target": row["targets"][view],
            }
            example = annotate_example(
                tokenizer, adapted, group_index * 5 + view,
                "archived_five_view", infer_reaction_center=False,
            )
            expanded.append((example, "dev", group_index, view, row))
    model = load_lora_model(
        MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab,
        attention_dropout=0.0, attn_implementation="sdpa", lora_rank=8, lora_alpha=8,
    ).to(args.device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    capture = SelectedStateCapture(model)
    for spec in selected_specs(args.keys):
        destination = args.output / "raw/development_invariance" / f"{spec.key}.jsonl"
        if destination.exists():
            print(json.dumps({"stage":"development_invariance_reused", "key":spec.key}), flush=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        predictions = {row["reaction_identity"]: row for row in [json.loads(line) for line in prediction_path_for_key(spec.key).read_text(encoding="utf-8").splitlines() if line]}
        load_adapter_checkpoint(model, ROOT / spec.checkpoint)
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            for start in range(0, len(expanded), args.batch_size * 5):
                chunk = expanded[start:start + args.batch_size * 5]
                records = extract_records(
                    model, capture, [(item[0], item[1]) for item in chunk],
                    args.device, args.batch_size,
                )
                by_index = {record["panel_index"]: record for record in records}
                for offset in range(0, len(chunk), 5):
                    views = chunk[offset:offset + 5]
                    if len(views) != 5: continue
                    group = views[0][4]
                    beam = beam_covariates(predictions[group["reaction_identity"]])
                    for layer in LAYERS:
                        for segment, canonical_key in (("source", "canonical_source"), ("product", "canonical_target")):
                            canonical_smiles = group[canonical_key]
                            atom_views = []
                            for example, _, _, _, _ in views:
                                atom_views.append(_pooled_graph_states(
                                    by_index[example.panel_index], example, layer, segment, canonical_smiles
                                ))
                            atom_values = torch.stack(atom_views, dim=1)
                            molecule = Chem.MolFromSmiles(canonical_smiles)
                            supports = {
                                "atom": [{atom} for atom in range(molecule.GetNumAtoms())],
                                "motif": [set(match) for query in MOTIF_QUERIES.values() for match in molecule.GetSubstructMatches(query, uniquify=True)],
                                "component": [set(fragment) for fragment in Chem.GetMolFrags(molecule)],
                            }
                            for object_type, atom_sets in supports.items():
                                if not atom_sets: continue
                                values = torch.stack([atom_values[sorted(atom_set)].mean(0) for atom_set in atom_sets])
                                output = {
                                    "checkpoint": spec.key, "seed": spec.seed,
                                    "reaction_identity": group["reaction_identity"],
                                    "panel_index": int(group["panel_index"]),
                                    "layer": layer, "segment": segment, "object": object_type,
                                    **invariance_metrics(values), **beam,
                                }
                                handle.write(json.dumps(output) + "\n")
                handle.flush()
                del records, by_index
        print(json.dumps({"stage":"development_invariance_complete", "key":spec.key}), flush=True)
    capture.close()


@torch.inference_mode()
def score_decoder(args) -> None:
    """Score decoder preservation with one suffix-cache build per reaction/layer.

    A reference implementation rebuilt the identical frozen prefix cache for
    every segment/horizon/history cell.  Here all preregistered cells are
    materialized first and replayed while that reaction's exact cache is live.
    Probe predictions, sampled positions, BF16 states, and replay mathematics
    are unchanged; only invariant cache construction is hoisted.
    """
    recheck_confirmation(args)
    output_path = args.output / "raw/decoder_metrics.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab = len(tokenizer); add_predictor_tokens(tokenizer)
    model = load_lora_model(MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab, attention_dropout=0.0, attn_implementation="sdpa", lora_rank=8, lora_alpha=8).to(args.device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    llama = find_llama(model)
    layer_numbers = {"layer_6": 6, "layer_16": 16, "layer_21": 21}
    with output_path.open("w", encoding="utf-8") as handle:
        for spec in selected_specs(args.keys):
            load_adapter_checkpoint(model, ROOT / spec.checkpoint)
            payload = torch.load(args.output / "cache/canonical" / f"{spec.key}.pt", map_location="cpu", weights_only=False)
            test_records = [row for row in payload["records"] if row["split"] == "test"]
            record_map = {row["reaction_identity"]: row for row in test_records}
            decoder_records = sorted(
                test_records,
                key=lambda row: hashlib.sha256(
                    f"decoder-replay-v2|{args.probe_seed}|{row['reaction_identity']}".encode()
                ).digest(),
            )[: min(args.decoder_reactions, len(test_records))]
            for layer in LAYERS:
                if args.layers and layer not in args.layers.split(","):
                    continue
                layer_records = (
                    test_records if layer == "final_post_norm"
                    else decoder_records
                )
                layer_state_cache = {
                    index: record["states"][layer].float()
                    for index, record in enumerate(layer_records)
                }
                cells = {}
                for segment in ("source", "product"):
                    for horizon in HORIZONS:
                        for mode in ("current", "history"):
                            plan, metadata = forecast_plan(
                                layer_records, segment, horizon
                            )
                            if not plan:
                                continue
                            if layer != "final_post_norm":
                                cap = (
                                    args.decoder_positions if segment == "product"
                                    else max(1, args.decoder_positions // 2)
                                )
                                sample = reaction_balanced_indices(
                                    metadata, cap, args.probe_seed
                                )
                                selected = set(sample.tolist())
                                rare = support_flags(metadata)
                                for offset, support in enumerate((
                                    "event_to_next_event", "component_boundary",
                                    "reaction_center_window",
                                ), 1):
                                    eligible = np.flatnonzero(rare[support]).tolist()
                                    if not eligible:
                                        continue
                                    local_metadata = [metadata[index] for index in eligible]
                                    local = reaction_balanced_indices(
                                        local_metadata, args.decoder_rare_positions,
                                        args.probe_seed + offset,
                                    )
                                    selected.update(eligible[index] for index in local.tolist())
                                indices = sorted(selected)
                                plan = [plan[index] for index in indices]
                                metadata = [metadata[index] for index in indices]
                            x, y = materialize_forecast_plan(
                                layer_records, layer, mode, plan,
                                state_cache=layer_state_cache,
                            )
                            key = f"{spec.key}__{layer}__{segment}__k{horizon}__{mode}"
                            artifact = torch.load(
                                args.output / "probes" / f"{key}.pt",
                                map_location="cpu", weights_only=False,
                            )
                            predictions = probe_predictions_from_artifact(
                                artifact, x, args.device, args.probe_batch_size,
                            )
                            cells[(segment, horizon, mode)] = {
                                "key": key, "x": x, "y": y, "metadata": metadata,
                                "predictions": predictions,
                                "parity_max_abs": 0.0,
                                "parity_max_rms": 0.0,
                                "parity_min_cosine": 1.0,
                                "parity_top1_mismatches": 0,
                            }

                if layer == "final_post_norm":
                    for cell in cells.values():
                        cell["true_final"] = cell["y"].to(args.device)
                        cell["predicted_final"] = {
                            kind: value.to(args.device)
                            for kind, value in cell["predictions"].items()
                        }
                else:
                    # Index every requested replay by reaction, then build the
                    # invariant remaining-block cache once for that reaction.
                    work = defaultdict(list)
                    for cell_key, cell in cells.items():
                        cell["true_rows"] = []
                        cell["reference_rows"] = []
                        cell["predicted_rows"] = {
                            kind: [] for kind in cell["predictions"]
                        }
                        for index, meta in enumerate(cell["metadata"]):
                            work[meta["reaction_identity"]].append((cell_key, index))
                    for reaction_identity in sorted(work):
                        record = record_map[reaction_identity]
                        layer_input = record["states"][layer].to(args.device).unsqueeze(0)
                        cache, _ = build_suffix_cache(
                            llama, layer_numbers[layer], layer_input
                        )
                        # Preserve the validated reference batch shape: one
                        # true state plus the three probe alternatives. The
                        # cache itself remains hoisted across every cell for
                        # this reaction/layer, which is the large exact saving.
                        for cell_key, index in sorted(
                            work[reaction_identity],
                            key=lambda item: (
                                cells[item[0]]["metadata"][item[1]]["future_index"],
                                item[0], item[1],
                            ),
                        ):
                            cell = cells[cell_key]
                            future_index = int(
                                cell["metadata"][index]["future_index"]
                            )
                            alternatives = torch.stack([
                                cell["y"][index],
                                *(cell["predictions"][kind][index]
                                  for kind in cell["predictions"]),
                            ]).to(device=args.device, dtype=layer_input.dtype)
                            replayed = replay_suffix_from_cache(
                                llama, layer_numbers[layer], cache, alternatives,
                                future_index,
                            )
                            reference = record["states"]["final_post_norm"][
                                future_index
                            ].float().to(args.device)
                            cell["true_rows"].append((index, replayed[0]))
                            cell["reference_rows"].append((index, reference))
                            for offset, kind in enumerate(cell["predictions"], 1):
                                cell["predicted_rows"][kind].append(
                                    (index, replayed[offset])
                                )
                        del cache, layer_input
                    for cell in cells.values():
                        cell["true_final"] = torch.stack([
                            value for _, value in sorted(cell.pop("true_rows"))
                        ])
                        references = torch.stack([
                            value for _, value in sorted(cell.pop("reference_rows"))
                        ])
                        difference = cell["true_final"].float() - references
                        rms = difference.square().mean(-1).sqrt()
                        cosine = F.cosine_similarity(
                            cell["true_final"].float(), references, dim=-1
                        )
                        weight = model.get_output_embeddings().weight[:chemfm_vocab]
                        replay_top1 = (
                            cell["true_final"].to(weight.dtype) @ weight.T
                        ).argmax(-1)
                        reference_top1 = (
                            references.to(weight.dtype) @ weight.T
                        ).argmax(-1)
                        cell["parity_max_abs"] = float(difference.abs().max())
                        cell["parity_max_rms"] = float(rms.max())
                        cell["parity_min_cosine"] = float(cosine.min())
                        cell["parity_top1_mismatches"] = int(
                            replay_top1.ne(reference_top1).sum()
                        )
                        if (cell["parity_max_rms"] > 2e-2
                                or cell["parity_min_cosine"] < 0.999):
                            raise RuntimeError(
                                "true-state suffix replay failed full-forward parity: "
                                f"checkpoint={spec.key} layer={layer} "
                                f"max_abs={cell['parity_max_abs']} "
                                f"max_rms={cell['parity_max_rms']} "
                                f"min_cosine={cell['parity_min_cosine']}"
                            )
                        cell["predicted_final"] = {
                            kind: torch.stack([
                                value for _, value in sorted(values)
                            ])
                            for kind, values in cell.pop("predicted_rows").items()
                        }

                for (segment, horizon, mode), cell in cells.items():
                    metadata = cell["metadata"]
                    gold = torch.tensor(
                        [row["gold_id"] for row in metadata], device=args.device
                    )
                    true_final = cell["true_final"]
                    predicted_final = cell["predicted_final"]
                    weight = model.get_output_embeddings().weight[:chemfm_vocab]
                    true_logits = true_final.to(weight.dtype) @ weight.T
                    flags = support_flags(metadata)
                    for kind in ("constant", "ridge", "residual_mlp"):
                        prediction_logits = predicted_final[kind].to(weight.dtype) @ weight.T
                        metrics = decoder_distribution_metrics(
                            true_logits, prediction_logits, gold
                        )
                        # One device transfer replaces thousands of scalar GPU
                        # synchronizations in support/reaction reductions.
                        metrics_cpu = {
                            name: values.float().cpu()
                            for name, values in metrics.items()
                        }
                        support_metrics = {}
                        reaction_metrics = {}
                        for support, mask_np in flags.items():
                            selected = np.flatnonzero(mask_np)
                            if not len(selected):
                                continue
                            support_metrics[support] = {
                                name: float(values[torch.tensor(selected)].mean())
                                for name, values in metrics_cpu.items()
                            }
                            grouped = {}
                            for index in selected:
                                grouped.setdefault(
                                    metadata[int(index)]["reaction_identity"], []
                                ).append(int(index))
                            reaction_metrics[support] = {
                                reaction: {
                                    name: float(values[torch.tensor(indices)].mean())
                                    for name, values in metrics_cpu.items()
                                }
                                for reaction, indices in sorted(grouped.items())
                            }
                        record = {
                            "checkpoint":spec.key, "layer":layer, "segment":segment,
                            "horizon":horizon, "mode":mode, "probe":kind,
                            "n":len(cell["x"]),
                            "intermediate_replay_reactions": (
                                len(decoder_records) if layer != "final_post_norm" else None
                            ),
                            "intermediate_base_position_cap": (
                                args.decoder_positions if layer != "final_post_norm" else None
                            ),
                            "intermediate_per_rare_support_cap": (
                                args.decoder_rare_positions if layer != "final_post_norm" else None
                            ),
                            "true_state_replay_max_abs_error": cell["parity_max_abs"],
                            "true_state_replay_max_rms_error": cell["parity_max_rms"],
                            "true_state_replay_min_cosine": cell["parity_min_cosine"],
                            "true_state_replay_top1_mismatches": cell["parity_top1_mismatches"],
                            **{name:float(value.mean()) for name,value in metrics_cpu.items()},
                            "supports": support_metrics,
                            "reaction_metrics": reaction_metrics,
                        }
                        handle.write(json.dumps(record) + "\n"); handle.flush()
                        print(json.dumps({
                            "stage":"decoder_complete", "key":cell["key"],
                            "probe":kind,
                        }), flush=True)


def score_candidates(args) -> None:
    """Apply locked validation probes to archived candidate trajectories."""
    recheck_confirmation(args)
    output_path = args.output / "raw/candidate_predictability.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    chemfm_vocab = len(tokenizer); add_predictor_tokens(tokenizer)
    model = load_lora_model(MODEL_DIR, tokenizer, chemfm_vocab_size=chemfm_vocab, attention_dropout=0.0, attn_implementation="sdpa", lora_rank=8, lora_alpha=8).to(args.device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    with output_path.open("w", encoding="utf-8") as handle:
        for spec in selected_specs(args.keys):
            load_adapter_checkpoint(model, ROOT / spec.checkpoint)
            payload = torch.load(args.output / "cache/candidates" / f"{spec.key}.pt", map_location="cpu", weights_only=False)
            records = payload["records"]
            for horizon in HORIZONS:
                for mode in ("current", "history"):
                    key = f"{spec.key}__final_post_norm__product__k{horizon}__{mode}"
                    artifact = torch.load(
                        args.output / "probes" / f"{key}.pt",
                        map_location="cpu", weights_only=False,
                    )
                    for record in records:
                        x, y, metadata = forecast_matrices([record], "final_post_norm", "product", horizon, mode)
                        if not len(x): continue
                        beam = record["beam_metadata"]
                        predictions = probe_predictions_from_artifact(
                            artifact, x, args.device, args.probe_batch_size,
                        )
                        for kind in ("constant", "ridge", "residual_mlp"):
                            prediction = predictions[kind]
                            latent = latent_metrics(y, prediction, artifact["basis"].mean)
                            weight = model.get_output_embeddings().weight[:chemfm_vocab]
                            gold = torch.tensor([row["gold_id"] for row in metadata], device=args.device)
                            functional = decoder_distribution_metrics(
                                y.to(device=args.device, dtype=weight.dtype) @ weight.T,
                                prediction.to(device=args.device, dtype=weight.dtype) @ weight.T,
                                gold,
                            )
                            row = {
                                "checkpoint":spec.key, "reaction_identity":record["reaction_identity"],
                                "panel_index":beam["panel_index"], "view":beam["view"], "role":beam["role"],
                                "candidate":beam["candidate"], "horizon":horizon, "mode":mode, "probe":kind,
                                **{f"latent_{name}":value for name,value in latent.items()},
                                **{f"decoder_{name}":float(value.float().mean()) for name,value in functional.items()},
                            }
                            handle.write(json.dumps(row) + "\n")
                    handle.flush()
                print(json.dumps({"stage":"candidate_scores_complete", "checkpoint":spec.key, "horizon":horizon}), flush=True)


def summarize(args) -> None:
    rows = [json.loads(line) for line in (args.output / "raw/probe_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    def optional_rows(name):
        path = args.output / "raw" / name
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line] if path.exists() else []
    decoder = optional_rows("decoder_metrics.jsonl")
    invariance = optional_rows("invariance.jsonl")
    candidates = optional_rows("candidate_predictability.jsonl")
    candidate_groups = {}
    for row in candidates:
        key = (row["checkpoint"], row["horizon"], row["mode"], row["probe"], row["role"])
        candidate_groups.setdefault(key, []).append(row)
    candidate_summary = [
        {
            "checkpoint": key[0], "horizon": key[1], "mode": key[2], "probe": key[3], "role": key[4],
            "n": len(values),
            "latent_normalized_mse": float(np.mean([row["latent_normalized_mse"] for row in values])),
            "decoder_js": float(np.mean([row["decoder_js"] for row in values])),
            "decoder_kl": float(np.mean([row["decoder_kl_true_predicted"] for row in values])),
        }
        for key, values in sorted(candidate_groups.items())
    ]
    summary = {
        "type": "latent_predictability_audit_summary",
        "cells": len(rows),
        "primary": [row for row in rows if row["segment"] == "product"],
        "decoder": decoder,
        "cross_view_invariance": invariance,
        "candidate_role_summary": candidate_summary,
        "notes": {
            "predictable_fraction": "untruncated R2 relative to the train-target mean",
            "kl_direction": "true decoder distribution || predicted-state distribution",
            "confirmation_outcomes_consumed": False,
        },
    }
    write_json(args.output / "analysis.json", summary)
    plot_dir = args.output / "plots"; plot_dir.mkdir(parents=True, exist_ok=True)
    write_json(plot_dir / "predictability_plot_data.json", [
        {
            "checkpoint":row["checkpoint"], "layer":row["layer"], "horizon":row["horizon"],
            "mode":row["mode"], "ridge_r2":row["metrics"]["ridge"]["arbitrary"]["r2"],
            "mlp_r2":row["metrics"]["residual_mlp"]["arbitrary"]["r2"],
        }
        for row in rows if row["segment"] == "product"
    ])
    write_json(plot_dir / "invariance_plot_data.json", invariance)
    write_json(plot_dir / "decoder_plot_data.json", decoder)
    print(json.dumps({"stage":"summary_complete", "cells":len(rows)}), flush=True)


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    common.add_argument("--confirmation-manifest", type=Path, required=True)
    common.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    common.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    common.add_argument("--device", default="cuda")
    common.add_argument("--keys")
    common.add_argument("--batch-size", type=int, default=8)
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("lock-splits", parents=[common]).set_defaults(function=lock_splits)
    p = sub.add_parser("extract", parents=[common]); p.add_argument("--overwrite", action="store_true"); p.set_defaults(function=extract)
    p = sub.add_parser("extract-views", parents=[common]); p.set_defaults(function=extract_views)
    p = sub.add_parser("extract-candidates", parents=[common]); p.set_defaults(function=extract_candidates)
    p = sub.add_parser("analyze-views", parents=[common]); p.set_defaults(function=analyze_views)
    p = sub.add_parser("extract-development-views", parents=[common]); p.set_defaults(function=extract_development_view_invariance)
    p = sub.add_parser("fit-probes", parents=[common])
    p.add_argument("--pca-rank", type=int, default=256)
    p.add_argument("--train-positions", type=int, default=16384)
    p.add_argument("--probe-seed", type=int, default=20260904)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--probe-batch-size", type=int, default=512)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--mlp-decay", type=float, default=1e-4)
    p.add_argument("--mlp-width", type=int, default=128)
    p.add_argument("--resume", action="store_true")
    p.set_defaults(function=fit_probes)
    p = sub.add_parser("score-decoder", parents=[common])
    p.add_argument("--layers", default="layer_6,layer_16,layer_21,final_post_norm")
    p.add_argument("--decoder-positions", type=int, default=96)
    p.add_argument("--decoder-rare-positions", type=int, default=32)
    p.add_argument("--decoder-reactions", type=int, default=64)
    p.add_argument("--probe-seed", type=int, default=20260904)
    p.add_argument("--probe-batch-size", type=int, default=512)
    p.set_defaults(function=score_decoder)
    p = sub.add_parser("score-candidates", parents=[common])
    p.add_argument("--probe-batch-size", type=int, default=512)
    p.set_defaults(function=score_candidates)
    sub.add_parser("summarize", parents=[common]).set_defaults(function=summarize)
    return root


def main() -> None:
    args = parser().parse_args()
    args.output = args.output.resolve()
    args.split_manifest = args.split_manifest.resolve()
    args.function(args)


if __name__ == "__main__":
    main()
