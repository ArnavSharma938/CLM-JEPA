#!/usr/bin/env python
"""Predictor-only NextLat warmup and frozen-checkpoint objective-gradient audit."""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling import load_rita
from src.nextlat import FaithfulNextLatPredictor, faithful_losses
from src.stats import cluster_bootstrap_mean, sign_flip_pvalue
from src.stp import released_stp_loss
from src.tokenization import collate_encoded, encode_canonical, next_residue_transition_mask


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def iter_cache(cache_dir: Path):
    for path in sorted(cache_dir.glob("shard_*.pt")):
        yield from torch.load(path, map_location="cpu", weights_only=False)


def predictor_examples(cache_dir: Path, split: str, assignments: dict[str, str], cap_per_protein: int = 12):
    examples = []
    for item in iter_cache(cache_dir):
        if assignments[item["id"]] != split:
            continue
        length = item["final"].shape[0]
        positions = np.linspace(0, length - 2, min(cap_per_protein, length - 1), dtype=int)
        for position in np.unique(positions):
            examples.append((
                item["final"][position], item["final"][position + 1],
                int(item["input_ids"][position + 1]), item["id"], item["cluster_id"],
            ))
    return examples


def evaluate_predictor(predictor, examples, embedding, device, batch_size=256):
    predictor.eval()
    losses = []
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            current = torch.stack([x[0] for x in batch]).to(device=device, dtype=torch.float32)
            future = torch.stack([x[1] for x in batch]).to(device=device, dtype=torch.float32)
            ids = torch.tensor([x[2] for x in batch], device=device)
            prediction = predictor(current, embedding[ids].float())
            losses.extend(F.smooth_l1_loss(prediction, future, reduction="none").mean(-1).cpu().tolist())
    return float(np.mean(losses))


def evaluate_predictor_full(predictor, examples, embedding, head, device, batch_size=256):
    predictor.eval(); totals = {"latent": [], "kl": [], "total": []}
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            current = torch.stack([x[0] for x in batch]).to(device=device, dtype=torch.float32)
            future = torch.stack([x[1] for x in batch]).to(device=device, dtype=torch.float32)
            ids = torch.tensor([x[2] for x in batch], device=device)
            mask = torch.ones(len(batch), dtype=torch.bool, device=device)
            latent, kl, _ = faithful_losses(
                predictor, current, future, embedding[ids].float(), mask, head.float()
            )
            totals["latent"].append((float(latent), len(batch)))
            totals["kl"].append((float(kl), len(batch)))
            totals["total"].append((float(latent + kl), len(batch)))
    return {name: sum(value * count for value, count in cells) / sum(count for _, count in cells)
            for name, cells in totals.items()}


def warmup(cache_dir: Path, split_manifest: Path, output: Path, seed: int = 20260914):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = load_rita(device=device)
    embedding = loaded.model.get_input_embeddings().weight.detach()
    assignments = {row["id"]: row["split"] for row in read_jsonl(split_manifest)}
    train = predictor_examples(cache_dir, "train", assignments)
    validation = predictor_examples(cache_dir, "validation", assignments)
    predictor = FaithfulNextLatPredictor(1024).to(device)
    initial = evaluate_predictor(predictor, validation, embedding, device)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=2e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best, best_state, history, patience = initial, copy.deepcopy(predictor.state_dict()), [], 0
    for epoch in range(12):
        predictor.train()
        order = torch.randperm(len(train), generator=generator).tolist()
        epoch_losses = []
        for start in range(0, len(order), 128):
            batch = [train[i] for i in order[start:start + 128]]
            current = torch.stack([x[0] for x in batch]).to(device=device, dtype=torch.float32)
            future = torch.stack([x[1] for x in batch]).to(device=device, dtype=torch.float32)
            ids = torch.tensor([x[2] for x in batch], device=device)
            predicted = predictor(current, embedding[ids].float())
            loss = F.smooth_l1_loss(predicted, future)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            epoch_losses.append(float(loss.detach()))
        value = evaluate_predictor(predictor, validation, embedding, device)
        history.append({"epoch": epoch + 1, "train_smooth_l1": float(np.mean(epoch_losses)), "validation_smooth_l1": value})
        if value < best - 1e-5:
            best, best_state, patience = value, copy.deepcopy(predictor.state_dict()), 0
        else:
            patience += 1
            if patience >= 3:
                break
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "hidden_size": 1024, "seed": seed}, output)
    output.with_suffix(".json").write_text(json.dumps({
        "train_examples": len(train), "validation_examples": len(validation),
        "initial_validation_smooth_l1": initial, "best_validation_smooth_l1": best,
        "relative_validation_improvement": (initial - best) / initial,
        "history": history, "backbone_frozen": True,
    }, indent=2) + "\n", encoding="utf-8")


def warmup_full(
    cache_dir: Path, split_manifest: Path, output: Path, seed: int = 20260914,
    cap_per_protein: int = 12, max_epochs: int = 12,
):
    """Fit a fresh predictor under the complete faithful latent + KL objective."""
    torch.manual_seed(seed); device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = load_rita(device=device); model = loaded.model
    embedding = model.get_input_embeddings().weight.detach(); head = model.get_output_embeddings().weight.detach()
    assignments = {row["id"]: row["split"] for row in read_jsonl(split_manifest)}
    train = predictor_examples(cache_dir, "train", assignments, cap_per_protein)
    validation = predictor_examples(cache_dir, "validation", assignments, cap_per_protein)
    predictor = FaithfulNextLatPredictor(1024).to(device)
    initial = evaluate_predictor_full(predictor, validation, embedding, head, device)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=2e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best, best_state, history, patience = initial["total"], copy.deepcopy(predictor.state_dict()), [], 0
    rita_versions = [parameter._version for parameter in model.parameters()]
    for epoch in range(max_epochs):
        predictor.train(); order = torch.randperm(len(train), generator=generator).tolist()
        epoch_cells = {"latent": [], "kl": [], "total": []}
        for start in range(0, len(order), 128):
            batch = [train[i] for i in order[start:start + 128]]
            current = torch.stack([x[0] for x in batch]).to(device=device, dtype=torch.float32)
            future = torch.stack([x[1] for x in batch]).to(device=device, dtype=torch.float32)
            ids = torch.tensor([x[2] for x in batch], device=device)
            mask = torch.ones(len(batch), dtype=torch.bool, device=device)
            latent, kl, _ = faithful_losses(
                predictor, current, future, embedding[ids].float(), mask, head.float()
            )
            total = latent + kl
            optimizer.zero_grad(set_to_none=True); total.backward(); optimizer.step()
            for name, value in (("latent", latent), ("kl", kl), ("total", total)):
                epoch_cells[name].append(float(value.detach()))
        value = evaluate_predictor_full(predictor, validation, embedding, head, device)
        history.append({"epoch": epoch + 1,
            **{f"train_{name}": float(np.mean(cells)) for name, cells in epoch_cells.items()},
            **{f"validation_{name}": metric for name, metric in value.items()}})
        if value["total"] < best - 1e-5:
            best, best_state, patience = value["total"], copy.deepcopy(predictor.state_dict()), 0
        else:
            patience += 1
            if patience >= 3: break
    predictor.load_state_dict(best_state)
    final = evaluate_predictor_full(predictor, validation, embedding, head, device)
    if rita_versions != [parameter._version for parameter in model.parameters()]:
        raise AssertionError("RITA parameter version changed during predictor-only warmup")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "hidden_size": 1024, "seed": seed,
                "objective": "smooth_l1_plus_teacher_to_student_kl"}, output)
    output.with_suffix(".json").write_text(json.dumps({
        "train_examples": len(train), "validation_examples": len(validation),
        "maximum_transitions_per_protein": cap_per_protein,
        "initial_validation": initial, "final_validation": final,
        "relative_total_improvement": (initial["total"] - final["total"]) / initial["total"],
        "history": history, "selected_epoch": int(np.argmin([row["validation_total"] for row in history])) + 1,
        "backbone_frozen": True, "rita_parameter_versions_unchanged": True,
        "optimizer_parameter_scope": "FaithfulNextLatPredictor only",
        "objective": "unit SmoothL1 + unit teacher-to-student KL",
    }, indent=2) + "\n", encoding="utf-8")


def masked_ntp(outputs, ids, attention):
    mask = next_residue_transition_mask(ids, attention)
    logits = outputs.logits[:, :-1]
    targets = ids[:, 1:]
    return F.cross_entropy(logits[mask].float(), targets[mask])


def group_name(name: str) -> str:
    if "transformer.layers." in name:
        index = int(name.split("transformer.layers.", 1)[1].split(".", 1)[0])
        return "early_0_7" if index < 8 else "middle_8_15" if index < 16 else "late_16_23"
    if "embedding" in name or "lm_head" in name:
        return "tied_embedding_head"
    return "other"


def gradient_pair(model, parameters, names, batch, auxiliary, seed):
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    ntp = masked_ntp(outputs, batch["input_ids"], batch["attention_mask"])
    ntp_grads_gpu = torch.autograd.grad(ntp, parameters, allow_unused=True)
    ntp_grads = [None if g is None else g.detach().float().cpu() for g in ntp_grads_gpu]
    del outputs, ntp_grads_gpu

    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    states = outputs.hidden_states
    if auxiliary[0] == "stp":
        rows = [states[i, :int(batch["lengths"][i])] for i in range(states.shape[0])]
        aux, _ = released_stp_loss(rows, seed=seed)
    else:
        predictor = auxiliary[1]
        mask = next_residue_transition_mask(batch["input_ids"], batch["attention_mask"])
        # Predictor-only warmup is FP32; explicit casts preserve gradients back
        # through the FP16 checkpoint tensors while avoiding mixed-LN ambiguity.
        current, future = states[:, :-1].float(), states[:, 1:].float()
        next_embedding = model.get_input_embeddings()(batch["input_ids"][:, 1:]).float()
        latent, kl, _ = faithful_losses(
            predictor, current, future, next_embedding, mask,
            model.get_output_embeddings().weight.float(),
        )
        aux = latent + kl
    aux_grads_gpu = torch.autograd.grad(aux, parameters, allow_unused=True)
    groups = {}
    total = {"ntp_sq": 0.0, "aux_sq": 0.0, "dot": 0.0}
    for name, gn, ga in zip(names, ntp_grads, aux_grads_gpu):
        if gn is None or ga is None:
            continue
        ga = ga.detach().float().cpu()
        cell = groups.setdefault(group_name(name), {"ntp_sq": 0.0, "aux_sq": 0.0, "dot": 0.0})
        for destination in (cell, total):
            destination["ntp_sq"] += float(torch.sum(gn * gn))
            destination["aux_sq"] += float(torch.sum(ga * ga))
            destination["dot"] += float(torch.sum(gn * ga))
    def finish(cell):
        nn, an = math.sqrt(cell["ntp_sq"]), math.sqrt(cell["aux_sq"])
        cosine = cell["dot"] / max(nn * an, 1e-30)
        ratio = an / max(nn, 1e-30)
        # Retained NTP-direction magnitude after adding the faithful unit-weight auxiliary.
        retention = 1.0 + cell["dot"] / max(cell["ntp_sq"], 1e-30)
        return {"norm_ratio": ratio, "cosine": cosine, "ntp_direction_retention": retention}
    return float(ntp), float(aux), finish(total), {key: finish(value) for key, value in groups.items()}


def audit(manifest: Path, output: Path, objective: str, predictor_path: Path | None, batches: int, seed: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = load_rita(device=device, freeze=False)
    model, tokenizer = loaded.model, loaded.tokenizer
    # The tied weight is listed twice in named_parameters only once; all unique backbone
    # parameters participate, while the predictor remains outside the audited backbone.
    chosen = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    names, parameters = zip(*chosen)
    auxiliary = ("stp", None)
    if objective == "nextlat":
        payload = torch.load(predictor_path, map_location=device, weights_only=True)
        predictor = FaithfulNextLatPredictor(payload["hidden_size"]).to(device)
        predictor.load_state_dict(payload["state_dict"]); predictor.eval()
        for parameter in predictor.parameters(): parameter.requires_grad_(False)
        auxiliary = ("nextlat", predictor)
    rows = [row for row in read_jsonl(manifest) if row["split"] == "test"]
    rows = sorted(rows, key=lambda row: row["id"])[:batches]
    records = []
    for index, row in enumerate(rows):
        # Fixed 128-residue prefixes bound activation memory and make batches comparable.
        sequence = row["sequence"][:128]
        encoded = [encode_canonical(tokenizer, sequence)]
        batch = {key: value.to(device) for key, value in collate_encoded(encoded).items() if key != "residue_mask"}
        batch["lengths"] = torch.tensor([len(sequence)], device=device)
        ntp, aux, metrics, depth = gradient_pair(model, parameters, names, batch, auxiliary, seed + index)
        records.append({"id": row["id"], "cluster_id": row["cluster_id"], "ntp_loss": ntp,
                        "auxiliary_loss": aux, **metrics, "depth": depth})
        torch.cuda.empty_cache()
    summary = {}
    for metric in ("norm_ratio", "cosine", "ntp_direction_retention"):
        values = [row[metric] for row in records]
        mean, ci = cluster_bootstrap_mean(values, [row["cluster_id"] for row in records], seed=seed)
        summary[metric] = {"mean": mean, "ci95": ci, "sign_flip_p_vs_zero": sign_flip_pvalue(values, seed=seed)}
    ratio = summary["norm_ratio"]["mean"]
    summary["scale_for_3pct_full_backbone_pressure_only"] = .03 / ratio
    depth_summary = {}
    for depth_name in sorted(set().union(*(row["depth"].keys() for row in records))):
        depth_summary[depth_name] = {}
        for metric in ("norm_ratio", "cosine", "ntp_direction_retention"):
            values = [row["depth"][depth_name][metric] for row in records if depth_name in row["depth"]]
            mean, ci = cluster_bootstrap_mean(values, [row["cluster_id"] for row in records], seed=seed)
            depth_summary[depth_name][metric] = {"mean": mean, "ci95": ci,
                "sign_flip_p_vs_zero": sign_flip_pvalue(values, seed=seed)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"objective": objective, "records": records, "summary": summary,
        "depth_summary": depth_summary,
        "parameter_scope": "all unique RITA-M parameters", "backbone_updated": False,
        "nextlat_auxiliary": "SmoothL1 + teacher-to-student KL, unit coefficients" if objective == "nextlat" else None,
        "lora_coefficient_warning": "full-backbone geometry cannot calibrate a rank-32 LoRA subspace",
    }, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("warmup"); p.add_argument("cache_dir", type=Path); p.add_argument("split_manifest", type=Path); p.add_argument("output", type=Path)
    p = sub.add_parser("warmup-full"); p.add_argument("cache_dir", type=Path); p.add_argument("split_manifest", type=Path); p.add_argument("output", type=Path)
    p = sub.add_parser("warmup-dense"); p.add_argument("cache_dir", type=Path); p.add_argument("split_manifest", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--cap-per-protein", type=int, default=64); p.add_argument("--max-epochs", type=int, default=12)
    p = sub.add_parser("audit"); p.add_argument("manifest", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--objective", choices=("stp", "nextlat"), required=True)
    p.add_argument("--predictor", type=Path); p.add_argument("--batches", type=int, default=6)
    args = parser.parse_args()
    if args.stage == "warmup": warmup(args.cache_dir, args.split_manifest, args.output)
    elif args.stage == "warmup-full": warmup_full(args.cache_dir, args.split_manifest, args.output)
    elif args.stage == "warmup-dense": warmup_full(args.cache_dir, args.split_manifest, args.output,
        cap_per_protein=args.cap_per_protein, max_epochs=args.max_epochs)
    else:
        if args.objective == "nextlat" and args.predictor is None: raise SystemExit("--predictor required")
        audit(args.manifest, args.output, args.objective, args.predictor, args.batches, 20260914)


if __name__ == "__main__": main()
