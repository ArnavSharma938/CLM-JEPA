#!/usr/bin/env python
"""Gradient-only faithful NextLat audit in RITA-M's rank-32 LoRA subspace."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.modeling import load_rita
from src.nextlat import FaithfulNextLatPredictor, faithful_losses
from src.stats import cluster_bootstrap_mean
from src.tokenization import collate_encoded, encode_canonical, next_residue_transition_mask

TARGET_SUFFIXES = ("query", "key", "value", "proj")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def select_test_proteins(rows, count):
    eligible = sorted((row for row in rows if row["split"] == "test" and len(row["sequence"]) >= 128),
                      key=lambda row: (len(row["sequence"]), row["id"]))
    unique = []; seen = set()
    for row in eligible:
        if row["cluster_id"] not in seen:
            unique.append(row); seen.add(row["cluster_id"])
    indices = np.linspace(0, len(unique) - 1, count, dtype=int)
    selected = [unique[index] for index in indices]
    if len(selected) != count or len({row["cluster_id"] for row in selected}) != count:
        raise AssertionError("could not select the requested number of distinct test clusters")
    return selected


def depth_name(parameter_name):
    marker = "transformer.layers."
    if marker not in parameter_name:
        raise AssertionError(f"unexpected LoRA parameter outside RITA layers: {parameter_name}")
    index = int(parameter_name.split(marker, 1)[1].split(".", 1)[0])
    return "early_0_7" if index < 8 else "middle_8_15" if index < 16 else "late_16_23"


def masked_ntp(outputs, ids, attention):
    mask = next_residue_transition_mask(ids, attention)
    return F.cross_entropy(outputs.logits[:, :-1][mask].float(), ids[:, 1:][mask])


def summarize_pair(names, ntp_grads, auxiliary_grads):
    accumulators = {"global": {"ntp_sq": 0.0, "aux_sq": 0.0, "dot": 0.0}}
    for name, ntp, auxiliary in zip(names, ntp_grads, auxiliary_grads):
        if ntp is None or auxiliary is None:
            continue
        ntp = ntp.detach().float().cpu(); auxiliary = auxiliary.detach().float().cpu()
        for key in ("global", depth_name(name)):
            cell = accumulators.setdefault(key, {"ntp_sq": 0.0, "aux_sq": 0.0, "dot": 0.0})
            cell["ntp_sq"] += float(torch.sum(ntp * ntp))
            cell["aux_sq"] += float(torch.sum(auxiliary * auxiliary))
            cell["dot"] += float(torch.sum(ntp * auxiliary))
    result = {}
    for key, cell in accumulators.items():
        ntp_norm, auxiliary_norm = math.sqrt(cell["ntp_sq"]), math.sqrt(cell["aux_sq"])
        result[key] = {
            "norm_ratio": auxiliary_norm / max(ntp_norm, 1e-30),
            "cosine": cell["dot"] / max(ntp_norm * auxiliary_norm, 1e-30),
            "ntp_direction_retention": 1.0 + cell["dot"] / max(cell["ntp_sq"], 1e-30),
        }
    return result


def bootstrap_summary(records, path, seed):
    summary = {}
    for metric in ("norm_ratio", "cosine", "ntp_direction_retention"):
        values = [row[path][metric] for row in records]
        mean, ci = cluster_bootstrap_mean(values, [row["cluster_id"] for row in records], seed=seed)
        summary[metric] = {"mean": mean, "ci95": ci}
    negative = [float(row[path]["cosine"] < 0) for row in records]
    mean, ci = cluster_bootstrap_mean(negative, [row["cluster_id"] for row in records], seed=seed)
    summary["fraction_negative_cosine"] = {"mean": mean, "ci95": ci,
                                            "count": int(sum(negative)), "denominator": len(negative)}
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path); parser.add_argument("predictor", type=Path)
    parser.add_argument("output", type=Path); parser.add_argument("--proteins", type=int, default=32)
    args = parser.parse_args(); seed = 20260914
    torch.manual_seed(seed); device = "cuda" if torch.cuda.is_available() else "cpu"

    loaded = load_rita(device=device, freeze=True)
    base = loaded.model
    discovered = [name for name, module in base.named_modules()
                  if isinstance(module, torch.nn.Linear) and name.endswith(TARGET_SUFFIXES)]
    expected = [f"transformer.layers.{layer}.self_attention.{suffix}"
                for layer in range(24) for suffix in TARGET_SUFFIXES]
    if sorted(discovered) != sorted(expected):
        raise AssertionError({"discovered": discovered, "expected": expected})
    config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=32,
                        lora_dropout=0.0, bias="none", target_modules=list(TARGET_SUFFIXES),
                        init_lora_weights=True)
    model = get_peft_model(base, config).to(device).eval()
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise AssertionError("only LoRA parameters may require gradients")
    lora_a = [parameter for name, parameter in trainable if ".lora_A." in name]
    lora_b = [parameter for name, parameter in trainable if ".lora_B." in name]
    standard_noop_verified = (len(lora_a) == 96 and len(lora_b) == 96
                              and all(torch.count_nonzero(parameter).item() == 0 for parameter in lora_b)
                              and all(torch.count_nonzero(parameter).item() > 0 for parameter in lora_a))
    if not standard_noop_verified:
        raise AssertionError("expected standard nonzero-A/zero-B LoRA initialization")
    matched_modules = sorted({name.split(".lora_", 1)[0] for name, _ in trainable})
    if len(matched_modules) != 96 or any(".self_attention." not in name for name in matched_modules):
        raise AssertionError(f"unexpected LoRA targets: {matched_modules}")

    payload = torch.load(args.predictor, map_location=device, weights_only=True)
    predictor = FaithfulNextLatPredictor(payload["hidden_size"]).to(device).eval()
    predictor.load_state_dict(payload["state_dict"])
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in predictor.parameters()):
        raise AssertionError("predictor must remain frozen")

    names, parameters = zip(*trainable)
    all_versions = {name: parameter._version for name, parameter in model.named_parameters()}
    predictor_versions = [parameter._version for parameter in predictor.parameters()]
    selected = select_test_proteins(read_jsonl(args.manifest), args.proteins)
    records = []
    for row in selected:
        sequence = row["sequence"][:128]
        encoded = [encode_canonical(loaded.tokenizer, sequence)]
        batch = {key: value.to(device) for key, value in collate_encoded(encoded).items()
                 if key != "residue_mask"}

        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        ntp = masked_ntp(outputs, batch["input_ids"], batch["attention_mask"])
        ntp_grads = torch.autograd.grad(ntp, parameters, allow_unused=True)
        del outputs

        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        states = outputs.hidden_states
        mask = next_residue_transition_mask(batch["input_ids"], batch["attention_mask"])
        current, future = states[:, :-1].float(), states[:, 1:].float()
        next_embedding = model.get_input_embeddings()(batch["input_ids"][:, 1:]).float()
        latent, kl, _ = faithful_losses(predictor, current, future, next_embedding, mask,
                                        model.get_output_embeddings().weight.float())
        auxiliary = latent + kl
        auxiliary_grads = torch.autograd.grad(auxiliary, parameters, allow_unused=True)
        metrics = summarize_pair(names, ntp_grads, auxiliary_grads)
        records.append({"id": row["id"], "cluster_id": row["cluster_id"], "prefix_length": 128,
                        "ntp_loss": float(ntp), "latent_loss": float(latent), "kl_loss": float(kl),
                        "auxiliary_loss": float(auxiliary), "global": metrics["global"],
                        "depth": {key: value for key, value in metrics.items() if key != "global"}})
        del outputs, ntp_grads, auxiliary_grads
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if all_versions != {name: parameter._version for name, parameter in model.named_parameters()}:
        raise AssertionError("a RITA or LoRA parameter changed during the gradient-only audit")
    if predictor_versions != [parameter._version for parameter in predictor.parameters()]:
        raise AssertionError("a predictor parameter changed during the gradient-only audit")

    summary = bootstrap_summary(records, "global", seed)
    depth_summary = {depth: bootstrap_summary(
        [{**row, "selected_depth": row["depth"][depth]} for row in records], "selected_depth", seed)
        for depth in ("early_0_7", "middle_8_15", "late_16_23")}
    result = {
        "checkpoint": "lightonai/RITA_m@d819157a3a96d278500232b2cbb2a02d9646bcf6",
        "predictor_checkpoint": str(args.predictor).replace("\\", "/"),
        "predictor_objective": "unit SmoothL1 + unit teacher-to-student KL",
        "lora": {"rank": 32, "alpha": 32, "scaling": 1.0, "dropout": 0.0, "bias": "none",
                 "target_suffixes": list(TARGET_SUFFIXES), "matched_module_count": len(matched_modules),
                 "matched_modules": matched_modules, "trainable_parameter_count": sum(p.numel() for _, p in trainable),
                 "trainable_tensor_count": len(trainable),
                 "standard_noop_initialization_verified": standard_noop_verified,
                 "initialization_tangent_note": "At zero-B initialization, A gradients are zero and B gradients define the initial trainable tangent."},
        "sample": {"proteins": len(records), "distinct_homology_clusters": len({r['cluster_id'] for r in records}),
                   "selection": "deterministic length-stratified test proteins with length >=128",
                   "fixed_prefix_length": 128},
        "summary": summary, "depth_summary": depth_summary, "records": records,
        "comparison_references": {
            "RITA_full_backbone": {"norm_ratio": 0.3649112336570964, "cosine": 0.16456895076544195},
            "ChemFM_trained_LoRA": {"norm_ratio_range": [2.8, 4.6], "description": "substantial middle/late conflict"}},
        "only_lora_requires_grad": True, "predictor_frozen": True,
        "parameters_updated": False, "optimizer_constructed": False,
        "parameter_versions_unchanged": True, "seed": seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
