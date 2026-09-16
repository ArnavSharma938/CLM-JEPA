#!/usr/bin/env python
"""Evaluate one frozen causal-pilot adapter without modifying parameters."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.modeling import load_rita
from src.pilot import load_protocol, sha256_file
from src.pilot_eval import (
    bidirectional_summary, contact_probe, directional_lm_rows, ec_probe,
    cached_generate_rita, generated_quality, reference_composition, secondary_probe,
    sequence_representations,
)
from src.tokenization import encode_canonical


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_model(adapter: Path, device: str):
    loaded = load_rita(device=device, dtype=torch.bfloat16, freeze=True)
    model = PeftModel.from_pretrained(
        loaded.model, adapter, is_trainable=False, autocast_adapter_dtype=False
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return loaded.tokenizer, model


def mean(rows, key):
    return float(np.mean([row[key] for row in rows]))


def length_bin(length: int) -> str:
    return "64_159" if length < 160 else "160_319" if length < 320 else "320_512"


def primary(args, tokenizer, model, protocol):
    heldout = read_jsonl(args.data / "heldout_lm.jsonl")
    directional = directional_lm_rows(model, tokenizer, heldout, args.device, args.batch_size)
    paired = bidirectional_summary(directional)
    lm = {
        "proteins": len(paired),
        "forward_ntp_loss": mean(paired, "forward_ntp_loss"),
        "reverse_ntp_loss": mean(paired, "reverse_ntp_loss"),
        "bidirectional_ntp_loss": mean(paired, "bidirectional_ntp_loss"),
        "correct_probability": mean(paired, "correct_probability"),
        "top1": mean(paired, "top1"),
        "length_bins": {
            name: {
                "n": len(selected),
                "bidirectional_ntp_loss": mean(selected, "bidirectional_ntp_loss"),
            }
            for name in ("64_159", "160_319", "320_512")
            for selected in [[row for row in paired if length_bin(row["length"]) == name]]
        },
        "records": paired,
    }
    panel = json.loads((args.data / "proteingym_panel.json").read_text())
    assays = []
    for assay in panel["assays"]:
        table = pd.read_csv(args.data / "proteingym" / assay["DMS_filename"])
        rows = [
            {"id": str(mutant), "sequence": str(sequence), "cluster_id": assay["DMS_id"]}
            for mutant, sequence in zip(table.mutant, table.mutated_sequence)
        ]
        scored = bidirectional_summary(
            directional_lm_rows(model, tokenizer, rows, args.device, args.batch_size)
        )
        by_id = {row["id"]: -(row["forward_ntp_loss"] + row["reverse_ntp_loss"])
                 for row in scored}
        predictions = np.asarray([by_id[str(mutant)] for mutant in table.mutant])
        phenotype = table.DMS_score.to_numpy(float)
        rho = float(spearmanr(predictions, phenotype).statistic)
        assays.append({
            "DMS_id": assay["DMS_id"],
            "category": assay["coarse_selection_type"],
            "n": len(table),
            "spearman": rho,
        })
    proteingym = {
        "assays": assays,
        "macro_spearman": float(np.mean([row["spearman"] for row in assays])),
        "categories": {
            category: float(np.mean([row["spearman"] for row in assays
                                     if row["category"] == category]))
            for category in sorted({row["category"] for row in assays})
        },
        "fitness_convention": "negative forward mean CE plus negative reverse mean CE",
    }
    return {"lm": lm, "proteingym": proteingym}


def representations(args, tokenizer, model):
    ec_rows = read_jsonl(args.data / "swissprot_ec.jsonl")
    ec_forward = sequence_representations(model, tokenizer, ec_rows, args.device,
                                          args.batch_size, reverse=False)
    ec_reverse = sequence_representations(model, tokenizer, ec_rows, args.device,
                                          args.batch_size, reverse=True)
    ec = ec_probe(ec_rows, ec_forward, ec_reverse)
    del ec_forward, ec_reverse

    ss_rows = read_jsonl(args.data / "tape_secondary.jsonl")
    ss_features = sequence_representations(model, tokenizer, ss_rows, args.device,
                                           args.batch_size, reverse=False)
    ss = secondary_probe(ss_rows, ss_features)
    del ss_features

    contact_rows = read_jsonl(args.data / "proteinnet_contact.jsonl")
    contact_features = sequence_representations(
        model, tokenizer, contact_rows, args.device, args.batch_size, reverse=False
    )
    contact = contact_probe(contact_rows, contact_features)
    return {"swissprot_ec": ec, "secondary_structure": ss, "long_range_contact": contact}


@torch.inference_mode()
def generate(args, tokenizer, model, protocol):
    prompts = read_jsonl(args.data / "generation_prompts.jsonl")
    heldout = read_jsonl(args.data / "heldout_lm.jsonl")
    composition = reference_composition(heldout)
    config = protocol["evaluation"]
    records = []
    prefixes = [list(encode_canonical(tokenizer, row["prompt"]).input_ids[:-1])
                for row in prompts]
    if len({len(prefix) for prefix in prefixes}) != 1:
        raise ValueError("locked generation prompts must have equal lengths")
    ids = torch.tensor(prefixes, device=args.device)
    uniforms = torch.tensor(np.stack([
        np.random.default_rng(9_000_000 + index).random(
            config["generation_max_new_tokens"]
        ) for index in range(len(prompts))
    ]), device=args.device, dtype=torch.float32)
    all_sampled = cached_generate_rita(
        model, ids, uniforms=uniforms, do_sample=True,
        temperature=config["generation_temperature"],
        top_p=config["generation_top_p"],
        max_new_tokens=config["generation_max_new_tokens"], eos_token_id=2,
    ).tolist()
    for row, prefix, sampled in zip(prompts, prefixes, all_sampled):
        continuation = sampled[len(prefix):]
        terminated = 2 in continuation
        if terminated:
            continuation = continuation[:continuation.index(2)]
        tokens = tokenizer.convert_ids_to_tokens(prefix + continuation)
        sequence = "".join(str(token) for token in tokens)
        quality = generated_quality(sequence, composition)
        records.append({
            "id": row["id"], "prompt": row["prompt"], "sequence": sequence,
            "terminated": terminated, **quality,
        })
    # Batched official-direction fitness is evaluated only where token decoding
    # produced a nonempty canonical sequence.
    valid = [row for row in records if row["canonical_fraction"] == 1.0 and row["length"] > 0]
    scored = bidirectional_summary(directional_lm_rows(
        model, tokenizer,
        [{"id": row["id"], "sequence": row["sequence"]} for row in valid],
        args.device, args.batch_size,
    ))
    fitness = {row["id"]: -(row["forward_ntp_loss"] + row["reverse_ntp_loss"])
               for row in scored}
    for row in records:
        row["rita_bidirectional_fitness"] = fitness.get(row["id"])
    keys = (
        "length", "canonical_fraction", "entropy", "max_homopolymer",
        "unique_bigram_fraction", "composition_l1", "rita_bidirectional_fitness",
    )
    return {
        "records": records,
        "summary": {
            "n": len(records),
            "termination_rate": float(np.mean([row["terminated"] for row in records])),
            **{
                key: float(np.mean([row[key] for row in records if row[key] is not None]))
                for key in keys
            },
        },
        "settings": {
            "prompt_length": config["generation_prompt_residues"],
            "max_new_tokens": config["generation_max_new_tokens"],
            "temperature": config["generation_temperature"],
            "top_p": config["generation_top_p"],
            "per_sample_rng_seed": "9000000 + locked prompt index",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("primary", "representations", "generation", "all"))
    parser.add_argument("adapter", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "data/causal_pilot")
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/causal_pilot.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    tokenizer, model = load_model(args.adapter, args.device)
    result = {
        "stage": args.stage,
        "adapter": str(args.adapter),
        "adapter_config_sha256": sha256_file(args.adapter / "adapter_config.json"),
        "adapter_weights_sha256": sha256_file(args.adapter / "adapter_model.safetensors"),
        "protocol_sha256": sha256_file(args.protocol),
        "parameters_updated": False,
    }
    if args.stage in {"primary", "all"}:
        result.update(primary(args, tokenizer, model, protocol))
    if args.stage in {"representations", "all"}:
        result["representations"] = representations(args, tokenizer, model)
    if args.stage in {"generation", "all"}:
        result["generation"] = generate(args, tokenizer, model, protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
