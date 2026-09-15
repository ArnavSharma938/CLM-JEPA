#!/usr/bin/env python
"""Small preregistered native-generation surrogate-selectivity replay."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.constants import CANONICAL_AA, RITA_EOS_ID
from src.geometry import adjacent_tangent_correlation, local_curvature, path_efficiency, tube_radius
from src.modeling import load_rita
from src.nextlat import FaithfulNextLatPredictor
from src.stats import cluster_bootstrap_mean, sign_flip_pvalue
from src.tokenization import collate_encoded, encode_canonical


def top_p_sample(logits, allowed, generator, top_p=.9):
    masked = torch.full_like(logits, -torch.inf); masked[:, allowed] = logits[:, allowed]
    sorted_logits, indices = torch.sort(masked, descending=True)
    probabilities = torch.softmax(sorted_logits, -1); cumulative = probabilities.cumsum(-1)
    remove = cumulative - probabilities > top_p; sorted_logits[remove] = -torch.inf
    sampled = torch.multinomial(torch.softmax(sorted_logits, -1), 1, generator=generator)
    return indices.gather(1, sampled).squeeze(1)


@torch.inference_mode()
def generate(model, tokenizer, count, batch_size, seed, minimum=64, maximum=128):
    token_ids = tokenizer.convert_tokens_to_ids(list(CANONICAL_AA))
    generator = torch.Generator(device="cuda").manual_seed(seed); sequences = []
    for batch_start in range(0, count, batch_size):
        size = min(batch_size, count - batch_start)
        ids = torch.full((size, 1), tokenizer.convert_tokens_to_ids("M"), dtype=torch.long, device="cuda")
        finished = torch.zeros(size, dtype=torch.bool, device="cuda")
        allowed = list(token_ids)
        for position in range(1, maximum):
            outputs = model(input_ids=ids, attention_mask=torch.ones_like(ids))
            choices = allowed + ([RITA_EOS_ID] if position >= minimum else [])
            next_id = top_p_sample(outputs.logits[:, -1].float(), choices, generator)
            finished |= next_id.eq(RITA_EOS_ID)
            # Keep completed rows shape-compatible; their tail is discarded.
            next_id = torch.where(finished, torch.full_like(next_id, RITA_EOS_ID), next_id)
            ids = torch.cat((ids, next_id[:, None]), 1)
            if bool(finished.all()): break
        for row in ids.cpu().tolist():
            row = row[:row.index(RITA_EOS_ID)] if RITA_EOS_ID in row else row
            sequences.append("".join(tokenizer.convert_ids_to_tokens(row)))
    return sequences


def preregistered_quality(sequence, reference_composition):
    counts = Counter(sequence); probabilities = np.asarray([counts[aa] / len(sequence) for aa in CANONICAL_AA])
    entropy = float(-(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0])).sum())
    longest, run = 1, 1
    for left, right in zip(sequence, sequence[1:]):
        run = run + 1 if left == right else 1; longest = max(longest, run)
    trimer_unique = len({sequence[i:i + 3] for i in range(len(sequence) - 2)}) / max(1, len(sequence) - 2)
    composition_l1 = float(np.abs(probabilities - reference_composition).sum())
    return {"entropy": entropy, "longest_homopolymer": longest,
            "unique_trimer_fraction": trimer_unique, "composition_l1": composition_l1}


@torch.inference_mode()
def surrogate_values(model, tokenizer, predictor, sequences, batch_size):
    result = []
    for start in range(0, len(sequences), batch_size):
        chosen = sequences[start:start + batch_size]
        encoded = [encode_canonical(tokenizer, sequence) for sequence in chosen]
        batch = {key: value.cuda() for key, value in collate_encoded(encoded).items() if key != "residue_mask"}
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        for row, sequence in enumerate(chosen):
            states = outputs.hidden_states[row, :len(sequence)].float()
            ids = batch["input_ids"][row]
            prediction = predictor(states[:-1], model.get_input_embeddings()(ids[1:len(sequence)]).float())
            result.append({"nextlat_error": float(F.smooth_l1_loss(prediction, states[1:])),
                "curvature": float(local_curvature(states).mean()),
                "tangent_c1": float(adjacent_tangent_correlation(states).mean()),
                "tube_radius": float(tube_radius(states).mean()),
                "path_efficiency": float(path_efficiency(states))})
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("reference_manifest", type=Path)
    parser.add_argument("predictor", type=Path); parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=100); parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args(); seed = 20260914
    reference = [json.loads(line)["sequence"] for line in args.reference_manifest.read_text().splitlines()]
    reference_composition = np.asarray([sum(sequence.count(aa) for sequence in reference) /
        sum(map(len, reference)) for aa in CANONICAL_AA])
    loaded = load_rita(); model, tokenizer = loaded.model, loaded.tokenizer
    payload = torch.load(args.predictor, map_location="cuda", weights_only=True)
    predictor = FaithfulNextLatPredictor(payload["hidden_size"]).cuda().eval(); predictor.load_state_dict(payload["state_dict"])
    sequences = generate(model, tokenizer, args.count, args.batch_size, seed)
    quality = [preregistered_quality(sequence, reference_composition) for sequence in sequences]
    # Equal-weight, direction-aligned z-score fixed before viewing surrogate values.
    fields = (("entropy", 1), ("longest_homopolymer", -1), ("unique_trimer_fraction", 1), ("composition_l1", -1))
    matrix = np.asarray([[row[name] * direction for name, direction in fields] for row in quality])
    composite = ((matrix - matrix.mean(0)) / np.maximum(matrix.std(0), 1e-8)).mean(1)
    surrogate = surrogate_values(model, tokenizer, predictor, sequences, args.batch_size)
    median = float(np.median(composite)); labels = np.where(composite >= median, "better", "worse")
    records = [{"id": f"generation_{i:03d}", "sequence": sequence, "quality": quality[i],
        "quality_composite": float(composite[i]), "quality_group": str(labels[i]), **surrogate[i]}
        for i, sequence in enumerate(sequences)]
    comparisons = {}
    for metric in surrogate[0]:
        better = np.asarray([row[metric] for row in records if row["quality_group"] == "better"])
        worse = np.asarray([row[metric] for row in records if row["quality_group"] == "worse"])
        # Independent generated sequences; bootstrap each group and permutation-test labels.
        rng = np.random.default_rng(seed); boot = np.asarray([rng.choice(better, len(better)).mean() -
            rng.choice(worse, len(worse)).mean() for _ in range(5000)])
        observed = float(better.mean() - worse.mean()); pooled = np.r_[better, worse]
        null = []
        for _ in range(10000):
            shuffled = rng.permutation(pooled); null.append(shuffled[:len(better)].mean() - shuffled[len(better):].mean())
        comparisons[metric] = {"better_minus_worse": observed,
            "ci95": [float(x) for x in np.quantile(boot, [.025, .975])],
            "permutation_p": float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / 10001)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"configuration": {"prompt": "M", "temperature": 1.0, "top_p": .9,
        "minimum_residues": 64, "maximum_residues": 128, "canonical_tokens_plus_EOS_only": True,
        "seed": seed, "quality_rule_fixed_before_surrogates": True}, "records": records,
        "comparisons": comparisons, "backbone_updated": False}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
