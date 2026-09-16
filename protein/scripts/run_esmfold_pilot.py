#!/usr/bin/env python
"""One-load ESMFold check for the locked 16 generated sequences per model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, EsmForProteinFolding

AA = set("ACDEFGHIKLMNPQRSTVWY")


def rank(identifier: str) -> str:
    return hashlib.sha256(f"rita-pilot-esmfold-v1\0{identifier}".encode()).hexdigest()


def structure_metrics(output, length: int):
    # ESMFold reports atom37 pLDDT; CA is atom index 1. Averaging padded/nonexistent
    # atom slots would bias the sequence-confidence statistic.
    plddt = output.plddt[0, :length, 1].float().cpu().numpy()
    positions = output.positions[-1, 0, :length, 1].float().cpu().numpy()
    i, j = np.triu_indices(length, k=3)
    distance = np.linalg.norm(positions[i] - positions[j], axis=1)
    return {
        "mean_plddt": float(plddt.mean()),
        # Transformers' EsmForProteinFolding exposes categorical lDDT on [0, 1].
        # A conventional pLDDT threshold of 70 therefore corresponds to 0.70.
        "fraction_plddt_ge_70": float(np.mean(plddt >= 0.70)),
        "nonlocal_ca_clash_fraction": float(np.mean(distance < 3.0)) if len(distance) else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("generation_json", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1", low_cpu_mem_usage=True
    ).to(args.device).eval()
    model.esm = model.esm.half()
    model.trunk.set_chunk_size(128)
    results = {}
    for path in args.generation_json:
        payload = json.loads(path.read_text())
        eligible = [
            row for row in payload["generation"]["records"]
            if row["sequence"] and set(row["sequence"]) <= AA
        ]
        selected = sorted(eligible, key=lambda row: rank(row["id"]))[:16]
        records = []
        for row in selected:
            ids = tokenizer(
                [row["sequence"]], return_tensors="pt", add_special_tokens=False
            )["input_ids"].to(args.device)
            with torch.inference_mode():
                output = model(ids)
            records.append({
                "id": row["id"], "sequence_sha256": hashlib.sha256(
                    row["sequence"].encode()
                ).hexdigest(), "length": len(row["sequence"]),
                **structure_metrics(output, len(row["sequence"])),
            })
        results[str(path)] = {
            "eligible": len(eligible), "selected": len(records), "records": records,
            "summary": {
                key: float(np.mean([row[key] for row in records]))
                for key in ("mean_plddt", "fraction_plddt_ge_70",
                            "nonlocal_ca_clash_fraction")
            } if records else {},
        }
    result = {
        "model": "facebook/esmfold_v1",
        "selection": "lowest SHA256(rita-pilot-esmfold-v1, prompt id)",
        "samples_per_model": 16,
        "results": results,
        "parameters_updated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
