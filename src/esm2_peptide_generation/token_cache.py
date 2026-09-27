from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .config import MODEL_ID, MODEL_REVISION
from .prepare_data import sha256_file


def build_token_cache(pair_manifest: Path, output_path: Path, tokenizer) -> dict:
    source_hash = sha256_file(pair_manifest)
    rows = [json.loads(line) for line in pair_manifest.read_text(encoding="utf-8").splitlines() if line]
    receptor_sequences = {row["receptor_id"]: row["receptor_sequence"] for row in rows}
    peptide_sequences = {
        hashlib.sha256(row["peptide_sequence"].encode("ascii")).hexdigest(): row["peptide_sequence"]
        for row in rows
    }
    receptor_tokens = {
        key: tokenizer.encode(sequence, add_special_tokens=False)
        for key, sequence in receptor_sequences.items()
    }
    peptide_tokens = {
        key: tokenizer.encode(sequence, add_special_tokens=False)
        for key, sequence in peptide_sequences.items()
    }
    for identifier, sequence in receptor_sequences.items():
        if len(receptor_tokens[identifier]) != len(sequence):
            raise RuntimeError(f"receptor tokenization changed sequence length: {identifier}")
    for identifier, sequence in peptide_sequences.items():
        if len(peptide_tokens[identifier]) != len(sequence):
            raise RuntimeError(f"peptide tokenization changed sequence length: {identifier}")
    pair_records = {
        row["pair_id"]: {
            **row,
            "peptide_id": hashlib.sha256(row["peptide_sequence"].encode("ascii")).hexdigest(),
        }
        for row in rows
    }
    payload = {
        "format_version": 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "source_manifest_sha256": source_hash,
        "receptor_tokens": receptor_tokens,
        "peptide_tokens": peptide_tokens,
        "pairs": pair_records,
        "special_tokens": {
            # ESM calls its leading sequence token <cls>, not <bos>.
            "bos": tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id,
            "eos": tokenizer.eos_token_id,
            "mask": tokenizer.mask_token_id,
            "pad": tokenizer.pad_token_id,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output_path)
    return {
        "cache_path": str(output_path),
        "source_manifest_sha256": source_hash,
        "cache_sha256": sha256_file(output_path),
        "pairs": len(pair_records),
        "unique_receptors": len(receptor_tokens),
        "unique_peptides": len(peptide_tokens),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def load_token_cache(path: Path, expected_source_hash: str | None = None) -> dict:
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if cache["model_id"] != MODEL_ID or cache["model_revision"] != MODEL_REVISION:
        raise RuntimeError("token cache does not match the pinned ESM-2 revision")
    if expected_source_hash and cache["source_manifest_sha256"] != expected_source_hash:
        raise RuntimeError("token cache source-manifest hash mismatch")
    return cache
