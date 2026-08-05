from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clm_jepa.assay import relationship_metrics  # noqa: E402
from clm_jepa.chemfm_native import ReactionCollator, load_lora_model, load_reaction_tokenizer  # noqa: E402
from clm_jepa.modeling import add_predictor_tokens, extract_source_and_target  # noqa: E402
from clm_jepa.paths import MODEL_DIR, TOKENIZER_DIR  # noqa: E402


@torch.inference_mode()
def encode_position(
    model, rows: list[torch.Tensor], pad_token_id: int, *, offset: int = 1, batch_size: int = 16
) -> torch.Tensor:
    results = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        padded = pad_sequence(chunk, batch_first=True, padding_value=pad_token_id).to(model.device)
        attention = padded.ne(pad_token_id)
        hidden = model(input_ids=padded, attention_mask=attention, output_hidden_states=True).hidden_states[-1]
        indices = attention.sum(dim=1) - offset
        if (indices < 0).any():
            raise ValueError(f"active sequence is shorter than requested offset {offset}")
        states = hidden[torch.arange(len(chunk), device=model.device), indices]
        results.append(states.float())
    return torch.cat(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "gate1" / "train_128.csv")
    parser.add_argument("--dataset", default="uspto_mit_synthesis")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    torch.manual_seed(533)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    predictor_ids = add_predictor_tokens(tokenizer)
    model = load_lora_model(MODEL_DIR, tokenizer).cuda().eval()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    batch = ReactionCollator(tokenizer)(records)
    tensor_batch = {key: value for key, value in batch.items() if torch.is_tensor(value)}
    sources, targets = extract_source_and_target(tensor_batch)
    identities = [".".join(sorted(record["tgt"].split("."))) for record in records]
    target_states = encode_position(model, targets, tokenizer.pad_token_id, batch_size=args.batch_size)
    target_lengths_by_identity: dict[str, list[int]] = defaultdict(list)
    for identity, target in zip(identities, targets):
        target_lengths_by_identity[identity].append(len(target))
    token_lengths = {
        identity: round(sum(lengths) / len(lengths))
        for identity, lengths in target_lengths_by_identity.items()
    }
    atom_pattern = re.compile(r"Cl|Br|Si|Se|Na|Li|Mg|Al|Ca|[A-Z]|[cnopsb]")
    heavy_atoms = {identity: len(atom_pattern.findall(identity)) for identity in sorted(set(identities))}

    source_states = {}
    raw = {}
    for k in (-1, 0, 1, 2, 3):
        suffix = [] if k == -1 else list(reversed(predictor_ids[:k]))
        rows = [torch.cat((source, source.new_tensor(suffix))) if suffix else source for source in sources]
        source_states[k] = encode_position(
            model, rows, tokenizer.pad_token_id, offset=2 if k == -1 else 1, batch_size=args.batch_size
        )
        raw[str(k)] = relationship_metrics(
            source_states[k], target_states, identities, token_lengths, heavy_atoms, centered=False
        )
        print(json.dumps({"k": k, **raw[str(k)]}), flush=True)

    retained = [int(k) for k, value in raw.items() if value["retains_pair_signal"]]
    rescue = None
    if not retained:
        rescue = {}
        for k in (-1, 0, 1, 2, 3):
            rescue[str(k)] = relationship_metrics(
                source_states[k], target_states, identities, token_lengths, heavy_atoms, centered=True
            )
        retained = [int(k) for k, value in rescue.items() if value["retains_pair_signal"]]

    output = {
        "checkpoint": "ChemFM/ChemFM-1B f99dc2e89726539bb9cf31b2e2b4360650bac6a8",
        "dataset": args.dataset,
        "manifest": str(args.manifest.resolve().relative_to(ROOT)),
        "examples": len(records),
        "unique_target_identities": len(set(identities)),
        "selection_rule": "positive correct-vs-random and correct-vs-matched margins, matched retrieval top-1 above candidate chance, target effective rank > 1.5",
        "identity_policy": "released target string; disconnected components sorted",
        "size_matching": "SMILES atom-token count (RDKit-free and identical across datasets)",
        "raw": raw,
        "rescue": rescue,
        "retained_k": retained,
        "stop": not bool(retained),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }
    path = args.output or ROOT / "artifacts" / "gate3" / f"relationship_assay_{args.dataset}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"retained_k": retained, "stop": output["stop"]}))


if __name__ == "__main__":
    main()
