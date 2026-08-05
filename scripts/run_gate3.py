from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from rdkit import Chem
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clm_jepa.assay import relationship_metrics  # noqa: E402
from clm_jepa.chemfm_native import ReactionCollator, canonicalize, load_lora_model, load_reaction_tokenizer  # noqa: E402
from clm_jepa.modeling import add_predictor_tokens, extract_source_and_target  # noqa: E402
from clm_jepa.paths import MODEL_DIR, TOKENIZER_DIR  # noqa: E402


@torch.inference_mode()
def encode_final(model, rows: list[torch.Tensor], pad_token_id: int, batch_size: int = 8) -> torch.Tensor:
    results = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        padded = pad_sequence(chunk, batch_first=True, padding_value=pad_token_id).to(model.device)
        attention = padded.ne(pad_token_id)
        hidden = model(input_ids=padded, attention_mask=attention, output_hidden_states=True).hidden_states[-1]
        indices = attention.sum(dim=1) - 1
        states = hidden[torch.arange(len(chunk), device=model.device), indices]
        results.append(states.float().cpu())
    return torch.cat(results)


def main() -> None:
    torch.manual_seed(533)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    predictor_ids = add_predictor_tokens(tokenizer)
    model = load_lora_model(MODEL_DIR, tokenizer).cuda().eval()
    with (ROOT / "data" / "manifests" / "gate1" / "train_128.csv").open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    batch = ReactionCollator(tokenizer)(records)
    tensor_batch = {key: value for key, value in batch.items() if torch.is_tensor(value)}
    sources, targets = extract_source_and_target(tensor_batch)
    identities = [canonicalize(record["tgt"]) for record in records]
    target_states = encode_final(model, targets, tokenizer.pad_token_id)
    target_lengths_by_identity: dict[str, list[int]] = defaultdict(list)
    for identity, target in zip(identities, targets):
        target_lengths_by_identity[identity].append(len(target))
    token_lengths = {
        identity: round(sum(lengths) / len(lengths))
        for identity, lengths in target_lengths_by_identity.items()
    }
    heavy_atoms = {
        identity: Chem.MolFromSmiles(identity).GetNumHeavyAtoms() for identity in sorted(set(identities))
    }

    source_states = {}
    raw = {}
    for k in range(5):
        suffix = list(reversed(predictor_ids[:k]))
        rows = [torch.cat((source, source.new_tensor(suffix))) if suffix else source for source in sources]
        source_states[k] = encode_final(model, rows, tokenizer.pad_token_id)
        raw[str(k)] = relationship_metrics(
            source_states[k], target_states, identities, token_lengths, heavy_atoms, centered=False
        )
        print(json.dumps({"k": k, **raw[str(k)]}), flush=True)

    retained = [int(k) for k, value in raw.items() if value["retains_pair_signal"]]
    rescue = None
    if not retained:
        rescue = {}
        for k in range(5):
            rescue[str(k)] = relationship_metrics(
                source_states[k], target_states, identities, token_lengths, heavy_atoms, centered=True
            )
        retained = [int(k) for k, value in rescue.items() if value["retains_pair_signal"]]

    output = {
        "checkpoint": "ChemFM/ChemFM-1B f99dc2e89726539bb9cf31b2e2b4360650bac6a8",
        "manifest": "data/manifests/gate1/train_128.csv",
        "examples": len(records),
        "unique_target_identities": len(set(identities)),
        "selection_rule": "positive correct-vs-random and correct-vs-matched margins, matched retrieval top-1 above candidate chance, target effective rank > 1.5",
        "raw": raw,
        "rescue": rescue,
        "retained_k": retained,
        "stop": not bool(retained),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }
    path = ROOT / "artifacts" / "gate3" / "relationship_assay.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"retained_k": retained, "stop": output["stop"]}))


if __name__ == "__main__":
    main()
