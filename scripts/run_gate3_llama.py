from __future__ import annotations

import csv
import hashlib
import heapq
import json
import sys
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clm_jepa.assay import relationship_metrics  # noqa: E402
from clm_jepa.modeling import add_predictor_tokens  # noqa: E402


MODEL = ROOT / "data" / "models" / "llama-3.2-1b-instruct"
TRAIN = ROOT / "_references" / "llm-jepa" / "datasets" / "synth_train.jsonl"
TEST = ROOT / "_references" / "llm-jepa" / "datasets" / "synth_test.jsonl"
MANIFEST = ROOT / "data" / "manifests" / "gate3_multi" / "nl_rx_synth_llama.csv"
OUTPUT = ROOT / "artifacts" / "gate3" / "multi" / "nl_rx_synth_llama.json"
SAMPLE_SIZE = 1024
SEED = 533


def read_rows(path: Path) -> list[tuple[str, str, str]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            messages = json.loads(line)["messages"]
            rows.append((messages[0]["content"], messages[1]["content"], messages[2]["content"]))
    return rows


def deterministic_sample(rows: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    heap: list[tuple[int, str, str, str]] = []
    seen_targets: set[str] = set()
    for system, user, assistant in rows:
        if assistant in seen_targets:
            continue
        seen_targets.add(assistant)
        priority = int.from_bytes(
            hashlib.sha256(f"{SEED}|nl_rx_synth|{user}|{assistant}".encode()).digest()[:8], "big"
        )
        item = (-priority, assistant, system, user)
        if len(heap) < SAMPLE_SIZE:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    if len(heap) != SAMPLE_SIZE:
        raise RuntimeError(f"only {len(heap)} unique NL-RX-SYNTH targets were available")
    return [(system, user, assistant) for _, assistant, system, user in sorted(heap, key=lambda x: (-x[0], x[1]))]


def encode_chat(tokenizer, role: str, content: str) -> torch.Tensor:
    # This reproduces official LLM-JEPA's isolated one-message view and its
    # tokenize=False -> tokenizer(add_special_tokens=True) sequence.
    formatted = tokenizer.apply_chat_template(
        [{"role": role, "content": content}], tokenize=False, add_generation_prompt=False
    )
    values = tokenizer(
        formatted, truncation=True, max_length=512, add_special_tokens=True
    )["input_ids"]
    return torch.tensor(values, dtype=torch.long)


@torch.inference_mode()
def encode_position(model, rows: list[torch.Tensor], pad_id: int, *, offset: int, batch_size: int = 16) -> torch.Tensor:
    output = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        padded = pad_sequence(chunk, batch_first=True, padding_value=pad_id).to(model.device)
        attention = padded.ne(pad_id)
        hidden = model(input_ids=padded, attention_mask=attention, output_hidden_states=True).hidden_states[-1]
        indices = attention.sum(dim=1) - offset
        output.append(hidden[torch.arange(len(chunk), device=model.device), indices].float())
    return torch.cat(output)


def insert_before_eot(row: torch.Tensor, suffix: list[int]) -> torch.Tensor:
    if not suffix:
        return row
    return torch.cat((row[:-1], row.new_tensor(suffix), row[-1:]))


def main() -> None:
    torch.manual_seed(SEED)
    train = read_rows(TRAIN)
    test = read_rows(TEST)
    selected = deterministic_sample(train)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["system", "src", "tgt"])
        writer.writeheader()
        writer.writerows({"system": system, "src": user, "tgt": assistant} for system, user, assistant in selected)

    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
    if tokenizer.pad_token_id is None:
        raise RuntimeError("Llama tokenizer does not expose its reserved fine-tuning pad token")
    predictor_ids = add_predictor_tokens(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=False
    )
    old_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    with torch.no_grad():
        model.get_input_embeddings().weight[old_size:] = model.get_input_embeddings().weight[:old_size].mean(dim=0, keepdim=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model = model.cuda().eval()

    # Official LLM-JEPA isolates messages[1:2] and messages[2:3], excluding system.
    sources = [encode_chat(tokenizer, "user", user) for _, user, _ in selected]
    targets = [encode_chat(tokenizer, "assistant", assistant) for _, _, assistant in selected]
    identities = [assistant for _, _, assistant in selected]
    target_states = encode_position(model, targets, tokenizer.pad_token_id, offset=1)
    token_lengths = {identity: len(row) for identity, row in zip(identities, targets)}
    expression_sizes = {identity: len(identity) for identity in identities}

    raw = {}
    source_states = {}
    for k in (-1, 0, 1, 2, 3):
        suffix = [] if k < 1 else list(reversed(predictor_ids[:k]))
        rows = [insert_before_eot(source, suffix) for source in sources]
        # Predictor tokens precede the chat-template EOT exactly as in LLM-JEPA.
        offset = 2 if k == -1 or k >= 1 else 1
        source_states[k] = encode_position(model, rows, tokenizer.pad_token_id, offset=offset)
        raw[str(k)] = relationship_metrics(
            source_states[k], target_states, identities, token_lengths, expression_sizes, centered=False
        )
        print(json.dumps({"k": k, **raw[str(k)]}), flush=True)

    retained = [int(k) for k, value in raw.items() if value["retains_pair_signal"]]
    rescue = None
    if not retained:
        rescue = {
            str(k): relationship_metrics(
                source_states[k], target_states, identities, token_lengths, expression_sizes, centered=True
            )
            for k in (-1, 0, 1, 2, 3)
        }
        retained = [int(k) for k, value in rescue.items() if value["retains_pair_signal"]]

    train_pairs = {(user, assistant) for _, user, assistant in train}
    test_pairs = {(user, assistant) for _, user, assistant in test}
    output = {
        "dataset": "NL-RX-SYNTH",
        "checkpoint": "meta-llama/Llama-3.2-1B-Instruct",
        "revision": "9213176726f574b556790deb65791e0c5aa438b6a",
        "source_files": {"train_rows": len(train), "test_rows": len(test)},
        "sample_size": len(selected),
        "seed": SEED,
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "train_test_exact_pair_overlap": len(train_pairs & test_pairs),
        "train_test_user_overlap": len({x[0] for x in train_pairs} & {x[0] for x in test_pairs}),
        "train_test_target_overlap": len({x[1] for x in train_pairs} & {x[1] for x in test_pairs}),
        "view_policy": "official isolated user and assistant one-message Llama chat templates; system omitted",
        "predictor_policy": "descending predictor tokens inserted immediately before user EOT",
        "position_policy": {"-1": "source second-to-last pre-EOT token", "0": "source final EOT", "1..3": "final predictor before EOT", "target": "final EOT"},
        "matching_size": "assistant expression character count",
        "raw": raw,
        "rescue": rescue,
        "retained_k": retained,
        "stop": not bool(retained),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"retained_k": retained, "stop": output["stop"]}))


if __name__ == "__main__":
    main()
