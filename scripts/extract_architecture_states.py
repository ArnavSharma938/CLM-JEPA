"""Common, fail-closed final-state extraction for Report-07 model arms."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from architecture_analysis import product_transition_rows
from chemfm import (MODEL_DIR, TOKENIZER_DIR, add_predictor_tokens,
                    load_adapter_checkpoint, load_lora_model, load_reaction_tokenizer)
from latent_predictability import sha256_file

OUT = ROOT / "runs/faithful_nextlat/architecture/cache"
CHECKPOINTS = {
    "faithful_nextlat": "runs/faithful_nextlat/seed_{seed}/training/checkpoints/epoch_4",
    "decoder_projected": "runs/decoder_projected/seed_{seed}/training/checkpoints/epoch_4",
}


def active_weight(module):
    if hasattr(module, "modules_to_save"):
        module = module.modules_to_save["USPTO-MIT-Synthesis"]
    return module.weight.detach()


def tensor_sha(value):
    array = value.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def native(seed):
    source = ROOT / f"runs/oracle_transition/cache_native_{seed}.pt"
    payload = torch.load(source, map_location="cpu", weights_only=False)
    assert len(payload["records"]) == 1024
    assert all(len(row["final_states"]) == len(row["product_indices"])
               for row in payload["records"])
    product_transition_rows(payload["records"][:2])
    payload["provenance"] = {**payload["provenance"], "source_cache": str(source.resolve()),
                             "source_cache_sha256": sha256_file(source),
                             "report07_reuse": "known-correct Report-03 product-sliced cache"}
    return payload


def extract(seed, arm, batch_size):
    destination = OUT / f"{arm}_{seed}.pt"
    if destination.exists():
        saved = torch.load(destination, map_location="cpu", weights_only=False)
        assert len(saved["records"]) == 1024
        product_transition_rows(saved["records"][:2])
        print(json.dumps({"reused": str(destination)}), flush=True)
        return
    if arm == "native":
        payload = native(seed)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, destination)
        print(json.dumps({"arm": arm, "seed": seed, "reused_report03": True}), flush=True)
        return
    base = native(seed)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    native_vocab = len(tokenizer)
    add_predictor_tokens(tokenizer)
    model = load_lora_model(MODEL_DIR, tokenizer, chemfm_vocab_size=native_vocab,
                            attention_dropout=0.0, attn_implementation="sdpa",
                            lora_rank=8, lora_alpha=8).cuda().eval()
    checkpoint = ROOT / CHECKPOINTS[arm].format(seed=seed)
    load_adapter_checkpoint(model, checkpoint)
    model.requires_grad_(False)
    llama = model.base_model.model.model
    ordered = sorted(base["records"], key=lambda row: (len(row["input_ids"]), row["reaction_identity"]))
    records = []
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    with torch.inference_mode():
        for begin in range(0, len(ordered), batch_size):
            batch = ordered[begin:begin + batch_size]
            maximum = max(len(row["input_ids"]) for row in batch)
            ids = torch.zeros((len(batch), maximum), dtype=torch.long, device="cuda")
            mask = torch.zeros_like(ids)
            for index, row in enumerate(batch):
                length = len(row["input_ids"])
                ids[index, :length] = row["input_ids"].long().cuda()
                mask[index, :length] = 1
            states = llama(input_ids=ids, attention_mask=mask, use_cache=False,
                           return_dict=True).last_hidden_state
            for index, row in enumerate(batch):
                positions = list(map(int, row["product_indices"]))
                sliced = states[index, positions].to(torch.bfloat16).cpu()
                assert sliced.shape == (len(positions), 2048)
                records.append({"reaction_identity": row["reaction_identity"],
                                "split": row["split"], "input_ids": row["input_ids"],
                                "product_indices": positions, "final_states": sliced})
        embedding = active_weight(model.get_input_embeddings()).float().cpu().clone()
        lm_head = active_weight(model.get_output_embeddings()).float().cpu().clone()
    product_transition_rows(records[:2])
    adapter = checkpoint / "USPTO-MIT-Synthesis/adapter_model.safetensors"
    payload = {
        "type": "report07_product_sliced_final_post_rmsnorm",
        "arm": arm, "seed": seed, "records": sorted(records, key=lambda row: row["reaction_identity"]),
        "embedding": embedding, "lm_head": lm_head,
        "provenance": {"checkpoint": str(checkpoint.resolve()),
                       "checkpoint_sha256": sha256_file(adapter),
                       "embedding_fp32_sha256": tensor_sha(embedding),
                       "lm_head_fp32_sha256": tensor_sha(lm_head),
                       "source_identity_cache_sha256": base["provenance"]["source_cache_sha256"],
                       "state_indexing": "last_hidden_state[:, product_indices]",
                       "state_layer": "final_post_RMSNorm", "records": len(records),
                       "seconds": time.perf_counter() - started,
                       "peak_cuda_bytes": torch.cuda.max_memory_allocated()},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    print(json.dumps(payload["provenance"]), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="533,917")
    parser.add_argument("--arms", default="native,faithful_nextlat,decoder_projected")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    for seed in map(int, args.seeds.split(",")):
        for arm in args.arms.split(","):
            extract(seed, arm, args.batch_size)


if __name__ == "__main__":
    main()
