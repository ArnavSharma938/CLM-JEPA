#!/usr/bin/env python3
"""Extract only final post-RMSNorm product states for Report-03 rows.

Native states are reused from the immutable oracle-transition caches.  This
script performs no training and writes one compact cache per treatment/seed.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from chemfm import MODEL_DIR, TOKENIZER_DIR, add_predictor_tokens, load_adapter_checkpoint, load_lora_model, load_reaction_tokenizer
from run_geodesic_audit import SelectedStateCapture

TREATMENTS = {
    "projected": "runs/decoder_projected/seed_{seed}/training/checkpoints/epoch_4",
    "full_state": "runs/decoder_projected/full_state_nextlat/seed_{seed}/training/checkpoints/epoch_4",
}

def native_records(seed):
    payload = torch.load(ROOT / f"runs/oracle_transition/cache_native_{seed}.pt", map_location="cpu", weights_only=False)
    return payload["records"]

def extract_one(seed, arm, device, batch_size):
    out = ROOT / f"runs/decoder_projected/representation/cache/{arm}_{seed}.pt"
    if out.exists():
        print(json.dumps({"reused": str(out)}), flush=True); return
    base = native_records(seed)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR); vocab = len(tokenizer); add_predictor_tokens(tokenizer)
    model = load_lora_model(MODEL_DIR, tokenizer, chemfm_vocab_size=vocab, attention_dropout=0.0, attn_implementation="sdpa", lora_rank=8, lora_alpha=8).to(device).eval()
    load_adapter_checkpoint(model, ROOT / TREATMENTS[arm].format(seed=seed))
    for p in model.parameters(): p.requires_grad_(False)
    capture = SelectedStateCapture(model); records=[]; start=time.perf_counter()
    ordered = sorted(base, key=lambda r: len(r["input_ids"]))
    with torch.inference_mode():
        for begin in range(0, len(ordered), batch_size):
            batch=ordered[begin:begin+batch_size]; maxlen=max(len(r["input_ids"]) for r in batch)
            ids=torch.zeros((len(batch),maxlen),dtype=torch.long,device=device); mask=torch.zeros_like(ids,dtype=torch.bool)
            for i,r in enumerate(batch):
                n=len(r["input_ids"]); ids[i,:n]=r["input_ids"].to(device); mask[i,:n]=True
            capture.clear(); model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
            states=capture.values["final_post_norm"]
            for i,r in enumerate(batch):
                n=len(r["input_ids"]); records.append({"reaction_identity":r["reaction_identity"],"split":r["split"],"input_ids":r["input_ids"],"product_indices":r["product_indices"],"final_states":states[i,:n].float().cpu()})
    capture.close(); model.cpu(); del model
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"type":"decoder_projected_final_product_state_cache","arm":arm,"seed":seed,"records":sorted(records,key=lambda r:r["reaction_identity"]),"seconds":time.perf_counter()-start},out)
    print(json.dumps({"arm":arm,"seed":seed,"records":len(records),"seconds":time.perf_counter()-start,"path":str(out)}),flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--device",default="cuda"); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--seeds",default="533,917"); p.add_argument("--arms",default="projected,full_state"); a=p.parse_args()
    for arm in a.arms.split(','):
        for seed in map(int,a.seeds.split(',')): extract_one(seed,arm,a.device,a.batch_size)
if __name__=="__main__": main()
