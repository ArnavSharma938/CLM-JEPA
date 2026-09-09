"""Frozen teacher-forced checkpoint diagnostics for faithful NextLat."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chemfm import MODEL_DIR, TOKENIZER_DIR, add_predictor_tokens, load_adapter_checkpoint, load_lora_model, load_reaction_tokenizer
from faithful_nextlat import FaithfulNextLatObjective
from latent_predictability import reaction_balanced_indices, sha256_file

OUT = ROOT / "runs/faithful_nextlat/diagnostics"
PROBE_SEED, CAP = 20260904, 2048

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def make_plan(records, split):
    meta=[]
    for r in sorted((x for x in records if x["split"]==split), key=lambda x:x["reaction_identity"]):
        pos=list(map(int,r["product_indices"]))
        meta += [{"reaction_identity":r["reaction_identity"],"offset":j,"absolute_t":pos[j]}
                 for j in range(len(pos)-2)]
    chosen=reaction_balanced_indices(meta,CAP,PROBE_SEED)
    rows=[meta[i] for i in chosen.tolist()]
    digest=hashlib.sha256(json.dumps([(r["reaction_identity"],r["absolute_t"]) for r in rows],
                                     separators=(",", ":")).encode()).hexdigest()
    return rows,digest

@torch.inference_mode()
def diagnose(seed,epoch,records,plans,tokenizer,native_vocab):
    checkpoint=(ROOT/f"runs/faithful_nextlat/seed_{seed}/extension_corrected/checkpoints/step_480" if epoch==6 else
                ROOT/f"runs/faithful_nextlat/seed_{seed}/training/checkpoints/epoch_{epoch}")
    model=load_lora_model(MODEL_DIR,tokenizer,chemfm_vocab_size=native_vocab,attention_dropout=0.0,
                          attn_implementation="sdpa",lora_rank=8,lora_alpha=8).cuda().eval()
    load_adapter_checkpoint(model,checkpoint)
    method=FaithfulNextLatObjective(model.get_output_embeddings().weight[:native_vocab],
        product_start_token_id=tokenizer.convert_tokens_to_ids("<prostart>"),eos_token_id=tokenizer.eos_token_id,
        native_vocab_size=native_vocab,resume_state=checkpoint/"auxiliary_training_state.pt").cuda().eval()
    byid={r["reaction_identity"]:r for r in records}; llama=model.base_model.model.model
    embed=model.get_input_embeddings(); head=model.get_output_embeddings().weight[:native_vocab].detach()
    result={"seed":seed,"epoch":epoch,"step":80*epoch,"checkpoint":str(checkpoint.resolve()),
            "adapter_sha256":sha256_file(checkpoint/"USPTO-MIT-Synthesis/adapter_model.safetensors"),
            "auxiliary_sha256":sha256_file(checkpoint/"auxiliary_training_state.pt"),"splits":{}}
    for split,plan in plans.items():
        offsets={}
        for item in plan: offsets.setdefault(item["reaction_identity"],[]).append(item["offset"])
        latent_sum=kl_sum=rank_sum=margin_sum=ntp_sum=0.0; n=ntp_n=0
        for identity in sorted(offsets):
            row=byid[identity]; ids=row["input_ids"].long().cuda(); pos=list(map(int,row["product_indices"]))
            hidden=llama(input_ids=ids[None],attention_mask=torch.ones_like(ids)[None],use_cache=False,
                         return_dict=True).last_hidden_state[0]
            chosen=offsets[identity]; absolute=torch.tensor([pos[j] for j in chosen],device="cuda")
            current=hidden[absolute]; future=hidden[absolute+1]; action=embed(ids[absolute+1])
            mask=torch.ones(1,len(chosen),dtype=torch.bool,device="cuda")
            lz,kl,pred=method.losses(current[None],future[None],action[None],mask,head)
            logits=F.linear(pred[0].float(),head.float()); gold=ids[absolute+2]
            gold_logits=logits[torch.arange(len(gold),device="cuda"),gold]
            other=logits.clone(); other[torch.arange(len(gold),device="cuda"),gold]=-torch.inf
            latent_sum += float(lz)*len(chosen); kl_sum += float(kl)*len(chosen)
            rank_sum += float((logits>gold_logits[:,None]).sum(-1).add(1).sum())
            margin_sum += float((gold_logits-other.max(-1).values).sum()); n += len(chosen)
            targets=ids[pos[0]:pos[-1]+2]; ntp_logits=F.linear(hidden[pos[0]-1:pos[-1]+1].float(),head.float())
            ntp_sum += float(F.cross_entropy(ntp_logits,targets,reduction="sum")); ntp_n += len(targets)
        result["splits"][split]={"positions":n,"reactions":len(offsets),"latent_smooth_l1":latent_sum/n,
            "teacher_student_kl":kl_sum/n,"predicted_gold_rank":rank_sum/n,"predicted_gold_margin":margin_sum/n,
            "ordinary_ntp_loss":ntp_sum/ntp_n}
    del method,model; torch.cuda.empty_cache(); return result

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--extension-only",action="store_true"); args=parser.parse_args()
    tokenizer=load_reaction_tokenizer(TOKENIZER_DIR); native_vocab=len(tokenizer); add_predictor_tokens(tokenizer)
    if args.extension_only:
        seed=533; path=OUT/f"seed_{seed}.json"; existing=json.loads(path.read_text())
        records=torch.load(ROOT/f"runs/oracle_transition/cache_native_{seed}.pt",map_location="cpu",weights_only=False)["records"]
        plans={split:make_plan(records,split)[0] for split in ("train","validation")}
        existing["checkpoints"]=[c for c in existing["checkpoints"] if c["step"]!=480]
        existing["checkpoints"].append(diagnose(seed,6,records,plans,tokenizer,native_vocab)); write(path,existing); return
    for seed in (533,917):
        records=torch.load(ROOT/f"runs/oracle_transition/cache_native_{seed}.pt",map_location="cpu",weights_only=False)["records"]
        plans={}; hashes={}
        for split in ("train","validation"): plans[split],hashes[split]=make_plan(records,split)
        outputs=[diagnose(seed,e,records,plans,tokenizer,native_vocab) for e in (1,2,3,4)]
        write(OUT/f"seed_{seed}.json",{"selected_key_hashes":hashes,"cap":CAP,"checkpoints":outputs})
    write(OUT/"complete.json",{"status":"complete","seeds":[533,917]})

if __name__=="__main__": main()
