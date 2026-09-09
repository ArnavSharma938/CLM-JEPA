"""One prespecified seed-533 continuation from step 320 to step 480."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import set_seed

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"scripts"))
from chemfm import MODEL_DIR,TOKENIZER_DIR,ADAPTER_NAME,ReactionCollator,add_predictor_tokens,load_adapter_checkpoint,load_lora_model,load_reaction_tokenizer
from faithful_nextlat import FaithfulNextLatObjective
from latent_predictability import sha256_file
from run_stp_matrix import TRAIN
from train import read_rows,gradient_diagnostics

SEED=533; SOURCE=ROOT/"runs/faithful_nextlat/seed_533/training/checkpoints/epoch_4"
OUT=ROOT/"runs/faithful_nextlat/seed_533/extension_corrected"; DEST=OUT/"checkpoints/step_480"

def main():
    if (OUT/"result.json").exists(): return
    set_seed(SEED); tokenizer=load_reaction_tokenizer(TOKENIZER_DIR); native_vocab=len(tokenizer); add_predictor_tokens(tokenizer)
    model=load_lora_model(MODEL_DIR,tokenizer,chemfm_vocab_size=native_vocab,attention_dropout=0.0,
                          attn_implementation="sdpa",lora_rank=8,lora_alpha=8).cuda().train(); load_adapter_checkpoint(model,SOURCE)
    method=FaithfulNextLatObjective(model.get_output_embeddings().weight[:native_vocab],
        product_start_token_id=tokenizer.convert_tokens_to_ids("<prostart>"),eos_token_id=tokenizer.eos_token_id,
        native_vocab_size=native_vocab,resume_state=SOURCE/"auxiliary_training_state.pt").cuda()
    parameters=[p for p in model.parameters() if p.requires_grad]+list(method.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01,fused=True)
    state=torch.load(SOURCE/"training_state.pt",map_location="cuda",weights_only=False); assert state["global_step"]==320
    optimizer.load_state_dict(state["optimizer"])
    for group in optimizer.param_groups: group["lr"]=1e-5
    generator=torch.Generator(); generator.set_state(state["loader_generator_state"].cpu())
    loader=DataLoader(read_rows("uspto_mit_synthesis",path=TRAIN),batch_size=4,shuffle=True,generator=generator,
                      collate_fn=ReactionCollator(tokenizer),pin_memory=True)
    optimizer.zero_grad(set_to_none=True); curves=[]; step=320; started=time.perf_counter()
    for epoch in (5,6):
        window=[]
        for index,raw in enumerate(loader):
            batch={k:v.cuda(non_blocking=True) for k,v in raw.items() if torch.is_tensor(v)}
            output=method(model,batch); (output.loss/4).backward(); window.append((float(output.native_loss.detach()),float(output.lz.detach()),float(output.kl.detach())))
            if (index+1)%4: continue
            total,largest=gradient_diagnostics(model,(("faithful_nextlat",method),)); optimizer.step(); optimizer.zero_grad(set_to_none=True); step+=1
            curves.append({"step":step,"epoch":epoch,"learning_rate":1e-5,"native_loss":sum(x[0] for x in window)/4,
                           "latent":sum(x[1] for x in window)/4,"kl":sum(x[2] for x in window)/4,
                           "chemfm_gradient_norm":total,"largest_gradient":largest}); window=[]
        assert not window and step==80*epoch
    DEST.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(DEST,safe_serialization=True,selected_adapters=[ADAPTER_NAME],save_embedding_layers=False)
    tokenizer.save_pretrained(DEST); method.save_training_state(DEST/"auxiliary_training_state.pt")
    torch.save({"source":str(SOURCE.resolve()),"source_global_step":320,"global_step":480,
                "continuation_steps":160,"lr_policy":"terminal ChemFM min LR held fixed at 1e-5",
                "global_gradient_clip":"ChemFM and predictor jointly, norm 1.0",
                "optimizer":optimizer.state_dict(),"loader_generator_state":generator.get_state()},DEST/"extension_state.pt")
    result={"seed":SEED,"source_checkpoint":str(SOURCE.resolve()),"source_adapter_sha256":sha256_file(SOURCE/ADAPTER_NAME/"adapter_model.safetensors"),
            "checkpoint":str(DEST.resolve()),"optimizer_steps_total":480,"continuation_steps":160,"learning_rate":1e-5,
            "global_gradient_clip":"ChemFM and predictor jointly, norm 1.0",
            "curves":curves,"seconds":time.perf_counter()-started}
    (OUT/"result.json").write_text(json.dumps(result,indent=2)+"\n")
if __name__=="__main__": main()
