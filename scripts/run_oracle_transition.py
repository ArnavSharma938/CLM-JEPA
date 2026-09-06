"""Minimal frozen Native final-state oracle transition decomposition.

Stages are resumable; extraction never computes gradients or generates tokens.
All scientific evaluation uses the same product t-3..t+2 rows.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from latent_predictability import (
    TargetBasis, Standardizer, RidgeProbe, ResidualMLPProbe, fit_probe,
    reaction_balanced_indices, shuffled_reaction_targets,
    decoder_distribution_metrics, sha256_file, assert_disjoint_confirmation,
)

OUT = ROOT / "runs/oracle_transition"
REPORT = ROOT / "docs/reports/03_NATIVE_ORACLE_TRANSITION_DECOMPOSITION.md"
SPLIT = ROOT / "data/clm_jepa_uspto_mit_latent_audit/splits.json"
PANEL = ROOT / "data/clm_jepa_uspto_mit_validation_1024/uspto_mit_validation_1024.csv"
SEEDS = (533, 917)
PROBE_SEED = 20260904
ARMS = ("A", "B", "C", "D", "C_shuffle")


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def log(**value):
    print(json.dumps(value), flush=True)


def checkpoint(seed):
    return ROOT / f"runs/pair_residual/a6000/results/seed_{seed}/native/training/checkpoints/epoch_4"


def active_weight(module):
    if hasattr(module, "modules_to_save"):
        module = module.modules_to_save["USPTO-MIT-Synthesis"]
    return module.weight.detach()


def extract(smoke=False):
    from chemfm import (MODEL_DIR, TOKENIZER_DIR, load_reaction_tokenizer,
                        load_lora_model, ReactionCollator, REACTANT_START, PRODUCT_START, END)
    from jepa import add_predictor_tokens
    from train import load_adapter_checkpoint
    from frozen_geometry import _sampled_parameter_fingerprint
    split = json.loads(SPLIT.read_text())
    assert Counter(r["split"] for r in split["records"]) == {"train":640,"validation":192,"test":192}
    assert_disjoint_confirmation(
        [r["chemical_pair_id"] for r in split["records"]],
        ROOT / "data/clm_jepa_uspto_mit_stp_confirmation/untouched_640.jsonl")
    assignment = {r["reaction_identity"]:r["split"] for r in split["records"]}
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    vocab = len(tokenizer)
    add_predictor_tokens(tokenizer)
    collator = ReactionCollator(tokenizer)
    examples = []
    with PANEL.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source, target = row["source"], row["target"]
            prefix = f"{REACTANT_START}{source}{END}{PRODUCT_START}"
            encoded = tokenizer(prefix + target + END, add_special_tokens=False, return_offsets_mapping=True)
            assert encoded["input_ids"] == collator([{"src":source,"tgt":target}])["input_ids"][0].tolist()
            positions = [i for i,(a,b) in enumerate(encoded["offset_mapping"])
                         if a >= len(prefix) and b <= len(prefix)+len(target)]
            assert positions == list(range(positions[0], positions[-1]+1))
            examples.append(dict(reaction_identity=row["reaction_identity"],
                                 split=assignment[row["reaction_identity"]],
                                 input_ids=torch.tensor(encoded["input_ids"],dtype=torch.int32),
                                 product_indices=positions))
    assert len(examples) == len(assignment) == 1024
    assert {r["reaction_identity"] for r in examples} == set(assignment)
    examples.sort(key=lambda r:(len(r["input_ids"]),r["reaction_identity"]))
    # A longest TRAIN sequence is a conservative memory smoke check, never a test metric.
    if smoke:
        examples = [max((r for r in examples if r["split"]=="train"),key=lambda r:len(r["input_ids"]))]
    OUT.mkdir(parents=True, exist_ok=True)
    log(stage="extraction_plan",reactions=len(examples),max_length=max(len(r["input_ids"]) for r in examples),
        product_positions=sum(len(r["product_indices"]) for r in examples),batch_size=1)
    model = load_lora_model(MODEL_DIR,tokenizer,chemfm_vocab_size=vocab,
                           attention_dropout=0,attn_implementation="sdpa",lora_rank=8,lora_alpha=8).to("cuda").eval()
    model.requires_grad_(False)
    llama = model.base_model.model.model
    for seed in SEEDS[:1] if smoke else SEEDS:
        destination = OUT / f"cache_native_{seed}.pt"
        if destination.exists() and not smoke:
            saved=load(destination)
            assert len(saved["records"])==1024 and saved["provenance"]["seed"]==seed
            save_json(OUT/f"extraction_{seed}.json",{**saved["provenance"],
                      "cache_bytes":destination.stat().st_size,"cache_sha256":sha256_file(destination)})
            del saved
            log(stage="cache_reused",seed=seed)
            continue
        config = json.loads((checkpoint(seed)/"USPTO-MIT-Synthesis/adapter_config.json").read_text())
        assert config["r"] == config["lora_alpha"] == 8
        load_adapter_checkpoint(model,checkpoint(seed))
        fingerprint = _sampled_parameter_fingerprint(model)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        records = []
        with torch.inference_mode():
            for row in examples:
                ids = row["input_ids"].long().unsqueeze(0).cuda()
                hidden = llama(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,
                               return_dict=True).last_hidden_state
                states = hidden[0,row["product_indices"]].to(torch.bfloat16).cpu()
                assert states.shape == (len(row["product_indices"]),2048)
                assert torch.isfinite(states).all()
                records.append({**row,"final_states":states})
            embedding = active_weight(model.get_input_embeddings()).cpu().clone()
            head = active_weight(model.get_output_embeddings()).cpu().clone()
            # Verify saved matrix implements the active unchanged module on real states.
            actual = model.get_output_embeddings()(hidden[0,-2:])
            replay = F.linear(hidden[0,-2:],head.cuda())
            assert torch.equal(actual,replay)
        torch.cuda.synchronize()
        assert fingerprint == _sampled_parameter_fingerprint(model)
        info = dict(seed=seed,seconds=time.perf_counter()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                    split_sha256=sha256_file(SPLIT),panel_sha256=sha256_file(PANEL),
                    checkpoint_sha256=sha256_file(checkpoint(seed)/"USPTO-MIT-Synthesis/adapter_model.safetensors"),
                    frozen_fingerprint=fingerprint,torch_version=torch.__version__,native_vocab=vocab,
                    lm_head_parity="bitwise",model_forward="LlamaModel final post-RMSNorm; no LM loss or generation")
        if smoke:
            save_json(OUT/"extraction_smoke.json",info)
        else:
            torch.save(dict(records=records,embedding=embedding,lm_head=head,provenance=info),destination)
            save_json(OUT/f"extraction_{seed}.json",{**info,"cache_bytes":destination.stat().st_size,
                                                      "cache_sha256":sha256_file(destination)})
        log(stage="smoke_complete" if smoke else "extraction_complete",**info)


def rows(payload, split):
    """Product-local offsets map explicitly to original absolute causal indices."""
    records = sorted((r for r in payload["records"] if r["split"]==split),key=lambda r:r["reaction_identity"])
    history,targets,tokens,gold,metadata = [],[],[],[],[]
    for record in records:
        positions = record["product_indices"]
        for t in range(3,len(positions)-2):
            absolute = positions[t]
            assert positions[t-3:t+3] == list(range(absolute-3,absolute+3))
            past = positions[t-3:t+1]
            assert max(past) == absolute and all(i<=absolute for i in past)
            history.append(record["final_states"][t-3:t+1].reshape(-1))
            targets.append(record["final_states"][t+1])
            tokens.append(int(record["input_ids"][absolute+1]))
            gold.append(int(record["input_ids"][absolute+2]))
            metadata.append(dict(reaction_identity=record["reaction_identity"],current_index=absolute,
                                 future_index=t+1,sequence_length=len(positions),product_offset=t,
                                 input_indices=past,target_index=absolute+1,decoder_target_index=absolute+2))
    return dict(history=torch.stack(history),y=torch.stack(targets),token=torch.tensor(tokens),
                gold=torch.tensor(gold),metadata=metadata)


def inputs(data, embedding, arm):
    state = data["history"][:,-2048:].float()
    token = embedding[data["token"]].float()
    if arm == "A": return state
    if arm == "B": return token
    if arm == "C_shuffle":
        token, donor = shuffled_reaction_targets(token,data["metadata"],PROBE_SEED+1)
        assert all(data["metadata"][i]["reaction_identity"] != data["metadata"][j]["reaction_identity"]
                   for i,j in enumerate(donor.tolist()))
    if arm in ("C","C_shuffle"): return torch.cat((state,token),dim=1)
    assert arm == "D"
    return torch.cat((data["history"].float(),token),dim=1)


def prepare(seed):
    path = OUT/f"preprocessing_{seed}.pt"
    if path.exists(): return load(path)
    payload = load(OUT/f"cache_native_{seed}.pt")
    train = rows(payload,"train")
    chosen = reaction_balanced_indices(train["metadata"],16384,PROBE_SEED)
    # Identical selected training rows for every arm and both checkpoints.
    selected_keys = [(train["metadata"][i]["reaction_identity"],train["metadata"][i]["current_index"])
                     for i in chosen.tolist()]
    basis = TargetBasis.fit(train["y"][chosen].float().cuda(),256,PROBE_SEED+1)
    result = dict(basis=basis,chosen=chosen,selected_keys=selected_keys)
    torch.save(result,path)
    log(stage="pca_complete",seed=seed,variance_coverage=basis.variance_coverage,train_positions=len(chosen))
    return result


def probe_smoke():
    """One real training batch per input width, including frozen-ridge gradients."""
    prep=prepare(533)
    payload=load(OUT/"cache_native_533.pt")
    train=rows(payload,"train")
    chosen=prep["chosen"][:512]
    batch={k:([v[i] for i in chosen.tolist()] if k=="metadata" else v[chosen]) for k,v in train.items()}
    del train
    target=prep["basis"].encode(batch["y"])
    records=[]
    for arm in ("A","B","C","D"):
        x=inputs(batch,payload["embedding"],arm)
        standardized=Standardizer.fit(x)(x).cuda()
        torch.manual_seed(PROBE_SEED)
        ridge=RidgeProbe(x.shape[1],256).cuda()
        model=ResidualMLPProbe(ridge,x.shape[1],256,128).cuda()
        before={k:v.clone() for k,v in model.ridge.state_dict().items()}
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=1e-4)
        torch.cuda.synchronize();start=time.perf_counter()
        prediction=model(standardized)
        loss=F.mse_loss(prediction,target.cuda())
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None for p in model.ridge.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.residual.parameters())
        optimizer.step()
        assert all(torch.equal(v,model.ridge.state_dict()[k]) for k,v in before.items())
        torch.cuda.synchronize()
        seconds=time.perf_counter()-start
        reconstruction=prep["basis"].decode(prediction.detach().cpu())
        assert reconstruction.shape==(len(chosen),2048) and torch.isfinite(reconstruction).all()
        records.append(dict(arm=arm,input_shape=list(x.shape),score_shape=list(prediction.shape),
                            loss=float(loss),batch_seconds=seconds,frozen_ridge_unchanged=True))
        del model,ridge,optimizer,standardized,prediction,loss
    save_json(OUT/"probe_smoke.json",records)
    log(stage="probe_smoke_complete",batches=records)


def fit(seed, stage):
    prep = prepare(seed)
    payload = load(OUT/f"cache_native_{seed}.pt")
    chosen,basis = prep["chosen"],prep["basis"]
    train=rows(payload,"train")
    data={"train":{k:([v[i] for i in chosen.tolist()] if k=="metadata" else v[chosen])
                   for k,v in train.items()}}
    del train
    data["validation"]=rows(payload,"validation")
    embedding=payload["embedding"]
    del payload
    ty = basis.encode(data["train"]["y"])
    vy = basis.encode(data["validation"]["y"])
    for arm in ARMS if stage=="ridge" else ("A","C","D"):
        ridge_path = OUT/f"probe_{seed}_{arm}_ridge.pt"
        if stage == "ridge" and ridge_path.exists(): continue
        if stage == "mlp" and all((OUT/f"probe_{seed}_{arm}_mlp_{s}.pt").exists()
                                   for s in range(PROBE_SEED,PROBE_SEED+3)): continue
        start=time.perf_counter()
        tx = inputs(data["train"],embedding,arm)
        vx = inputs(data["validation"],embedding,arm)
        if arm=="C_shuffle" and stage=="ridge":
            diagnostics={}
            for split,d in data.items():
                _,donor=shuffled_reaction_targets(d["token"],d["metadata"],PROBE_SEED+1)
                diagnostics[split]=dict(
                    self_reaction_matches=sum(d["metadata"][i]["reaction_identity"]==d["metadata"][j]["reaction_identity"]
                                              for i,j in enumerate(donor.tolist())),
                    same_token_fraction=float(d["token"].eq(d["token"][donor]).float().mean()),
                    mean_absolute_product_position_gap=float(np.mean([
                        abs(d["metadata"][i]["future_index"]-d["metadata"][j]["future_index"])
                        for i,j in enumerate(donor.tolist())])),
                    mean_absolute_product_length_gap=float(np.mean([
                        abs(d["metadata"][i]["sequence_length"]-d["metadata"][j]["sequence_length"])
                        for i,j in enumerate(donor.tolist())])))
            save_json(OUT/f"shuffle_diagnostics_{seed}.json",diagnostics)
        if stage=="ridge":
            standardizer=Standardizer.fit(tx)
        else:
            artifact=load(ridge_path)
            standardizer=artifact["standardizer"]
        tx=standardizer(tx); vx=standardizer(vx)
        torch.manual_seed(PROBE_SEED)
        ridge=RidgeProbe(tx.shape[1],256).cuda()
        if stage=="ridge":
            ridge,trace=fit_probe(ridge,tx,ty,vx,vy,weight_decay=1e-3,epochs=20,batch_size=512,seed=PROBE_SEED)
            torch.save(dict(state={k:v.cpu() for k,v in ridge.state_dict().items()},standardizer=standardizer,
                            input_size=tx.shape[1],trace=trace),ridge_path)
            log(stage="ridge_complete",seed=seed,arm=arm,seconds=time.perf_counter()-start,**trace)
        else:
            ridge.load_state_dict(artifact["state"])
            for probe_seed in range(PROBE_SEED,PROBE_SEED+3):
                destination=OUT/f"probe_{seed}_{arm}_mlp_{probe_seed}.pt"
                if destination.exists(): continue
                torch.manual_seed(probe_seed)
                mlp=ResidualMLPProbe(ridge,tx.shape[1],256,128).cuda()
                mlp,trace=fit_probe(mlp,tx,ty,vx,vy,weight_decay=1e-4,epochs=20,batch_size=512,seed=probe_seed)
                torch.save(dict(state={k:v.cpu() for k,v in mlp.state_dict().items()},trace=trace),destination)
                log(stage="mlp_complete",seed=seed,arm=arm,probe_seed=probe_seed,**trace)
                del mlp
        del tx,vx,ridge
        gc.collect()


@torch.inference_mode()
def evaluate(seed,stage):
    destination=OUT/f"metrics_{seed}_{stage}.json"
    if destination.exists(): return
    payload=load(OUT/f"cache_native_{seed}.pt")
    prep=load(OUT/f"preprocessing_{seed}.pt")
    basis=prep["basis"]
    data=rows(payload,"test")
    head=payload["lm_head"].cuda()
    # The oracle request specifies the unchanged saved head: retain every output
    # (including reserved IDs), rather than Report 02's native-vocabulary renormalization.
    assert data["gold"].max()<len(head)
    identities=[m["reaction_identity"] for m in data["metadata"]]
    unique=sorted(set(identities))
    groups={r:np.array([i for i,x in enumerate(identities) if x==r]) for r in unique}
    results={}
    for arm in ARMS if stage=="ridge" else ("A","C","D"):
        artifact=load(OUT/f"probe_{seed}_{arm}_ridge.pt")
        x=artifact["standardizer"](inputs(data,payload["embedding"],arm))
        ridge=RidgeProbe(x.shape[1],256)
        ridge.load_state_dict(artifact["state"])
        for probe_seed in [PROBE_SEED] if stage=="ridge" else range(PROBE_SEED,PROBE_SEED+3):
            model=ridge if stage=="ridge" else ResidualMLPProbe(ridge,x.shape[1],256,128)
            if stage=="mlp": model.load_state_dict(load(OUT/f"probe_{seed}_{arm}_mlp_{probe_seed}.pt")["state"])
            model=model.cuda().eval()
            collected={}
            for start in range(0,len(x),256):
                stop=start+256
                pred=basis.decode(model(x[start:stop].cuda()).cpu())
                target=data["y"][start:stop].float()
                center=basis.mean
                values=dict(sse=(target-pred).square().sum(-1),sst=(target-center).square().sum(-1),
                            cosine=F.cosine_similarity(target,pred,dim=-1),
                            centered_cosine=F.cosine_similarity(target-center,pred-center,dim=-1))
                values.update(decoder_distribution_metrics(target.to(device="cuda",dtype=head.dtype)@head.T,
                                                          pred.to(device="cuda",dtype=head.dtype)@head.T,
                                                          data["gold"][start:stop].cuda()))
                for k,v in values.items(): collected.setdefault(k,[]).append(v.float().cpu())
            values={k:torch.cat(v).numpy() for k,v in collected.items()}
            reaction={}
            for r,idx in groups.items():
                entry={k:float(v[idx].mean()) for k,v in values.items()}
                entry.update(n=len(idx),sse=float(values["sse"][idx].sum()),sst=float(values["sst"][idx].sum()))
                entry["normalized_mse"]=entry["sse"]/entry["sst"]
                entry["r2"]=1-entry["normalized_mse"]
                reaction[r]=entry
            results[f"{arm}:{probe_seed}"]=reaction
            del model
        del x
    # Actual-state teacher metrics provide the functional reference for rank/margin.
    true_metrics={}
    if stage=="ridge":
        pieces={}
        for start in range(0,len(data["y"]),256):
            logits=data["y"][start:start+256].to(device="cuda",dtype=head.dtype)@head.T
            for k,v in decoder_distribution_metrics(logits,logits,data["gold"][start:start+256].cuda()).items():
                pieces.setdefault(k,[]).append(v.float().cpu())
        values={k:torch.cat(v).numpy() for k,v in pieces.items()}
        true_metrics={r:{k:float(v[idx].mean()) for k,v in values.items()} for r,idx in groups.items()}
    save_json(destination,dict(seed=seed,stage=stage,rows=len(identities),reactions=len(unique),
                              pca_variance_coverage=basis.variance_coverage,results=results,true_metrics=true_metrics,
                              row_keys=[(m["reaction_identity"],m["current_index"]) for m in data["metadata"]]))
    log(stage="evaluation_complete",seed=seed,probe_stage=stage,rows=len(identities),reactions=len(unique))


def check_common():
    left=load(OUT/"cache_native_533.pt")
    right=load(OUT/"cache_native_917.pt")
    a={r["reaction_identity"]:r for r in left["records"]}
    b={r["reaction_identity"]:r for r in right["records"]}
    assert a.keys()==b.keys()
    for key in a:
        assert a[key]["split"]==b[key]["split"]
        assert a[key]["product_indices"]==b[key]["product_indices"]
        assert torch.equal(a[key]["input_ids"],b[key]["input_ids"])
    counts={s:sum(max(0,len(r["product_indices"])-5) for r in a.values() if r["split"]==s)
            for s in ("train","validation","test")}
    save_json(OUT/"eligibility.json",dict(rows=counts,reaction_split=dict(Counter(r["split"] for r in a.values())),
              eligible_reaction_split=dict(Counter(r["split"] for r in a.values() if len(r["product_indices"])>=6)),
              identical_tokens_positions_splits=True,inputs="states t-3..t only; explicit oracle embedding x[t+1]",
              target="h[t+1]",decoder_target="x[t+2]",future_hidden_input_count=0))
    log(stage="common_rows_verified",**counts)


def summarize():
    metrics={}
    for seed in SEEDS:
        for stage in ("ridge","mlp"):
            metrics[seed,stage]=json.loads((OUT/f"metrics_{seed}_{stage}.json").read_text())
    keys=metrics[533,"ridge"]["row_keys"]
    assert all(m["row_keys"]==keys for m in metrics.values())
    assert load(OUT/"preprocessing_533.pt")["selected_keys"]==load(OUT/"preprocessing_917.pt")["selected_keys"]
    identities=sorted(next(iter(metrics[533,"ridge"]["results"].values())))
    names=[("ridge",arm) for arm in ARMS]+[("mlp",arm) for arm in ("A","C","D")]
    # Preserve pairing across arms and checkpoints; MLP replicates average metrics, not predictions.
    cubes={}
    for stage,arm in names:
        cubes[stage,arm]=[]
        for seed in SEEDS:
            samples=[v for k,v in metrics[seed,stage]["results"].items() if k.split(":")[0]==arm]
            cubes[stage,arm].append({r:{k:float(np.mean([v[r][k] for v in samples])) for k in samples[0][r]}
                                    for r in identities})
    rng=np.random.default_rng(PROBE_SEED)
    boot_reactions=rng.integers(len(identities),size=(4000,len(identities)))
    boot_seeds=rng.integers(2,size=(4000,2))
    def interval(array):
        draws=array[boot_seeds[:,:,None],boot_reactions[:,None,:]].mean((1,2))
        return np.quantile(draws,[.025,.975]).tolist()
    rows_out=[]
    for name,values in cubes.items():
        stage,arm=name
        fields=list(values[0][identities[0]])
        arrays={k:np.array([[v[r][k] for r in identities] for v in values]) for k in fields}
        for label,idx in [(533,0),(917,1),("paired mean",None)]:
            rows_out.append(dict(seed=label,stage=stage,arm=arm,
                **{k:float(a.mean() if idx is None else a[idx].mean()) for k,a in arrays.items()}))
    conditional={}
    for stage in ("ridge","mlp"):
        contrasts=[("P_token_given_h","A","C"),("P_history_given_h_token","C","D")]
        if stage=="ridge": contrasts.append(("P_h_given_token","B","C"))
        for label,baseline,improved in contrasts:
            array=np.array([[1-cubes[stage,improved][s][r]["sse"]/cubes[stage,baseline][s][r]["sse"]
                             for r in identities] for s in range(2)])
            conditional[f"{stage}:{label}"]=dict(seed_533=float(array[0].mean()),seed_917=float(array[1].mean()),
                                                     paired_mean=float(array.mean()),ci95=interval(array))
    paired={}
    for stage in ("ridge","mlp"):
        for baseline,improved in (("A","C"),("C","D")):
            for field in ("r2","js","top1_agreement","gold_rank","gold_margin"):
                array=np.array([[cubes[stage,improved][s][r][field]-cubes[stage,baseline][s][r][field]
                                 for r in identities] for s in range(2)])
                paired[f"{stage}:{baseline}->{improved}:{field}"]=dict(mean=float(array.mean()),ci95=interval(array))
    for (left_stage,left_arm),(right_stage,right_arm) in [
            (("ridge","C_shuffle"),("ridge","C")),
            *(( ("ridge",arm),("mlp",arm)) for arm in ("A","C","D"))]:
        for field in ("r2","js","top1_agreement","gold_rank","gold_margin"):
            array=np.array([[cubes[right_stage,right_arm][s][r][field]-cubes[left_stage,left_arm][s][r][field]
                             for r in identities] for s in range(2)])
            paired[f"{left_stage}:{left_arm}->{right_stage}:{right_arm}:{field}"]=dict(
                mean=float(array.mean()),ci95=interval(array))
    save_json(OUT/"summary.json",dict(rows=rows_out,conditional=conditional,paired=paired,
              pca_coverage={str(s):metrics[s,"ridge"]["pca_variance_coverage"] for s in SEEDS},
              uncertainty="4000 paired crossed checkpoint/reaction bootstrap draws; two checkpoints only",
              mlp_reduction="mean metrics of three independently initialized residual probes; no prediction ensemble"))
    with (OUT/"table.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows_out[0]));writer.writeheader();writer.writerows(rows_out)
    text=["# Native oracle next-token transition decomposition", "",
          "Frozen Native r8 seeds 533/917; product final post-RMSNorm; k=1. "
          "All models and seeds use identical t-3..t+2 eligible rows. "
          "Primary metrics average within reaction, then across reactions. "
          "MLP rows average metrics across probe seeds 20260904/5/6, without ensembling predictions.", "",
          f"Train-target rank-256 PCA coverage: 533={metrics[533,'ridge']['pca_variance_coverage']:.6f}; "
          f"917={metrics[917,'ridge']['pca_variance_coverage']:.6f}. "
          "PCA and input standardizers use only training data. "
          "The original Report-02 .641 is not a matched-row comparator.", "",
          "| Seed | Probe/input | R2 | nMSE | cos / centered | P token\u2223h | P h\u2223token | P history | JS | top-1 | x[t+2] rank | margin |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(rows_out,key=lambda r:((533,917,"paired mean").index(r["seed"]),r["stage"]!="ridge",ARMS.index(r["arm"]))):
        cond=[]
        for field,eligible in (("P_token_given_h","C"),("P_h_given_token","C"),("P_history_given_h_token","D")):
            item=conditional.get(f"{r['stage']}:{field}")
            key="paired_mean" if r["seed"]=="paired mean" else f"seed_{r['seed']}"
            cond.append(f"{item[key]:.3f}" if item and r["arm"]==eligible else "—")
        text.append(f"| {r['seed']} | {r['stage']} {r['arm']} | {r['r2']:.3f} | {r['normalized_mse']:.3f} | "
                    f"{r['cosine']:.3f} / {r['centered_cosine']:.3f} | {' | '.join(cond)} | {r['js']:.3f} | "
                    f"{r['top1_agreement']:.3f} | {r['gold_rank']:.2f} | {r['gold_margin']:.3f} |")
    text += ["", "Paired conditional information (mean; crossed checkpoint/reaction 95% bootstrap CI):", ""]
    for name,v in conditional.items():
        text.append(f"- {name}: {v['paired_mean']:.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}].")
    text += ["", "Paired functional and reconstruction differences (improved minus baseline):", ""]
    for name,v in paired.items():
        text.append(f"- {name}: {v['mean']:.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}].")
    text += ["", "The bootstrap uses 4,000 paired draws, resampling reactions jointly across checkpoints "
             "and checkpoint indices jointly across reactions. Inference across training seeds remains limited "
             "by having only two checkpoints. Conditional quantities use each reaction's SSE ratio first.", "",
             "All KL, JS, top-1/5/10, teacher probability/log-probability/rank/margin, latent cosine, centered "
             "cosine, SSE/SST, and per-reaction results are retained in metrics_*.json and table.csv. "
             "Actual-state decoder references are in the ridge metrics files. Saved BF16 LM-head weights "
             "are used unchanged over all saved outputs (402, including reserved IDs). Unlike "
             "Report 02's decoder tables, there is no native-vocabulary truncation or renormalization.", "",
             "Ridge: AdamW lr=.003, weight_decay=.001; residual MLP: frozen ridge plus "
             "Linear(input,128)-GELU-Linear(128,256), weight_decay=.0001. Both use batch size 512, "
             "at most 20 epochs, validation-only early stopping (patience 3, improvement 1e-8). "
             "Train sampling is reaction-balanced and capped at 16,384 positions. "
             "Initialization is explicitly seeded. No ChemFM weights are trained or tokens generated.", ""]
    interpretation=OUT/"interpretation.md"
    if interpretation.exists():
        text[2:2]=[interpretation.read_text(encoding="utf-8"),""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(text),encoding="utf-8")
    save_json(OUT/"artifact_manifest.json",{
        "source_sha256":sha256_file(Path(__file__)),"split_sha256":sha256_file(SPLIT),
        "base_model_provenance":json.loads((ROOT/"models/ChemFM-1B/PROVENANCE.json").read_text()),
        "files":{
            **{p.name:{"bytes":p.stat().st_size,"sha256":sha256_file(p)}
               for p in sorted(OUT.iterdir()) if p.is_file() and p.name!="artifact_manifest.json" and p.suffix!=".log"},
            "../../docs/reports/03_NATIVE_ORACLE_TRANSITION_DECOMPOSITION.md":{
                "bytes":REPORT.stat().st_size,"sha256":sha256_file(REPORT)}}})
    log(stage="summary_complete",conditional=conditional,paired=paired)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("smoke","extract","probe-smoke","ridge","mlp","summarize"))
    args=p.parse_args()
    torch.set_num_threads(4)
    if args.stage in ("smoke","extract"): extract(args.stage=="smoke")
    elif args.stage=="probe-smoke": probe_smoke()
    elif args.stage=="ridge":
        assert (OUT/"probe_smoke.json").exists(), "run one real probe batch before full fitting"
        check_common()
        for seed in SEEDS: fit(seed,"ridge");evaluate(seed,"ridge")
    elif args.stage=="mlp":
        assert all((OUT/f"metrics_{s}_ridge.json").exists() for s in SEEDS)
        for seed in SEEDS: fit(seed,"mlp");evaluate(seed,"mlp")
    else: summarize()


if __name__=="__main__": main()
