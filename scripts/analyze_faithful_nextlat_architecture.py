"""Correct architecture-level comparison of Native, faithful NextLat, and projected JEPA."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src")); sys.path.insert(0,str(ROOT/"scripts"))
from architecture_analysis import decoder_coordinates, functional_distribution_metrics, orthonormal_decoder_basis, product_transition_rows
from latent_predictability import TargetBasis, Standardizer, RidgeProbe, fit_probe, latent_metrics, reaction_balanced_indices, sha256_file
from run_stp_completion import native_evaluation

OUT=ROOT/"runs/faithful_nextlat/architecture"; CACHE=OUT/"cache"
ARMS=("native","faithful_nextlat","decoder_projected"); SEEDS=(533,917); PROBE_SEED=20260904

def save(path,value): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
def load(path): return torch.load(path,map_location="cpu",weights_only=False)
def key(row): return (row["reaction_identity"],row["absolute_t"])
def key_hash(keys): return hashlib.sha256(json.dumps(keys,separators=(",", ":")).encode()).hexdigest()

def matrices(payload,split,keys,projection,vt):
    rows={key(r):r for r in product_transition_rows(payload["records"]) if r["split"]==split}
    chosen=[rows[tuple(k)] for k in keys]
    h=torch.stack([r["h_t"] for r in chosen]); y=torch.stack([r["h_next"] for r in chosen])
    token=torch.tensor([r["oracle_token"] for r in chosen]); gold=torch.tensor([r["gold_token"] for r in chosen])
    hc=decoder_coordinates(h,projection,vt); yc=decoder_coordinates(y,projection,vt)
    action=F.linear(payload["embedding"][token].float(),projection.float())
    return {"x":torch.cat((hc["functional"],action),1),"h":y,"z":yc["functional"],
            "q":yc["orthonormal"],"null":yc["null"],"gold":gold,
            "ids":[r["reaction_identity"] for r in chosen]}

def fit_target(train,val,test,target,rank=None):
    standardizer=Standardizer.fit(train["x"]); tx=standardizer(train["x"]); vx=standardizer(val["x"]); ex=standardizer(test["x"])
    if rank:
        basis=TargetBasis.fit(train[target].float().cuda(),rank,PROBE_SEED+1)
        ty,vy=basis.encode(train[target]),basis.encode(val[target]); decode=basis.decode; mean=basis.mean
        coverage=basis.variance_coverage
    else:
        mean=train[target].float().mean(0); ty=train[target].float()-mean; vy=val[target].float()-mean
        decode=lambda value:value+mean; coverage=1.0
    torch.manual_seed(PROBE_SEED); probe=RidgeProbe(tx.shape[1],ty.shape[1]).cuda()
    probe,trace=fit_probe(probe,tx,ty,vx,vy,weight_decay=1e-3,epochs=20,batch_size=512,seed=PROBE_SEED)
    pieces=[]
    with torch.inference_mode():
        for begin in range(0,len(ex),512): pieces.append(probe(ex[begin:begin+512].cuda()).cpu())
    pred=decode(torch.cat(pieces)); probe.cpu(); torch.cuda.empty_cache()
    return pred,mean,{"coverage":coverage,"fit":trace}

def reaction_map(target,pred,mean,ids,extra=None):
    result={}
    for identity in sorted(set(ids)):
        ix=torch.tensor([x==identity for x in ids]); m=latent_metrics(target[ix],pred[ix],mean)
        if extra:
            for name,value in extra.items(): m[name]=float(value[ix].float().mean())
        result[identity]=m
    return result

def bootstrap_contrast(left,right,field,seed,n=10000):
    ids=sorted(set(left)&set(right)); delta=np.array([left[i][field]-right[i][field] for i in ids]); rng=np.random.default_rng(seed)
    draws=delta[rng.integers(0,len(delta),(n,len(delta)))].mean(1)
    return {"mean":float(delta.mean()),"median":float(np.median(delta)),"ci95":[float(x) for x in np.quantile(draws,[.025,.975])],"reactions":len(ids)}

def spectrum(z,pred,singular):
    variance=z.var(0,unbiased=False); error=(pred-z).square().mean(0); order=torch.argsort(singular,descending=True)
    deciles=[]
    for chunk in torch.tensor_split(order,10): deciles.append({"sigma_mean":float(singular[chunk].mean()),"target_variance":float(variance[chunk].mean()),"mse":float(error[chunk].mean())})
    delta=z-z.mean(0); eig=torch.linalg.eigvalsh(delta.T@delta/max(1,len(delta))).clamp_min(0)
    return {"per_axis_variance":variance.tolist(),"per_axis_mse":error.tolist(),"singular_deciles":deciles,
            "participation_ratio":float(eig.sum().square()/eig.square().sum().clamp_min(1e-30)),
            "dimensions_for_95pct":int((torch.cumsum(eig.flip(0),0)<.95*eig.sum()).sum()+1)}

def decoder_drift(head,w0,vt):
    wc=head.float()-head.float().mean(0); energy=wc.square().sum(); visible=(wc@vt.T).square().sum()/energy
    _,s,vh=torch.linalg.svd(wc,full_matrices=False); rank=int((s>s[0]*1e-6).sum()); final_vt=vh[:rank]
    cos=torch.linalg.svdvals(vt@final_vt.T).clamp(0,1); angles=torch.rad2deg(torch.acos(cos))
    overlap=cos.square().sum()/len(vt)
    return {"rho":float(visible),"final_rank":rank,"principal_angle_median_deg":float(angles.median()),
            "principal_angle_max_deg":float(angles.max()),"projection_overlap":float(overlap),
            "chordal_distance":float(torch.sqrt((len(vt)-cos.square().sum()).clamp_min(0)))}

def margin_decomposition(native,treatment,test_keys,native_vocab):
    nrows={key(r):r for r in product_transition_rows(native["records"])}; trows={key(r):r for r in product_transition_rows(treatment["records"])}
    hn=torch.stack([nrows[tuple(k)]["h_next"] for k in test_keys]).float(); ht=torch.stack([trows[tuple(k)]["h_next"] for k in test_keys]).float()
    gold=torch.tensor([nrows[tuple(k)]["gold_token"] for k in test_keys])
    wn=native["lm_head"][:native_vocab].float(); wt=treatment["lm_head"][:native_vocab].float()
    logitsn=hn@wn.T; predn=logitsn.argmax(1); competitor=logitsn.clone(); competitor[torch.arange(len(gold)),gold]=-torch.inf; c=competitor.argmax(1)
    dn=wn[gold]-wn[c]; dt=wt[gold]-wt[c]; dh=ht-hn; dd=dt-dn
    values={"representation":(dn*dh).sum(1),"head":(dd*hn).sum(1),"interaction":(dd*dh).sum(1)}
    values["total"]=sum(values.values()); torch.testing.assert_close(values["total"],(dt*ht).sum(1)-(dn*hn).sum(1),atol=2e-4,rtol=2e-4)
    predt=(ht@wt.T).argmax(1); strata=torch.where(predn.ne(gold)&predt.eq(gold),0,torch.where(predn.eq(gold)&predt.ne(gold),1,2))
    names=("fixes_native","breaks_native","unchanged_correctness"); out={}
    for label,name in enumerate(names):
        ix=strata==label; out[name]={"n":int(ix.sum()),**{k:{"mean":float(v[ix].mean()) if ix.any() else None,"median":float(v[ix].median()) if ix.any() else None} for k,v in values.items()}}
    identities=[k[0] for k in test_keys]; rng=np.random.default_rng(PROBE_SEED)
    out["all"]={}
    for name,value in values.items():
        per_reaction=np.array([float(value[torch.tensor([x==identity for x in identities])].mean())
                               for identity in sorted(set(identities))])
        draws=per_reaction[rng.integers(0,len(per_reaction),(10000,len(per_reaction)))].mean(1)
        out["all"][name]={"mean":float(value.mean()),"median":float(value.median()),
                           "reaction_balanced_mean":float(per_reaction.mean()),
                           "reaction_bootstrap_ci95":[float(x) for x in np.quantile(draws,[.025,.975])]}
    return out

def generation(seed,arm):
    path=(native_evaluation(8,seed)/"predictions.jsonl" if arm=="native" else
          ROOT/f"runs/faithful_nextlat/seed_{seed}/evaluation/predictions.jsonl" if arm=="faithful_nextlat" else
          ROOT/f"runs/decoder_projected/seed_{seed}/evaluation/predictions.jsonl")
    rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]; n=len(rows)
    result={f"top{k}":sum(r["target"] in r["ranked_candidates"][:k] for r in rows)/n for k in (1,3,5,10)}
    candidates=[c for r in rows for view in r["canonical_candidates_by_view"] for c in view]
    result.update(view_candidate_validity=sum(bool(c) for c in candidates)/len(candidates),
                  ranked_validity=sum(bool(c) for r in rows for c in r["ranked_candidates"][:10])/(10*n),
                  predictions=str(path),predictions_sha256=sha256_file(path)); return result

def run_seed(seed):
    payloads={a:load(CACHE/f"{a}_{seed}.pt") for a in ARMS}
    aux=load(ROOT/f"runs/decoder_projected/seed_{seed}/training/checkpoints/epoch_4/auxiliary_training_state.pt")
    state=aux["state_dict"]; projection=state["projection"].float(); u=state["u"].float(); w0=state["frozen_decoder"].float()
    singular=torch.tensor(aux["metadata"]["singular_values"][:projection.shape[0]])
    vt=orthonormal_decoder_basis(projection,singular)
    allrows={a:product_transition_rows(payloads[a]["records"]) for a in ARMS}
    keysets={a:{key(r) for r in allrows[a]} for a in ARMS}; assert len({frozenset(x) for x in keysets.values()})==1
    keys={}
    for split in ("train","validation","test"):
        base=[r for r in allrows["native"] if r["split"]==split]; ordered=[key(r) for r in base]
        if split=="train": ordered=[ordered[i] for i in reaction_balanced_indices(base,16384,PROBE_SEED).tolist()]
        keys[split]=ordered
    data={a:{s:matrices(payloads[a],s,keys[s],projection,vt) for s in keys} for a in ARMS}
    result={"seed":seed,"coordinate_basis":{"auxiliary_state":str((ROOT/f'runs/decoder_projected/seed_{seed}/training/checkpoints/epoch_4/auxiliary_training_state.pt').resolve()),
        "auxiliary_sha256":sha256_file(ROOT/f"runs/decoder_projected/seed_{seed}/training/checkpoints/epoch_4/auxiliary_training_state.pt"),
        "rank":len(singular),"selected_train_key_hash":key_hash(keys["train"]),"test_key_hash":key_hash(keys["test"])},"arms":{},"contrasts":{}}
    for arm in ARMS:
        d=data[arm]
        armout={"targets":{},
                "decoder_drift":decoder_drift(payloads[arm]["lm_head"][:w0.shape[0]],w0,vt),
                "generation":generation(seed,arm)}
        for target,rank in (("h",256),("z",None),("q",None),("null",128)):
            pred,mean,fit=fit_target(d["train"],d["validation"],d["test"],target,rank)
            metrics=latent_metrics(d["test"][target],pred,mean,d["test"]["ids"]); extra=None
            if target=="z":
                extra=functional_distribution_metrics(d["test"]["z"]@u.T,pred@u.T,d["test"]["gold"])
                metrics.update({k:float(v.float().mean()) for k,v in extra.items()})
                armout["spectrum"]=spectrum(d["test"]["z"],pred,singular)
            armout["targets"][target]={"metrics":metrics,"probe":fit,
                "reaction_metrics":reaction_map(d["test"][target],pred,mean,d["test"]["ids"],extra)}
        delta=d["test"]["h"]-torch.stack([r["h_t"] for r in [
            {key(x):x for x in allrows[arm]}[tuple(k)] for k in keys["test"]]])
        dc=decoder_coordinates(delta,projection,vt)
        armout["transition_magnitude"]={"parallel_rms":float(dc["parallel"].square().sum(1).sqrt().mean()),
                                         "null_rms":float(dc["null"].square().sum(1).sqrt().mean())}
        result["arms"][arm]=armout
    for left,right in (("faithful_nextlat","native"),("decoder_projected","native"),("decoder_projected","faithful_nextlat")):
        label=f"{left}-minus-{right}"; result["contrasts"][label]={}
        for target in ("h","z","q","null"):
            l=result["arms"][left]["targets"][target]["reaction_metrics"]; r=result["arms"][right]["targets"][target]["reaction_metrics"]
            result["contrasts"][label][target]={field:bootstrap_contrast(l,r,field,seed+len(field)) for field in ("r2","normalized_mse","cosine")}
        native_error=torch.tensor(result["arms"][right]["spectrum"]["per_axis_mse"])
        treatment_error=torch.tensor(result["arms"][left]["spectrum"]["per_axis_mse"])
        improvement=native_error-treatment_error; order=torch.argsort(singular,descending=True)
        deciles=[float(improvement[c].mean()) for c in torch.tensor_split(order,10)]
        result["contrasts"][label]["functional_spectrum"]={
            "error_improvement_by_descending_sigma_decile":deciles,
            "high_sigma_minus_low_sigma_improvement":deciles[0]-deciles[-1],
            "sigma_improvement_correlation":float(np.corrcoef(singular.numpy(),improvement.numpy())[0,1])}
    result["margin_decomposition"]={
        a:margin_decomposition(payloads["native"],payloads[a],keys["test"],w0.shape[0])
        for a in ARMS[1:]}
    return result

def correct_head_summaries():
    """Correct head-dependent summaries without rerunning the fitted probes."""
    path=OUT/"results.json"; outputs=json.loads(path.read_text())
    for seed in SEEDS:
        payloads={a:load(CACHE/f"{a}_{seed}.pt") for a in ARMS}
        aux=load(ROOT/f"runs/decoder_projected/seed_{seed}/training/checkpoints/epoch_4/auxiliary_training_state.pt")
        state=aux["state_dict"]; projection=state["projection"].float(); w0=state["frozen_decoder"].float()
        singular=torch.tensor(aux["metadata"]["singular_values"][:projection.shape[0]])
        vt=orthonormal_decoder_basis(projection,singular)
        test_keys=[key(r) for r in product_transition_rows(payloads["native"]["records"]) if r["split"]=="test"]
        result=outputs[str(seed)]
        for arm in ARMS:
            result["arms"][arm]["decoder_drift"]=decoder_drift(payloads[arm]["lm_head"][:w0.shape[0]],w0,vt)
        result["margin_decomposition"]={
            arm:margin_decomposition(payloads["native"],payloads[arm],test_keys,w0.shape[0])
            for arm in ARMS[1:]}
    save(path,outputs)
    save(OUT/"complete.json",{"status":"complete","seeds":list(SEEDS),"results_sha256":sha256_file(path),
                              "head_summaries_native_vocab_corrected":True})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--head-only",action="store_true")
    args=parser.parse_args()
    if args.head_only:
        correct_head_summaries(); return
    outputs={str(seed):run_seed(seed) for seed in SEEDS}; save(OUT/"results.json",outputs)
    save(OUT/"complete.json",{"status":"complete","seeds":list(SEEDS),"results_sha256":sha256_file(OUT/"results.json")})
if __name__=="__main__": main()
