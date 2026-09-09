"""Compact paired development summary for Report 07."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from run_stp_completion import native_evaluation

def read(path): return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def path(seed,arm):
    if arm=="native": return native_evaluation(8,seed)/"predictions.jsonl"
    if arm=="decoder_projected": return ROOT/f"runs/decoder_projected/seed_{seed}/evaluation/predictions.jsonl"
    return ROOT/f"runs/faithful_nextlat/seed_{seed}/evaluation/predictions.jsonl"
def correct(row,k=1): return row["target"] in row["ranked_candidates"][:k]
def pair(left,right,seed):
    l=np.array([correct(x) for x in left]); r=np.array([correct(x) for x in right]); delta=r.astype(int)-l.astype(int)
    only_r=int(((~l)&r).sum()); only_l=int((l&(~r)).sum()); rng=np.random.default_rng(seed)
    boot=delta[rng.integers(0,len(delta),(100000,len(delta)))].mean(1)
    lr=[]
    for a,b in zip(left,right):
        ra=(a["ranked_candidates"].index(a["target"])+1 if a["target"] in a["ranked_candidates"] else 11)
        rb=(b["ranked_candidates"].index(b["target"])+1 if b["target"] in b["ranked_candidates"] else 11); lr.append((ra,rb))
    return {"right_minus_left_top1":float(delta.mean()),"right_only":only_r,"left_only":only_l,
            "ties":len(delta)-only_r-only_l,"mcnemar_p":float(binomtest(min(only_r,only_l),only_r+only_l,.5).pvalue),
            "bootstrap_ci95":[float(x) for x in np.quantile(boot,[.025,.975])],
            "gold_rank":{"right_better":sum(b<a for a,b in lr),"left_better":sum(a<b for a,b in lr),"ties":sum(a==b for a,b in lr)}}
def main():
    arms=("native","faithful_nextlat","decoder_projected"); out={}
    for seed in (533,917):
        rows={a:read(path(seed,a)) for a in arms}; ids=[[r["reaction_identity"] for r in rows[a]] for a in arms]; assert ids[0]==ids[1]==ids[2]
        out[str(seed)]={"arms":{},"pairs":{}}
        for arm in arms:
            r=rows[arm]; candidates=[c for x in r for v in x["canonical_candidates_by_view"] for c in v]
            out[str(seed)]["arms"][arm]={**{f"top{k}":sum(correct(x,k) for x in r)/len(r) for k in (1,3,5,10)},
                "view_candidate_validity":sum(bool(c) for c in candidates)/len(candidates),
                "ranked_validity":sum(bool(c) for x in r for c in x["ranked_candidates"][:10])/(10*len(r))}
        for left,right in (("native","faithful_nextlat"),("native","decoder_projected"),("faithful_nextlat","decoder_projected")):
            out[str(seed)]["pairs"][f"{right}-minus-{left}"]=pair(rows[left],rows[right],seed+len(left))
    out["mean_effects"]={arm:float(np.mean([out[str(s)]["arms"][arm]["top1"]-out[str(s)]["arms"]["native"]["top1"] for s in (533,917)]))
                         for arm in ("faithful_nextlat","decoder_projected")}
    destination=ROOT/"runs/faithful_nextlat/behavioral_summary.json"; destination.write_text(json.dumps(out,indent=2)+"\n")
if __name__=="__main__": main()
