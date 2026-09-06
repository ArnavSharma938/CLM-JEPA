"""Amend Report 03 from saved oracle artifacts; no ChemFM forward is available here."""
import argparse
import json
import time

import numpy as np
import torch

import run_oracle_transition as oracle
from latent_predictability import assert_disjoint_confirmation, shuffled_reaction_targets

COMPACT = ("summary.json", "eligibility.json", "shuffle_diagnostics_533.json",
           "shuffle_diagnostics_917.json", "table.csv", "pca_ceiling.json",
           "native_vocab_decoder_summary.json", "nonlinear_c_shuffle_summary.json")


def read(name):
    return json.loads((oracle.OUT/name).read_text(encoding="utf-8"))


def projection_ceiling(target, basis, identities):
    """Orthogonal projection is the SSE optimum within the fixed affine PCA space."""
    target=target.float()
    prediction=basis.decode(basis.encode(target))
    orthogonality_error=float((basis.components@basis.components.T-torch.eye(len(basis.components))).abs().max())
    # Saved FP32 GPU PCA vectors have small numerical orthogonality error.
    # Preserve the fitted basis and measure the requested encode/decode operation.
    assert orthogonality_error<1e-3
    residual=(target-prediction).square().sum(-1)
    total=(target-basis.mean).square().sum(-1)
    reactions={}
    for identity in sorted(set(identities)):
        selected=torch.tensor([r==identity for r in identities])
        sse=float(residual[selected].sum()); sst=float(total[selected].sum())
        reactions[identity]=dict(n=int(selected.sum()),sse=sse,sst=sst,normalized_mse=sse/sst,r2=1-sse/sst)
    return dict(reaction_mean_r2=float(np.mean([r["r2"] for r in reactions.values()])),
                reaction_mean_normalized_mse=float(np.mean([r["normalized_mse"] for r in reactions.values()])),
                token_weighted_r2=float(1-residual.sum()/total.sum()),
                token_weighted_normalized_mse=float(residual.sum()/total.sum()),
                max_basis_orthogonality_error=orthogonality_error,reactions=reactions)


def diagnostics(data):
    _,donors=shuffled_reaction_targets(data["token"],data["metadata"],oracle.PROBE_SEED+1)
    pairs=[(data["metadata"][i],data["metadata"][j]) for i,j in enumerate(donors.tolist())]
    result=dict(self_reaction_matches=sum(a["reaction_identity"]==b["reaction_identity"] for a,b in pairs),
                same_token_fraction=float(data["token"].eq(data["token"][donors]).float().mean()),
                mean_absolute_product_position_gap=float(np.mean([abs(a["future_index"]-b["future_index"]) for a,b in pairs])),
                mean_absolute_product_length_gap=float(np.mean([abs(a["sequence_length"]-b["sequence_length"]) for a,b in pairs])))
    assert result["self_reaction_matches"]==0
    return result


def preflight_and_ceilings():
    # Preserve pre-amendment input hashes; existing model/probe files must never change.
    prior=read("artifact_manifest.json")
    inputs=(read("amendment_inputs.json") if (oracle.OUT/"amendment_inputs.json").exists()
            else prior.get("immutable_original_inputs"))
    if inputs is None:
        inputs={name:record for name,record in prior["files"].items()
                if name.endswith(".pt") or (name.startswith("metrics_") and "native" not in name)}
    for name,record in inputs.items():
        assert oracle.sha256_file(oracle.OUT/name)==record["sha256"],name
    if not (oracle.OUT/"amendment_inputs.json").exists():
        oracle.save_json(oracle.OUT/"amendment_inputs.json",inputs)
    oracle.check_common()
    split=json.loads(oracle.SPLIT.read_text())
    exclusion=assert_disjoint_confirmation([r["chemical_pair_id"] for r in split["records"]],
        oracle.ROOT/"data/clm_jepa_uspto_mit_stp_confirmation/untouched_640.jsonl")
    eligibility=read("eligibility.json")
    eligibility.update(confirmation_overlap=exclusion["chemical_pair_overlap"],confirmation_exclusion=exclusion,
                       sampled_train_positions=16384,common_rows_across_all_arms_and_seeds=True)
    oracle.save_json(oracle.OUT/"eligibility.json",eligibility)
    assignment={r["reaction_identity"]:r["split"] for r in split["records"]}
    ceilings={}
    for seed in oracle.SEEDS:
        payload=oracle.load(oracle.OUT/f"cache_native_{seed}.pt")
        prep=oracle.load(oracle.OUT/f"preprocessing_{seed}.pt")
        assert all(assignment[r["reaction_identity"]]==r["split"] for r in payload["records"])
        test=oracle.rows(payload,"test")
        ceilings[str(seed)]=projection_ceiling(test["y"],prep["basis"],[r["reaction_identity"] for r in test["metadata"]])
        ceilings[str(seed)]["train_pca_coverage"]=prep["basis"].variance_coverage
        diagnostic=read(f"shuffle_diagnostics_{seed}.json")
        diagnostic["test"]=diagnostics(test)
        diagnostic["matching"]="cross-reaction derangement; approximate product-position/length matching, not exact"
        oracle.save_json(oracle.OUT/f"shuffle_diagnostics_{seed}.json",diagnostic)
        # One real TRAIN batch, using the saved shuffle scaler and ridge; no test selection.
        train=oracle.rows(payload,"train")
        chosen=prep["chosen"]
        train={k:([v[i] for i in chosen.tolist()] if k=="metadata" else v[chosen]) for k,v in train.items()}
        assert diagnostics(train)=={k:v for k,v in diagnostic["train"].items()}
        x=oracle.inputs(train,payload["embedding"],"C_shuffle")
        artifact=oracle.load(oracle.OUT/f"probe_{seed}_C_shuffle_ridge.pt")
        fitted=oracle.Standardizer.fit(x)
        assert torch.equal(fitted.mean,artifact["standardizer"].mean)
        assert torch.equal(fitted.scale,artifact["standardizer"].scale)
        x=artifact["standardizer"](x[:512]).cuda()
        ridge=oracle.RidgeProbe(4096,256).cuda();ridge.load_state_dict(artifact["state"])
        probe=oracle.ResidualMLPProbe(ridge,4096,256,128).cuda()
        prediction=probe(x)
        loss=torch.nn.functional.mse_loss(prediction,prep["basis"].encode(train["y"][:512]).cuda())
        loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None for p in probe.ridge.parameters())
        with torch.no_grad():
            head=oracle.decoder_weight(payload,True).cuda()
            reconstructed=prep["basis"].decode(prediction.detach().cpu()).to(device="cuda",dtype=head.dtype)
            assert (reconstructed@head.T).shape==(512,392)
        del probe,ridge,prediction,x,loss
        oracle.log(stage="amendment_smoke_passed",seed=seed,test_pca_r2=ceilings[str(seed)]["reaction_mean_r2"])
    oracle.save_json(oracle.OUT/"pca_ceiling.json",dict(
        definition="decode(encode(true test h[t+1])); exact fixed-basis SSE ceiling, no fitting on test",
        normalization="SST around fitted train-target mean; primary equal reaction weights",
        seeds=ceilings,paired_mean={field:float(np.mean([r[field] for r in ceilings.values()]))
                                  for field in ("reaction_mean_r2","reaction_mean_normalized_mse")}))


def supplements():
    oracle.summarize(native_vocab=True)
    summary=read("summary.json")
    ceiling=read("pca_ceiling.json")
    summary["test_pca_ceiling"]={s:{k:v for k,v in r.items() if k!="reactions"} for s,r in ceiling["seeds"].items()}
    summary["eligibility"]=read("eligibility.json")
    oracle.save_json(oracle.OUT/"summary.json",summary)
    native_rows=[{k:v for k,v in r.items() if k in ("seed","stage","arm",*oracle.DECODER_FIELDS)}
                 for r in summary["rows"]]
    full_rows=[];max_discrepancy={k:0.0 for k in oracle.DECODER_FIELDS}
    original_latent_max_difference=0.0
    for seed in oracle.SEEDS:
        for stage in ("ridge","mlp"):
            original=read(f"metrics_{seed}_{stage}.json")
            native=read(f"metrics_{seed}_{stage}_native.json")
            assert original["row_keys"]==native["row_keys"]
            for key,values in original["results"].items():
                for reaction,old in values.items():
                    for field in ("r2","normalized_mse","cosine","centered_cosine"):
                        original_latent_max_difference=max(original_latent_max_difference,
                            abs(old[field]-native["results"][key][reaction][field]))
            for arm in oracle.ARMS if stage=="ridge" else ("A","C","D"):
                samples=[v for k,v in original["results"].items() if k.split(":")[0]==arm]
                row=dict(seed=seed,stage=stage,arm=arm,**{
                    field:float(np.mean([np.mean([r[field] for r in v.values()]) for v in samples]))
                    for field in oracle.DECODER_FIELDS})
                full_rows.append(row)
                n=next(r for r in native_rows if (r["seed"],r["stage"],r["arm"])==(seed,stage,arm))
                for field in oracle.DECODER_FIELDS:max_discrepancy[field]=max(max_discrepancy[field],abs(row[field]-n[field]))
    assert original_latent_max_difference<1e-6,original_latent_max_difference
    oracle.save_json(oracle.OUT/"native_vocab_decoder_summary.json",dict(
        primary_vocabulary_size=392,robustness_vocabulary_size=402,primary=native_rows,
        actual_state_reference={str(seed):{
            field:float(np.mean([r[field] for r in read(f"metrics_{seed}_ridge_native.json")["true_metrics"].values()]))
            for field in oracle.DECODER_FIELDS} for seed in oracle.SEEDS},
        full_head_robustness_only=full_rows,max_absolute_native_vs_full_cell_mean_difference=max_discrepancy,
        max_original_latent_metric_change=original_latent_max_difference,
        discrepancy="Original Report 03 included 10 added predictor/reserved outputs; amended primary uses head[:392] like Report 02."))
    # Keep paired reaction-level differences as compact arrays for independent verification.
    reaction_ids=None;differences={}
    for seed in oracle.SEEDS:
        results=read(f"metrics_{seed}_mlp_native.json")["results"]
        ids=sorted(results[f"C:{oracle.PROBE_SEED}"])
        assert reaction_ids is None or ids==reaction_ids
        reaction_ids=ids
        differences[str(seed)]={}
        for field in ("r2","normalized_mse",*oracle.DECODER_FIELDS):
            differences[str(seed)][field]=[float(np.mean([
                results[f"C:{p}"][r][field]-results[f"C_shuffle:{p}"][r][field]
                for p in range(oracle.PROBE_SEED,oracle.PROBE_SEED+3)])) for r in ids]
    effects={field:{**summary["paired"][f"mlp:C_shuffle->mlp:C:{field}"],
                    "seed_533":float(np.mean(differences["533"][field])),
                    "seed_917":float(np.mean(differences["917"][field]))}
             for field in ("r2","normalized_mse",*oracle.DECODER_FIELDS)}
    oracle.save_json(oracle.OUT/"nonlinear_c_shuffle_summary.json",dict(
        contrast="C_MLP minus C_shuffle_MLP",effects=effects,reaction_ids=reaction_ids,
        paired_reaction_differences=differences,probe_seeds=list(range(oracle.PROBE_SEED,oracle.PROBE_SEED+3)),
        protocol="same saved train PCA and shuffle ridge/standardizer; width 128; lr .003; decay .0001; batch 512; 20 epochs; patience 3",
        rows=[r for r in summary["rows"] if r["stage"]=="mlp" and r["arm"] in ("C","C_shuffle")]))
    text=["", "## Amendment checks", "", "The test projection ceiling uses the true held-out target projected "
          "through the saved train-only PCA. It is the maximum full-state R2 attainable within that fixed affine "
          "target space, under the same reaction-balanced SSE/SST reduction; no PCA is fitted on test.", "",
          "| Native seed | Test PCA ceiling R2 | Test projection nMSE |", "|---|---:|---:|"]
    for seed in (*map(str,oracle.SEEDS),"paired mean"):
        v=ceiling["paired_mean"] if seed=="paired mean" else ceiling["seeds"][seed]
        text.append(f"| {seed} | {v['reaction_mean_r2']:.6f} | {v['reaction_mean_normalized_mse']:.6f} |")
    e=read("eligibility.json")
    text += ["", f"Eligible train/validation/test reactions: {e['eligible_reaction_split']['train']}/"
             f"{e['eligible_reaction_split']['validation']}/{e['eligible_reaction_split']['test']}; "
             f"positions: {e['rows']['train']:,}/{e['rows']['validation']:,}/{e['rows']['test']:,}. "
             "Training uses the same 16,384 reaction-balanced positions in all arms and seeds. "
             "Confirmation chemical-pair overlap is zero. Cached input IDs, product positions, split assignments "
             "and all evaluated row identities match across checkpoints and arms.", "",
             "The token shuffle is cross-reaction and approximately product-position/length matched, not exact. "
             "Diagnostics below use the actual capped training set and complete validation/test sets.", "",
             "| Seed | Split | Self-reaction matches | Same-token fraction | Mean absolute position gap | Mean absolute length gap |",
             "|---|---|---:|---:|---:|---:|"]
    for seed in oracle.SEEDS:
        d=read(f"shuffle_diagnostics_{seed}.json")
        for split in ("train","validation","test"):
            v=d[split]
            text.append(f"| {seed} | {split} | {v['self_reaction_matches']} | {v['same_token_fraction']:.4f} | "
                        f"{v['mean_absolute_product_position_gap']:.3f} | {v['mean_absolute_product_length_gap']:.3f} |")
    text += ["", "The capacity-matched nonlinear correct-minus-shuffled contrast is:", "",
             "| Metric | Seed 533 | Seed 917 | Paired mean | Paired 95% CI |", "|---|---:|---:|---:|---|"]
    for field,v in effects.items():
        text.append(f"| {field} | {v['seed_533']:.6f} | {v['seed_917']:.6f} | {v['mean']:.6f} | "
                    f"[{v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}] |")
    text += ["", "The prior full-head decoder results remain robustness-only in "
             "`native_vocab_decoder_summary.json`; the main table and all primary comparisons now use 392 tokens. "
             f"The maximum absolute native-versus-full cell-mean discrepancy is {max_discrepancy['js']:.8g} "
             f"for JS and {max_discrepancy['top1_agreement']:.8g} for top-1. "
             "All nine decoder metrics and both vocabulary summaries are retained, not just JS/top-1.", "",
             "Compact committed audit artifacts: [summary](../../runs/oracle_transition/summary.json), "
             "[table](../../runs/oracle_transition/table.csv), "
             "[eligibility](../../runs/oracle_transition/eligibility.json), "
             "[PCA ceiling](../../runs/oracle_transition/pca_ceiling.json), "
             "[native/full decoder summaries](../../runs/oracle_transition/native_vocab_decoder_summary.json), "
             "[nonlinear shuffle contrasts](../../runs/oracle_transition/nonlinear_c_shuffle_summary.json), "
             "[seed 533 shuffle](../../runs/oracle_transition/shuffle_diagnostics_533.json), "
             "[seed 917 shuffle](../../runs/oracle_transition/shuffle_diagnostics_917.json), and "
             "[hash manifest](../../runs/oracle_transition/artifact_manifest.json). "
             "Hidden-state caches, fitted weights and full per-reaction metric files remain local and are not committed.", ""]
    oracle.REPORT.write_text(oracle.REPORT.read_text(encoding="utf-8")+"\n".join(text),encoding="utf-8")
    inputs=read("amendment_inputs.json")
    for name,record in inputs.items():assert oracle.sha256_file(oracle.OUT/name)==record["sha256"],name
    new_probes=list(oracle.OUT.glob("probe_*_C_shuffle_mlp_*.pt"))
    assert len(new_probes)==6
    for p in new_probes:
        seed=int(p.name.split("_")[1]);base=oracle.load(oracle.OUT/f"probe_{seed}_C_shuffle_ridge.pt")["state"]
        trained=oracle.load(p)["state"]
        assert all(torch.equal(v,trained["ridge."+k]) for k,v in base.items())
    oracle.save_json(oracle.OUT/"artifact_manifest.json",dict(
        analysis_sources={p.relative_to(oracle.ROOT).as_posix():oracle.sha256_file(p)
                          for p in (oracle.ROOT/"scripts/run_oracle_transition.py",oracle.ROOT/"scripts/amend_oracle_transition.py")},
        immutable_original_inputs=inputs,additional_local_probes={p.name:oracle.sha256_file(p) for p in new_probes},
        amended_local_metrics={p.name:oracle.sha256_file(p) for p in oracle.OUT.glob("metrics_*_native.json")},
        compact_files={name:dict(bytes=(oracle.OUT/name).stat().st_size,sha256=oracle.sha256_file(oracle.OUT/name)) for name in COMPACT},
        report=dict(path=oracle.REPORT.relative_to(oracle.ROOT).as_posix(),sha256=oracle.sha256_file(oracle.REPORT)),
        verification=dict(original_inputs_unchanged=True,new_frozen_ridges_verified=6,
                          no_chemfm_forward=True,no_generation=True,max_original_latent_metric_change=original_latent_max_difference)))
    oracle.log(stage="amendment_complete",pca_ceiling=ceiling["paired_mean"],nonlinear_control=effects)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=("compute","summarize"));args=p.parse_args()
    torch.set_num_threads(4)
    if args.stage=="compute":
        start=time.perf_counter()
        preflight_and_ceilings()
        for seed in oracle.SEEDS:
            oracle.fit(seed,"mlp",arms=("C_shuffle",))
            for stage in ("ridge","mlp"):oracle.evaluate(seed,stage,native_vocab=True)
        oracle.log(stage="amendment_compute_complete",seconds=time.perf_counter()-start)
    else:supplements()


if __name__=="__main__":main()
