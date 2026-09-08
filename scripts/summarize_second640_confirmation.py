"""Report the second-640 endpoints only after all four provenance checks pass."""
import json
import math
from collections import Counter
import numpy as np
from run_second640_confirmation import ROOT, OUT, CELLS, EXPECTED, verify_inputs, verify_cell, jsonl, write, sha

def mcnemar(wins, losses):
    n = wins + losses
    return min(1.0, 2 * sum(math.comb(n,k) for k in range(min(wins,losses)+1)) / 2**n) if n else 1.0

def sign_flip(d):
    # One sign per reaction shared across both seeds; exact integer convolution.
    weights = np.asarray(d).sum(axis=0).astype(int)
    distribution = Counter({0:1})
    nonzero = [abs(int(w)) for w in weights if w]
    for w in nonzero:
        following = Counter()
        for total, count in distribution.items():
            following[total+w] += count
            following[total-w] += count
        distribution = following
    observed = abs(int(weights.sum()))
    p = sum(n for total,n in distribution.items() if abs(total)>=observed) / 2**len(nonzero)
    return {'two_sided_p':p,'observed_signed_sum':int(weights.sum()),'nonzero_reaction_clusters':len(nonzero),
            'method':'Exact Rademacher sign flip of each reaction cluster, common sign across seeds; absolute summed difference statistic.'}

def bootstrap(d, seed, crossed=False):
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(20):
        if crossed:
            s = rng.integers(0,2,size=(1000,2))
            r = rng.integers(0,640,size=(1000,640))
            samples.extend(d[s[:,:,None],r[:,None,:]].mean(axis=(1,2))*100)
        else:
            r = rng.integers(0,640,size=(1000,640))
            samples.extend(d[r].mean(axis=1)*100)
    return np.quantile(samples,[.025,.975]).tolist()

def stats(rows):
    for r in rows:
        views = r['canonical_candidates_by_view']
        assert len(views)==5 and all(len(v)==10 for v in views)
        assert len(r['ranked_candidates'])==10
        # Independently reproduce frozen score.py aggregation from canonical views.
        scores = {}
        for view in views:
            for rank,c in enumerate(dict.fromkeys(c for c in view if c)):
                scores[c] = scores.get(c,0) + 1/(rank+1)
        ranked = sorted(scores,key=scores.get,reverse=True)[:10]
        ranked += ['']*(10-len(ranked))
        assert ranked == r['ranked_candidates']
        assert [c==r['target'] for c in ranked] == r['exact']
    counts = {str(k):sum(r['target'] in r['ranked_candidates'][:k] for r in rows) for k in (1,3,5,10)}
    vv = sum(bool(c) for r in rows for v in r['canonical_candidates_by_view'] for c in v)
    rv = sum(bool(c) for r in rows for c in r['ranked_candidates'])
    return {'top_counts':counts,'top_percent':{k:100*v/640 for k,v in counts.items()},
            'view_valid_count':vv,'view_total':32000,'view_validity_percent':100*vv/32000,
            'ranked_valid_count':rv,'ranked_total':6400,'ranked_validity_percent':100*rv/6400}

def main():
    verify_inputs()
    provenance = [verify_cell(s,a) for s,a in CELLS]
    write(OUT/'all_evaluations_verified.json',provenance)
    # No endpoints are read until every cell has passed the above gates.
    result = {'panel_sha256':EXPECTED,'provenance':provenance,'seeds':{},'bootstrap_repetitions':20000,
              'paired_bootstrap_seed':'corresponding model seed','crossed_bootstrap_seed':20273163,
              'crossed_method':'Resample two seed indices with replacement and 640 reaction indices with replacement, independently; shared reaction indices across sampled seeds.',
              'earlier_report05_first640_is_confirmation':False}
    differences = []
    for seed in (2027,3163):
        rows = {a:jsonl(OUT/f'seed_{seed}'/a/'evaluation/predictions.jsonl') for a in ('native','decoder_projected')}
        n = np.array([r['target']==r['ranked_candidates'][0] for r in rows['native']],dtype=int)
        p = np.array([r['target']==r['ranked_candidates'][0] for r in rows['decoder_projected']],dtype=int)
        d = p-n
        differences.append(d)
        wins,losses = int((d==1).sum()),int((d==-1).sum())
        result['seeds'][str(seed)] = {a:stats(r) for a,r in rows.items()}
        result['seeds'][str(seed)]['paired'] = {'projected_only':wins,'native_only':losses,'ties':int((d==0).sum()),
            'both_correct':int(((n==1)&(p==1)).sum()),'neither_correct':int(((n==0)&(p==0)).sum()),
            'effect_pp':float(100*d.mean()),'exact_mcnemar_two_sided_p':mcnemar(wins,losses),
            'paired_reaction_bootstrap_95_ci_pp':bootstrap(d,seed)}
    d = np.stack(differences)
    effects = d.mean(axis=1)
    result['across_seeds'] = {'mean_effect_pp':float(100*d.mean()),'pooled_projected_only_descriptive':int((d==1).sum()),
        'pooled_native_only_descriptive':int((d==-1).sum()),'crossed_seed_reaction_bootstrap_95_ci_pp':bootstrap(d,20273163,True),
        'reaction_cluster_sign_flip':sign_flip(d)}
    result['decision'] = 'PROCEED' if all(effects>0) else 'STOP' if all(effects<=0) else 'SEED_4211_REQUIRED'
    write(OUT/'confirmation_results.json',result)
    render_report(result)
    print(json.dumps(result,indent=2))

def render_report(result):
    lines = ['# Actual second-640 frozen-checkpoint confirmation', '',
        'This is the second-640 evaluation of the existing epoch-4 Native and decoder-projected checkpoints for seeds 2027/3163. No retraining was performed.', '',
        '**The earlier Report-05 first-640 result is not the confirmation result.** Its attribution to the second 640 was incorrect. Only the newly verified second-640 results below enter the frozen decision rule.', '',
        '## Panel and endpoint', '',
        '`data/clm_jepa_uspto_mit_stp_confirmation/untouched_second_640.jsonl` is the byte-preserving `rows[640:1280]` slice of `untouched_1280.jsonl`.', '',
        f'SHA-256: `{EXPECTED}`. Verified 640 unique reactions, zero first-640 overlap, exact identity order and original panel indices 640–1279. All four prediction files match the panel identities, official group indices, source views, targets, and ordering.', '',
        'The unchanged official evaluator uses five views, beam 10, ten candidates/view, canonical chemical identity, and reciprocal-rank aggregation. Aggregate exact top-1 is primary. The original optional exact execution paths and model configuration were preserved.', '',
        '## Results', '',
        '| Seed | Arm | Top-1 | Top-3 | Top-5 | Top-10 | View validity | Ranked validity |',
        '|---|---|---|---|---|---|---|---|']
    for seed in ('2027','3163'):
        for arm in ('native','decoder_projected'):
            s=result['seeds'][seed][arm]
            values=[f"{s['top_counts'][str(k)]}/640 ({s['top_percent'][str(k)]:.5f}%)" for k in (1,3,5,10)]
            lines.append('| '+' | '.join([seed,arm,*values,f"{s['view_validity_percent']:.6f}%",f"{s['ranked_validity_percent']:.6f}%"])+' |')
    lines += ['', '| Seed | Effect (pp) | Projected-only | Native-only | Ties | Exact McNemar p | Paired bootstrap 95% CI (pp) |',
        '|---|---|---|---|---|---|---|']
    for seed in ('2027','3163'):
        s=result['seeds'][seed]['paired']; ci=s['paired_reaction_bootstrap_95_ci_pp']
        lines.append(f"| {seed} | {s['effect_pp']:.5f} | {s['projected_only']} | {s['native_only']} | {s['ties']} | {s['exact_mcnemar_two_sided_p']:.8g} | [{ci[0]:.5f}, {ci[1]:.5f}] |")
    a=result['across_seeds']; ci=a['crossed_seed_reaction_bootstrap_95_ci_pp']
    lines += ['',f"Mean top-1 effect: **{a['mean_effect_pp']:.6f} pp**. Pooled discordances (descriptive): {a['pooled_projected_only_descriptive']} projected-only / {a['pooled_native_only_descriptive']} Native-only.", '',
        f"Crossed seed/reaction bootstrap 95% CI: **[{ci[0]:.6f}, {ci[1]:.6f}] pp**. Exact reaction-cluster sign-flip two-sided p: **{a['reaction_cluster_sign_flip']['two_sided_p']:.8g}**.", '',
        'Bootstrap intervals use 20,000 percentile resamples. Per-seed reaction bootstrap RNG seeds are 2027 and 3163. Crossed bootstrap RNG seed is 20273163: sample two seed indices and 640 reaction indices independently with replacement; apply the same sampled reaction indices across both sampled seeds. Sign flips use one common sign per reaction across seeds and exact integer convolution, comparing absolute summed differences. Pooled discordances are not treated as independent observations.', '',
        '## Frozen decision', '',f"**{result['decision']}**", '',
        'Rule: both effects >0 → PROCEED; both effects ≤0 → STOP; otherwise → SEED_4211_REQUIRED. No expansion seeds were launched.', '',
        '## Provenance', '',
        'All cells passed checkpoint and manifest hash verification before interpretation. Full file hashes are frozen in `runs/decoder_projected/confirmation_second640/frozen_inputs.json`; exact launch commands are saved in each evaluation directory. The following adapter hashes identify the active saved LoRA, embedding, and LM-head weights; the base-model/config/tokenizer hashes are also recorded in the frozen input manifest.', '',
        '| Seed | Arm | Adapter SHA-256 | Predictions SHA-256 |', '|---|---|---|---|']
    for p in result['provenance']:
        lines.append(f"| {p['seed']} | {p['arm']} | `{p['checkpoint_adapter_sha256']}` | `{p['predictions_sha256']}` |")
    lines += ['', 'Exact remote checkpoint paths and evaluator manifest paths:', '']
    for p in result['provenance']:
        lines += [f"- {p['seed']} {p['arm']}: `{p['checkpoint']}`; manifest `{p['manifest']}`; 640 predictions."]
    lines += ['', 'Machine-readable endpoints: `runs/decoder_projected/confirmation_second640/confirmation_results.json`.', '',
        'Scope ends at this confirmation. No NextLat, optimization, representation analysis, retraining, or expansion-seed work was performed.', '']
    path=ROOT/'docs/reports/06_ACTUAL_SECOND640_CONFIRMATION.md'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':
    import sys
    if '--self-test' in sys.argv:
        assert mcnemar(0,0)==1 and mcnemar(0,3)==.25 and mcnemar(3,0)==.25
        assert sign_flip([[1,1],[1,1]])['two_sided_p']==.5
        assert sign_flip([[1,-1],[1,-1]])['two_sided_p']==1
        assert sign_flip([[1,1,1],[1,1,1]])['two_sided_p']==.25
        assert sign_flip([[1],[-1]])['two_sided_p']==1
        print('Exact McNemar and reaction-cluster sign-flip synthetic checks passed.')
    else:
        main()
