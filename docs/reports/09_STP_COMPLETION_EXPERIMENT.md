# Paper-STP Formulation, Strength, and Capacity Completion

## Executive conclusion

Report 09 completed the specific cells left unresolved by Report 08. It did
not rerun any existing trajectory and did not use the repeatedly inspected
512-reaction panel as confirmation evidence.

The preregistered development decision is:

- **paper STP rank 8, lambda=0.02 remains the selected paper configuration**;
- lambda=0.08 was numerically highest, but improved the mean treatment effect
  over lambda=0.02 by only 0.293 percentage points, below the frozen 0.5-point
  selection threshold;
- lambda=0.12 did not trigger the conditional lambda=0.16 cell;
- paper STP rank 128 at lambda=0.02 was worse than rank 8 in both matched-seed
  treatment-effect contrasts, so no further rank-128 paper run was triggered;
- released STP rank 8/lambda=0.02 and paper STP rank 8/lambda=0.02 remain
  statistically unresolved head-to-head. Their mean treatment effects on
  seeds 533/917 are +1.56 and +1.37 points, respectively; the paper-minus-
  released treatment contrast is -0.20 points with opposite-signed seed
  contrasts and a two-way bootstrap interval of [-1.86,+1.37] points.

This is encouraging development evidence for rank-8 STP, especially because
paper STP was positive in both seeds at all three tested coefficients. It is
not a confirmation result. Only two training seeds support the paper cells,
the same reactions were used for selection, and formulation differences are
smaller than the observed trajectory noise.

## Provenance and execution

| Item | Exact value |
|---|---|
| Report 08 final commit | `841f6e95635f88706b7fcdec14866e3c31932629` |
| Report 08 execution source | `acba1594f46356b2e097ad8ca8479a7df8a82a0a` |
| Report 09 preregistration commit | `e58a45a1fe2622f8ae17e0c5cfca4813e5c5a560` |
| Report 09 runner commit | `9d63939b835081cf85ba478d079d6b11d97e1b44` |
| Exact experiment source snapshot | `3b86ea70c2ad9663069ebe036487197e10326d35` |
| Post-run padded-rank diagnostic fix | `8f8f1810c01e874b287aad19280978e78f46c942` |
| Post-run official per-view-rank fix | `aef24562724fca8924ee68f01c855a08cd11f7b1` |
| Official STP upstream | `galilai-group/llm-jepa@ea0017c654ad917066ff32afc88276bea8ca5f7e` |
| Development manifest SHA-256 | `a2e6202a4abaf9a70f4700e04299a09964d38c10fce004022dc43e759aa6057d` |
| ChemFM weights SHA-256 | `24686705d779db6876acc09c81d64d432262ef8b5dbfccc385212587079ce419` |

The exact six trajectories were executed on one Thunder NVIDIA L40 at the
user's direction after A6000 capacity remained unavailable. The instance used
6 vCPUs, 100 GB disk, the base Ubuntu 22.04 image, PyTorch 2.3.0+cu121,
Transformers 4.45.2, PEFT 0.13.2, and Accelerate 1.0.1. The inherited output
directory retains the name `a6000` solely for path compatibility.

Before training, the remote environment passed all repository tests, reuse
integrity checks, model/archive hashes, and a four-update paper-STP rank-128
GPU preflight. The final local suite after the analysis fixes passed 109 tests
with 1 skip. The analysis-only fix makes the diagnostic accept the official
ten-slot empty-string padding when fewer than ten distinct valid candidates
exist. It changes no training, prediction, score, or ordering.

The local compact archive is
`runs/stp_completion/stp_completion_l40_compact.tar.zst`, SHA-256
`fbaf3b44ffb4002ea7f0f7d481503cab2c58b101c68f76ceb22c2b7b823c8553`.
It contains all six final adapters, six complete 512-row ordered prediction
files, teacher-forced rows, fixed-span diagnostics, decisions, and analysis.
Disposable optimizer states and redundant worker shards are excluded. The
Thunder instance was deleted only after the local hash, row counts, and six
adapters were verified.

## Report 08 matrix audit before new training

The distinction between formulation, coefficient, and rank is essential.
Report 08 did not establish a generic rank or lambda conclusion for all STP.

| Evidence entering Report 09 | Seeds | Status before Report 09 | Reuse/action |
|---|---:|---|---|
| Native r8 | 533, 917, 1301 | Complete | Reused control/predictions |
| Released STP r8, lambda=.02 | 533, 917, 1301 | Complete | Reused; three-seed released evidence |
| Native r128 | 533, 917 | Complete | Reused control/predictions |
| Released STP r128, lambda=.02 | 533, 917 | Complete | Reused; only released formulation at r128 |
| Paper STP r8, lambda=.02 | 533, 917 | Complete | Reused |
| Released STP r8, lambda=.005/.08 | 533, 917 | Complete | Reused; not a paper-STP lambda screen |
| Paper STP r128, lambda=.02 | -- | Missing | Two new paired trajectories |
| Paper STP r8, lambda=.08 | -- | Missing | Two new paired trajectories |
| Paper STP r8, lambda=.12 | -- | Missing | Two new paired trajectories |
| Paper STP r8, lambda=.16 | -- | Conditional only | Not triggered |
| Paper STP r128 at selected lambda | -- | Conditional if selected lambda !=.02 | Not needed; selected lambda=.02 |

The frozen pretraining objective diagnostic in Report 08 measured a released
STP lambda-weighted STP/NTP gradient-norm ratio of 0.0450 versus 0.00776 for
paper STP at lambda=.02: approximately a 5.8-fold pressure difference. Thus
the old same-lambda formulation comparison was real but incomplete, and the
released-only lambda screen could not select a coefficient for paper STP.

## Preregistered design and decisions

The full protocol is in
`docs/reports/09_STP_COMPLETION_PREREGISTRATION.md`. Every new trajectory used
the Report 08 train manifest, 320 optimizer updates, paired seeds 533/917,
native NTP objective and schedule, one paper-STP sample per example, final
layer, symmetric gradients, FP32 cosine, rank/alpha scaling of one, and the
same official 512-reaction five-view evaluation.

The primary endpoint was paper-minus-same-rank-Native official aggregated
exact top-1. The frozen decisions were applied mechanically:

1. Run lambda=.16 only if lambda=.12 exceeded both .02 and .08 by at least
   1.0 point in mean effect and both .12 seed effects were nonnegative.
2. Default to lambda=.02. An alternative needed at least +0.5 points over .02
   in mean effect with both seed effects nonnegative; within 0.5 points choose
   the lower lambda.
3. Select rank 128 only if the mean r128-minus-r8 treatment interaction was
   at least +0.5 points and both seed interactions were nonnegative.
4. Declare formulation superiority only if the absolute mean treatment-effect
   contrast was at least 0.5 points, both seed contrasts had the same sign,
   and the seed-by-reaction bootstrap interval excluded zero.

No conditional trajectory was run. No confirmation seed was run.

## Seed-1301 released-STP aggregation diagnosis

Report 08's puzzling seed-1301 result is now localized. Aggregate top-1 fell
from 20/512 to 14/512 (-1.17 points), while the five individual-view effects
were +0.39, 0.00, -0.78, 0.00, and +0.59 points. The aggregate loss is not an
arithmetic contradiction: individual-view gains and losses occurred on
different reactions, and reciprocal-rank aggregation changed which candidate
won across views.

### Top-k and beam location

| Cutoff | Native | Released STP | Difference | Paired bootstrap 95% CI |
|---:|---:|---:|---:|---:|
| Top-1 | 20 | 14 | -1.17 pp | [-2.34,-0.20] pp |
| Top-3 | 73 | 76 | +0.59 pp | [-1.37,+2.54] pp |
| Top-5 | 115 | 117 | +0.39 pp | [-1.76,+2.54] pp |
| Top-10 | 169 | 172 | +0.59 pp | [-1.56,+2.93] pp |

There were seven Native-only aggregate top-1 reactions and one STP-only
reaction. In all seven losses, the gold product remained in at least one STP
view and in the aggregate top 10. None was a beam-entry failure:

- 4/7 were **cross-view aggregation failures**: gold was top-1 in at least one
  STP view but a false candidate accumulated the larger reciprocal-rank score;
- 3/7 were **within-beam ranking failures**: gold remained in returned beams
  but was no longer top-1 in any view;
- 0/7 lost the gold from all returned beams.

No Native-only loss gained an individual-view gold top-1. Six lost one or two
such views; one retained the same count. Conversely, most view-level gains
occurred on reactions that remained aggregate-incorrect. The joint table was:

| Aggregate delta / number of gold-top1 views delta | Reactions |
|---|---:|
| -1 / -2 | 3 |
| -1 / -1 | 3 |
| -1 / 0 | 1 |
| 0 / -1 | 9 |
| 0 / 0 | 476 |
| 0 / +1 | 18 |
| 0 / +2 | 1 |
| +1 / -1 | 1 |

The official individual-view mean can therefore stay near neutral while a few
formerly correct aggregate reactions are specifically demoted. Gold appeared
in any view for 235 Native reactions versus 228 STP reactions, but gold was a
view top-1 for 44 versus 52. Mean distinct valid candidates changed only
26.12 to 26.21, mean pairwise view Jaccard 0.1924 to 0.1989, invalid rate
1.356% to 1.477%, and the aggregate winner's mean view count 4.389 to 4.412.
The important change is candidate ordering/consensus, not a large diversity or
validity collapse.

### Teacher-forced localization

Across all 2,560 reaction-view rows, released STP changed correct-token rate by
+0.0677 points, had 832 token top-1 gains versus 756 losses, and changed the
mean correct-token rank by essentially zero. A top-1 token decision first
differed in 44.9% of rows at median zero-based product position 6. Only 1.17%
of rows ever changed correct-token rank by at least five.

For the seven Native-only generation reactions (35 view rows), correct-token
rate was exactly unchanged, token top-1 gains and losses were 7/7, mean margin
changed by +0.0033, and no row had a rank change of five or more. Only 12/35
rows changed any teacher-forced top-1 decision, with median first position 7.5.
Thus the aggregate failures are not explained by systematic teacher-forced
gold-token degradation. They are mainly early free-running trajectory and
cross-view candidate-ranking effects. This is diagnostic evidence, not a
proposal to change the official inference rule.

## Paper STP at rank 128, lambda=.02

The same rank-128 Native controls used for released STP were reused.

| Seed | Native | Released STP | Paper STP | Released effect | Paper effect |
|---:|---:|---:|---:|---:|---:|
| 533 | 46/512 (8.98%) | 44/512 (8.59%) | 49/512 (9.57%) | -0.39 pp | +0.59 pp |
| 917 | 45/512 (8.79%) | 52/512 (10.16%) | 39/512 (7.62%) | +1.37 pp | -1.17 pp |
| Mean | 8.89% | 9.38% | 8.59% | +0.49 pp | -0.29 pp |

Paper r128 per-seed paired intervals were [-1.56,+2.73] points for seed 533
and [-3.32,+0.98] for seed 917; exact McNemar p values were 0.711 and 0.377.
The two-way paper treatment interval was [-2.54,+1.95] points.

Paper's r128-minus-r8 treatment interactions were -0.78 and -2.54 points,
mean -1.66 points with two-way interval [-4.10,+0.78]. Both tested seeds move
against a capacity rescue, but two seeds do not establish a universal adverse
capacity interaction. The conclusion is deliberately narrow: **rank 128 did
not improve paper STP at lambda=.02 in these paired seeds**, just as Report 08
only tested released STP capacity at lambda=.02.

Rank-128 paper top-1 changed across views by +1.76, -0.78, -1.17, -0.39,
-0.59 points in seed 533 and -0.59, -1.17, -1.37, -0.20, -0.59 in seed 917.
Top-3 changed +0.39/+0.59 points, top-5 +0.39/-0.59, and top-10 0.00/-1.95.
The raw final top-1 beams changed more often than at rank 8 (53.4% and 58.0%
of reaction-view pairs) and first differed at median token 3/4 rather than 1.

## Paper-STP coefficient response at rank 8

### Primary endpoint and uncertainty

| Lambda | Seed 533 Native -> Paper | Seed 917 Native -> Paper | Mean effect | Two-way bootstrap 95% CI |
|---:|---:|---:|---:|---:|
| .02 | 13 -> 20 (+1.37 pp) | 11 -> 18 (+1.37 pp) | +1.37 pp | [+0.39,+2.54] pp |
| .08 | 13 -> 19 (+1.17 pp) | 11 -> 22 (+2.15 pp) | +1.66 pp | [+0.39,+3.12] pp |
| .12 | 13 -> 23 (+1.95 pp) | 11 -> 16 (+0.98 pp) | +1.46 pp | [+0.20,+2.93] pp |

Within-seed top-1 bootstrap intervals and exact McNemar p values were:

| Lambda / seed | 95% CI | McNemar p |
|---|---:|---:|
| .02 / 533 | [+0.20,+2.73] pp | 0.0654 |
| .02 / 917 | [+0.39,+2.54] pp | 0.0391 |
| .08 / 533 | [0.00,+2.34] pp | 0.1094 |
| .08 / 917 | [+0.78,+3.71] pp | 0.0074 |
| .12 / 533 | [+0.59,+3.32] pp | 0.0129 |
| .12 / 917 | [-0.20,+2.34] pp | 0.2266 |

These intervals are descriptive development-panel uncertainty. Multiple
coefficients were inspected and only two seeds were used; they are not a
license to claim a confirmed significant STP effect.

### Five-view top-1 effects

| Lambda / seed | Aggregate | View 1 | View 2 | View 3 | View 4 | View 5 |
|---|---:|---:|---:|---:|---:|---:|
| .02 / 533 | +1.37 | +1.37 | +0.98 | +1.37 | +1.56 | +1.56 |
| .02 / 917 | +1.37 | -0.20 | -0.20 | +0.78 | +1.76 | +0.78 |
| .08 / 533 | +1.17 | +1.17 | +0.98 | +0.98 | +2.73 | +0.59 |
| .08 / 917 | +2.15 | +0.78 | +1.37 | +1.56 | +2.54 | +2.34 |
| .12 / 533 | +1.95 | +0.78 | +0.78 | +0.78 | +2.15 | +1.37 |
| .12 / 917 | +0.98 | +0.39 | -0.20 | +0.59 | +0.59 | +0.98 |

All entries are percentage-point paper-minus-Native changes on 512 reactions.
The effect is not confined to one serialization view, but view effects remain
small and seed-dependent.

### Aggregate top-k effects

| Lambda / seed | Top-1 | Top-3 | Top-5 | Top-10 |
|---|---:|---:|---:|---:|
| .02 / 533 | +1.37 | +3.12 | +4.88 | -0.20 |
| .02 / 917 | +1.37 | +0.98 | +2.15 | +5.86 |
| .08 / 533 | +1.17 | +2.54 | +2.34 | -0.39 |
| .08 / 917 | +2.15 | +4.69 | +4.30 | +5.66 |
| .12 / 533 | +1.95 | +1.37 | +2.15 | +0.98 |
| .12 / 917 | +0.98 | +2.73 | +4.88 | +5.08 |

Rank-8 improvements generally extend beyond top-1, although seed 533 has a
small top-10 decline at .02/.08. Beam diagnostics show few Native-only top-1
losses relative to gains: 2/9 and 1/8 at .02, 2/8 and 2/13 at .08, and 2/12
and 3/8 at .12. Those losses are almost entirely within-beam ranking or
cross-view aggregation; no rank-8 paper cell has a gold beam-entry loss among
its Native-only top-1 cases.

### Teacher-forced behavior

| Lambda / seed | Token CE change | Correct-token rate | Margin change |
|---|---:|---:|---:|
| .02 / 533 | -0.00168 | +0.076 pp | +0.201 |
| .02 / 917 | -0.00574 | +0.234 pp | +0.285 |
| .08 / 533 | -0.00258 | +0.103 pp | +0.198 |
| .08 / 917 | -0.00788 | +0.307 pp | +0.439 |
| .12 / 533 | -0.00099 | +0.109 pp | +0.210 |
| .12 / 917 | -0.00846 | +0.298 pp | +0.282 |

All rank-8 paper cells improve all three teacher-forced summaries. They do not
select lambda: seed 533 has its weakest CE improvement but strongest generated
top-1 at .12, while seed 917 has its strongest CE improvement but weakest
generated top-1 at .12. The native-token to free-generation mapping remains
non-monotonic.

At rank 128/.02, CE improves by -0.00920/-0.00690, yet correct-token rate
changes by -0.052/-0.025 points and margin by -0.194/-0.095. Seed 917 combines
better CE with -1.17 points exact top-1, another explicit token-loss versus
sequence-generation disconnect.

### Trained-checkpoint STP pressure

Fixed spans and 16 reactions were used identically for every checkpoint.

| Rank/lambda / seed | Paper loss | Raw STP grad norm | Weighted STP/NTP ratio | Grad cosine with NTP |
|---|---:|---:|---:|---:|
| r8/.02 / 533 | 1.3135 | 5.638 | 0.0411 | -0.025 |
| r8/.02 / 917 | 1.3227 | 6.239 | 0.0411 | +0.001 |
| r8/.08 / 533 | 1.3059 | 6.164 | 0.1364 | +0.129 |
| r8/.08 / 917 | 1.3177 | 6.251 | 0.1842 | +0.104 |
| r8/.12 / 533 | 1.2936 | 5.059 | 0.1944 | -0.094 |
| r8/.12 / 917 | 1.2910 | 4.956 | 0.1815 | -0.074 |
| r128/.02 / 533 | 1.2833 | 6.991 | 0.0332 | -0.132 |
| r128/.02 / 917 | 1.3007 | 4.413 | 0.0199 | -0.074 |

Increasing lambda does increase persistent auxiliary pressure, but not in
direct proportion: by .12, the raw paper gradient has shrunk and the weighted
ratio is about 0.19 rather than 1.5 times the .08 value. Lambda=.12 lowers the
paper loss slightly and has a slightly lower mean generation effect than .08,
but there is no sharp collapse, validity failure, or clear excessive-
straightening threshold. The evidence is a flat, noisy generation response,
not a demonstrated high-lambda failure mechanism.

## Released versus paper after fair characterization

The frozen selector retains released r8/.02 as the released configuration and
paper r8/.02 as the paper configuration.

| Seed | Released treatment effect | Paper treatment effect | Paper minus released |
|---:|---:|---:|---:|
| 533 | +2.15 pp | +1.37 pp | -0.78 pp |
| 917 | +0.98 pp | +1.37 pp | +0.39 pp |
| Mean | +1.56 pp | +1.37 pp | -0.20 pp |

The two-way interval for the treatment-effect contrast is [-1.86,+1.37]
points. The direct paired model comparison gives the same seed effects, with
per-seed intervals [-2.34,+0.78] and [-0.98,+1.76]. The signs disagree and the
frozen superiority rule is false.

Direct paper-minus-released top-k changes are -0.78/-0.20/+1.56/-1.56 points
at top-1/3/5/10 for seed 533 and +0.39/+0.78/+1.37/+1.37 for seed 917.
Paper's teacher-forced CE is worse than released by +0.00215 in seed 533 and
better by -0.00319 in seed 917; paper margins are slightly lower in both.
There is no consistent proxy or generation basis for declaring either
formulation superior.

At equal lambda on frozen initial batches, paper used 5.8-fold less weighted
gradient pressure than released, yet achieved a similar rank-8 treatment
effect. The paper-specific lambda study shows that increasing its pressure to
.08/.12 does not materially exceed .02 under the prespecified selection rule.
This resolves the fairness concern without establishing that weak pressure is
causally optimal.

At rank 128/.02, direct paper-minus-released effects are +0.98 and -2.54
points, mean -0.78 with interval [-4.10,+2.34]. Formulation-by-capacity behavior
is strongly seed-dependent; absolute rank-128 accuracy is not evidence for an
STP treatment benefit.

## Runtime and resource record

Paired seeds trained concurrently. Five-view evaluation, not training, was the
dominant cost.

| Condition | Paired training wall | Mean GPU util. | Peak train VRAM | Eval seed 533 | Eval seed 917 |
|---|---:|---:|---:|---:|---:|
| Paper r128/.02 | 377.9 s | 77.4% | 17,984 MiB | 1,691.9 s | 1,647.9 s |
| Paper r8/.08 | 349.9 s | 84.5% | 14,958 MiB | 1,400.3 s | 1,403.5 s |
| Paper r8/.12 | 379.7 s | 77.2% | 14,958 MiB | 1,407.4 s | 1,406.3 s |

Evaluation sustained 98.0--98.8% mean GPU utilization. The longer-than-training
wall time was therefore genuine exact five-view beam work, not a stalled job.

## What Reports 07--09 establish

1. The executable released STP objective and the literal paper equation are
   distinct, faithfully implemented conditions.
2. Released r8/.02 improves the 512-reaction development endpoint in seeds
   533/917 but worsens seed 1301. It is not a deterministic or seed-uniform
   generation improvement.
3. Seed 1301's failure is specifically top-1 within-beam/cross-view ordering;
   gold survives in every causal loss, and teacher-forced gold-token behavior
   does not explain the demotions.
4. Paper r8 improves exact top-1 in both tested seeds at .02, .08, and .12,
   with supportive top-k and teacher-forced changes. The three lambda means are
   too close to identify a meaningful dose optimum on this development panel.
5. Rank 128 does not improve the paper treatment effect at .02 in either seed.
   Combined with Report 08, there is no evidence that simply increasing LoRA
   capacity rescues either tested formulation at the published coefficient.
6. Paper and released STP remain unresolved when compared at their selected
   rank/lambda. Similar effects arise despite materially different auxiliary
   gradient pressure.
7. Better CE/margins are neither necessary nor sufficient to rank the final
   generation configurations. Official generation remains the deciding
   endpoint.

## What remains unresolved

- Whether either r8/.02 STP formulation has a positive expected effect on new
  training seeds and untouched reactions.
- Whether paper STP's apparent two-seed consistency survives independent
  trajectories; seed 1301 was deliberately not added in this selection run.
- Whether released and paper STP differ by less than the trajectory noise or
  have genuinely different expected effects.
- Any capacity conclusion outside the explicitly tested formulation/lambda
  cells. No claim is made about rank 128 at unselected paper lambdas.
- The authors' natural-language claims, other data budgets, full-model tuning,
  and other architectures; none is tested here.

## Single best next experiment

The most justified continuation is one **locked three-arm confirmation**, not
more development tuning:

- Native r8;
- released STP r8/lambda=.02;
- paper STP r8/lambda=.02;
- at least four new, paired training seeds;
- a newly prespecified untouched 512-reaction panel sampled from the remaining
  official test set;
- the unchanged official five-view generation endpoint;
- prespecified Native contrasts for each STP arm and a paper-minus-released
  contrast with multiplicity-aware uncertainty;
- no lambda, rank, formulation, seed, or panel selection after outcomes are
  visible.

A three-arm confirmation is preferable to arbitrarily discarding one
formulation: Report 09 completed the pressure/capacity evidence and found no
reliable winner. If compute permits only one STP arm, paper r8/.02 is the more
informative choice because it is the preregistered paper selector result and
currently lacks an independent third seed; that constraint should be declared
before launch rather than justified from confirmation outcomes.

## Reproducibility map

- Preregistration: `docs/reports/09_STP_COMPLETION_PREREGISTRATION.md`
- Consolidated analysis: `runs/stp_completion/a6000/analysis.json`
- Seed-1301 ordered-beam diagnosis:
  `runs/stp_completion/a6000/existing_diagnostics/released_r8_l0.02_seed1301_beams.json`
- Seed-1301 teacher-token localization:
  `runs/stp_completion/a6000/existing_diagnostics/released_r8_l0.02_seed1301_teacher_tokens.json`
- Raw trajectories and endpoints: `runs/stp_completion/a6000/trajectories/`
- Environment/integrity metadata: `runs/stp_completion/a6000/environment.json`
- Compact archive: `runs/stp_completion/stp_completion_l40_compact.tar.zst`
