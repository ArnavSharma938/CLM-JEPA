# Untouched STP confirmation and frozen latent-predictability audit

## Executive record

This report closes two independent tracks. The first is a preregistered,
untouched-panel generation confirmation of Native ChemFM, released STP, and
paper-equation STP. The second is a frozen-checkpoint audit asking whether
future causal states are predictable, whether that information survives the
decoder, whether it is stable across equivalent SMILES serializations, and
whether STP changes those properties. The frozen audit was specified and its
reaction splits locked without access to confirmation predictions.

The endpoint result is **inconclusive, with a small, seed-consistent positive
signal for released STP and no replicated signal for paper STP**. On 640 new
reactions and four new paired seeds, released STP improved official five-view
top-1 by `+.547` percentage points; all four seed effects were positive, but
the crossed seed/reaction 95% interval was `[-.195,+1.328]` pp and the
Holm-adjusted reaction-cluster sign-flip p-value was `.1025`. Paper STP changed
top-1 by `+.117` pp with mixed seed effects and a 95% interval of
`[-.898,+1.133]` pp.

The mechanistic audit finds a clear but short-range property of ordinary
ChemFM: late-layer future product states are predictable from the current
state (`R2=.641,.432,.161,.008` at horizons 1/2/4/8 using the residual MLP),
and those predictions remain decoder-functional at short horizons. Neither
STP formulation increases that predictable fraction. Released STP changes
final product `R2` by `-.0057,-.0052,-.0006,+.0009`; paper STP changes it by
`-.0102,-.0080,-.0036,+.0011`. Cross-view chemical agreement also emerges
through depth, but neither STP formulation consistently improves it. A small
validation-defined subset is simultaneously predictable, cross-view stable,
and decoder-functional; STP does not consistently enlarge it.

These are separate statements. The generation result does not establish a
predictability mechanism, and the frozen probe result does not negate the
small released-STP endpoint signal.

## 1. Provenance and independence

| Item | Record |
|---|---|
| Scientific implementation entering both tracks | `57bdcd1` |
| Confirmation preregistration | `cf62629` |
| Platform-stable manifest hashes | `d3461a0` |
| Outcome-blind 640-panel amendment | `38dbfc8` |
| Frozen-audit preregistration | `2f94ad1` |
| Final audit execution source before this report | `eaac712` |
| Report commit | recorded by the commit containing this file |
| STP upstream | `galilai-group/llm-jepa@ea0017c654ad917066ff32afc88276bea8ca5f7e` |
| Hardware | one NVIDIA A6000, six vCPU, 100 GB Thunder volume |

The confirmation panel was selected from unused official USPTO-MIT test
reactions. Its SHA-256 is
`3655e58404c3509c04b15cd4ffcdf15723f9be62e76c48c676ccb7decf9e2945`.
It contains 640 unique reaction identities and 3,200 five-view examples. The
reaction-level result SHA-256 is
`f97a8fc5e6b1f58da8f753e283b4b7320ab1ecf417f3d555be686342d5366451`.
The original 1,280-panel manifest remains recorded, but inference had not
started when the user made the outcome-blind reduction to its first 640
salt-hash-ordered reactions.

The audit split SHA-256 is
`315a5012d79f8704b3119b03e3bfdc96f9618f189aa5ca1d1a602cfeed9ff560`.
It assigns 640/192/192 reaction identities to probe train/validation/test.
The runner checks chemical-pair identity rather than namespace-specific group
IDs and aborts on any confirmation overlap. It consumes no confirmation
prediction or outcome file.

## 2. Exact confirmation experiment

Each seed starts three arms from the same base model and follows the same
data order. All arms use rank 8, alpha 8, LoRA dropout `.1`, targets
`q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj`, and saved
`embed_tokens,lm_head`. Training uses the frozen 1,280 serialized rows (256
reactions x five views), four epochs, 320 optimizer steps, BF16 model
computation, fused AdamW, the existing cosine/min-LR schedule, and the original
ChemFM serialization. Native NTP remains active in all arms. The two treatment
arms add either released `linear=random_span` STP or literal paper-equation
STP with lambda `.02` and the already validated FP32 cosine/symmetric-gradient
implementation.

The primary endpoint is official five-view beam-10 reciprocal-rank aggregated
exact top-1. The first seed pair was 2027/3163. Because both treatments were
positive in both first seeds, the preregistered futility condition was false
and seeds 4211/5393 were run. There was no success stop.

### 2.1 Top-1 endpoint and paired inference

| Seed | Native | Released | Released-Native | wins/losses | Paper | Paper-Native | wins/losses |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2027 | 2.969% | 3.438% | +.469 pp | 6/3 | 3.594% | +.625 pp | 7/3 |
| 3163 | 3.750% | 4.375% | +.625 pp | 9/5 | 4.531% | +.781 pp | 11/6 |
| 4211 | 3.750% | 4.375% | +.625 pp | 9/5 | 2.813% | -.938 pp | 6/12 |
| 5393 | 3.594% | 4.063% | +.469 pp | 13/10 | 3.594% | 0 pp | 11/11 |
| Mean | 3.516% | 4.063% | **+.547 pp** | 37/23 | 3.633% | **+.117 pp** | 35/32 |

| Contrast | Mean effect | crossed seed/reaction 95% CI | reaction-cluster p | Holm p |
|---|---:|---:|---:|---:|
| Released-Native | +.547 pp | [-.195,+1.328] pp | .05125 | .10249 |
| Paper-Native | +.117 pp | [-.898,+1.133] pp | .76766 | .76766 |

The released effect is positive in all four seeds, but the reaction-aware
interval includes zero and the two-contrast adjusted p-value does not meet a
`.05` threshold. The seed-only t interval is positive because the four point
effects are unusually similar; with four seeds it is fragile and is not used
as the primary uncertainty statement. Paper STP does not reproduce its
positive two-seed development behavior on the untouched panel.

### 2.2 Top-k, views, and validity

| Arm | top-1 | top-3 | top-5 | top-10 | candidate validity |
|---|---:|---:|---:|---:|---:|
| Native | 3.516% | 17.227% | 24.648% | 35.859% | 98.445% |
| Released | 4.063% | 17.148% | 24.219% | 35.313% | 98.066% |
| Paper | 3.633% | 16.953% | 25.586% | 35.977% | 98.427% |
| Released-Native | +.547 pp | -.078 pp | -.430 pp | -.547 pp | -.379 pp |
| Paper-Native | +.117 pp | -.273 pp | +.938 pp | +.117 pp | -.018 pp |

Released STP's top-1 gain is not accompanied by wider gold survival: its mean
top-3/5/10 and candidate validity are slightly lower. That localizes the
positive endpoint to ordering near rank one rather than a general improvement
of the candidate set. Paper STP has no consistent top-1 effect and a positive
top-5 change, again showing that top-k movements are not interchangeable with
the prespecified endpoint.

Mean individual-view top-1 effects for views 0--4 were:

| Contrast | view 0 | view 1 | view 2 | view 3 | view 4 |
|---|---:|---:|---:|---:|---:|
| Released-Native | -.195 pp | +.859 pp | +.078 pp | +.391 pp | +.195 pp |
| Paper-Native | -.078 pp | -.078 pp | -.195 pp | -.078 pp | +.469 pp |

No single view accounts for all of released STP's aggregate effect. Official
aggregation is retained unchanged.

### 2.3 Compute record

| Arm | mean train wall | maximum peak VRAM | mean endpoint wall/checkpoint | mean validation CE |
|---|---:|---:|---:|---:|
| Native | 572.8 s | 6.207 GiB | about 2,290 s | .15704 |
| Released | 704.4 s | 6.206 GiB | about 2,284 s | .15668 |
| Paper | 725.9 s | 6.206 GiB | about 2,273 s | .15570 |

Validation CE is a two-reaction training-time diagnostic and is not an
endpoint. Its near equality does not explain the generated top-1 result.

## 3. Performance and numerical integrity

The confirmation evaluator was profiled before the new panel was opened.
Length-balanced assignment was ordered-beam exact but produced only `1.0195x`
median speedup, below the locked `1.03x` retention threshold. Five workers
gave `1.0128x` and six gave `.9833x`, below the `1.05x` continuation gate.
The established four-worker evaluator was therefore retained. STP has no
inference-time module, so there is no separate approximate STP generation
path.

The frozen audit exposed different bottlenecks and retained only changes that
were useful to this workload: cached gathered layer states, removal of
repeated tensor conversion, reuse of immutable probes, native-vocabulary-only
decoder metrics, grouped suffix replay, batched candidate scoring, and
vectorized reaction-cluster bootstrap. Candidate scoring was checked on 128
rows: batch 1 and batch 128 produced exactly identical JSON metrics. Exact
five-view generation was never replaced.

Intermediate-layer causal replay uses BF16 SDPA kernels whose batched and
one-position execution orders differ. Across the recorded 2,410-position
layer-6 parity panel, maximum RMS state error was `.003339`, minimum cosine was
`.999928`, and maximum element error was `.0625`; four LM-head top-1 ties
changed. This passed the preregistered RMS `<=.02` and cosine `>=.999` gate but
is not described as bitwise equivalence. Final-layer projection and candidate
batching are exact within their recorded gates.

Correct audit wall times are:

| Stage | Wall |
|---|---:|
| Frozen extraction | 235.5 s |
| Probe fitting | 17,492.6 s (4h51m33s) |
| Decoder audit | 2,424.5 s (40m24s) |
| Alternate-view extraction | 497.9 s |
| View analysis | 2,286.6 s (38m07s) |
| Candidate extraction | 1,863.8 s (31m04s) |
| Candidate scoring | 1,445.8 s (24m06s) |
| Development-view coupling | 439.8 s |
| Initial summarization | 70.1 s |
| Vectorized final uncertainty analysis | 172 s |

An earlier progress update incorrectly converted `2,424` seconds to 6h44m;
the table above is the corrected execution record.

## 4. Frozen audit design

Frozen r8 checkpoints are Native/released seeds 533, 917, and 1301 and paper
seeds 533 and 917. At layers 6, 16, 21, and final post-RMSNorm, constant,
ridge, and width-128 residual-MLP probes predict `h[t+k]` for `k=1,2,4,8`.
Source and product are separate. Inputs are either `h[t]` or a raw four-state
causal history. Targets use a train-only rank-256 PCA for fitting and are
reconstructed before full-state metrics. Test statistics are reaction
balanced; shuffled-reaction targets, matched horizons, causal suffix
perturbations, and fixed support annotations are retained.

At the final layer, predicted states go directly through the unchanged LM
head. Earlier predictions are injected into the remaining frozen blocks using
the causal prefix cache. Intermediate replay is a preregistered representative
64-reaction diagnostic; final-layer metrics cover all eligible test positions.

![Product predictable fraction](../../runs/latent_predictability_audit/plots/product_predictable_fraction.svg)

![Decoder JS divergence](../../runs/latent_predictability_audit/plots/product_decoder_js.svg)

## 5. Future-state predictability

### 5.1 Absolute predictable fraction

Residual-MLP product `R2` from the current state:

| Layer | k=1 | k=2 | k=4 | k=8 |
|---|---:|---:|---:|---:|
| 6 | -1.238 | -.683 | -1.114 | -1.176 |
| 16 | .277 | .153 | .009 | -.093 |
| 21 | .635 | .418 | .160 | .017 |
| final | **.641** | **.432** | **.161** | **.008** |

Native source values at the final layer are `.511,.298,.085,-.009`. Thus
late-layer product computation is more one- and two-token predictable than
source computation, while neither segment has useful eight-token predictable
variance under this probe. Negative early-layer `R2` means these fixed probes
underperform the train-target mean; it is not evidence that the state contains
no future information under every possible decoder.

At the final product layer, ridge gives `.563,.359,.121,-.010`; the residual
MLP improves by `.077,.073,.040,.018`. Nonlinearity matters, but it does not
create long-horizon predictability.

The raw-history residual MLP gives `.561,.344,.068,-.095`, worse than the
current-state probe. Because raw concatenation was held to the same small
capacity and fixed regularization, this is evidence about this controlled
probe class, not a claim that causal history contains less information.

The shuffled-reaction control gives final product k=1 `R2=-.606` versus
`.641` for matched targets. The future signal is reaction-specific and is not
explained by only position, length, or marginal state statistics.

### 5.2 STP effects on predictable fraction

Final product current-state residual-MLP `R2` treatment effects:

| Treatment | k=1 | k=2 | k=4 | k=8 |
|---|---:|---:|---:|---:|
| Released-Native | -.00568 | -.00520 | -.00058 | +.00092 |
| Paper-Native | -.01024 | -.00800 | -.00360 | +.00113 |

Reaction-bootstrap intervals exclude zero for released k=1/k=2 and paper
k=1/k=2; the changes are small and negative. At k=8, both absolute `R2` and
treatment effects are near zero. Layers 16 and 21 likewise show no consistent
positive pattern. Large, unstable layer-6 changes occur around strongly
negative baselines and are not interpreted as useful forecastability.

This falsifies the narrow mechanism that either tested STP improves generation
by increasing the fraction of late-layer future-state variance recoverable by
these locked constant/linear/small-MLP probes.

## 6. Decoder-functional predictability

For Native final product states predicted by the residual MLP:

| Horizon | JS(true,predicted) | top-1 agreement | next-token log p | next-token rank |
|---:|---:|---:|---:|---:|
| 1 | .126 | .784 | -.985 | 2.66 |
| 2 | .231 | .629 | -1.700 | 5.03 |
| 4 | .388 | .401 | -2.800 | 11.40 |
| 8 | .487 | .271 | -3.307 | 15.73 |

Latent forecastability is therefore behaviorally preserved at short horizons
and rapidly degrades. This is not merely low Euclidean MSE: predicted k=1
states reproduce the true state's decoder top-1 on 78.4% of positions.

Released STP changes final JS by `+.00029,+.00081,+.00084,+.00082`; all
intervals include zero. Its top-1-agreement changes are
`+.00187,-.00026,+.00352,-.00118`, also all interval-overlapping zero. Paper
changes JS by `+.00204,+.00320,+.00195,+.00035`; only k=2 is clearly positive
(worse), while all top-1-agreement intervals include zero. Intermediate replay
has smaller samples and alternating signs across layer and horizon. It does
not supply a consistent STP advantage.

Thus STP does not measurably strengthen decoder preservation of predictable
future state in the tested frozen checkpoints.

## 7. Chemical semantic supports

Native final k=1 product `R2` is `.689` on atom-to-atom positions, `.523` on
event-to-next-event positions, and `.506` in the reaction-center window,
compared with `.641` on arbitrary positions. Chemical supports remain
forecastable, but event and center supports are not more predictable than the
arbitrary average under this normalization.

At k=1, released treatment effects are `-.0072` atom-to-atom, `-.0018` event,
and `-.0072` reaction center. Paper effects are `-.0082`, `-.0104`, and
`-.0106`, respectively. The paper component-boundary estimate is `+.153`, but
only three product positions are eligible at k=1 and the result is discarded
as too sparse for inference. Across horizons, signs vary rather than defining
a robust support on which STP improves forecastability.

## 8. Gold versus wrong generated trajectories

The locked probes were replayed on archived gold products and beam candidates.
Under Native, wrong-minus-gold latent normalized MSE is
`-.0346,-.0218,-.0103,-.0046`: wrong trajectories are, if anything, slightly
easier to forecast in latent space. Decoder JS separation is
`+.0005,+.0140,+.0217,+.0181`, so beyond k=1 their predicted states are less
functionally faithful than gold trajectories. No single metric therefore
orders gold and wrong candidates at every horizon.

Released STP changes wrong-minus-gold latent MSE by
`+.0013,+.0015,-.0007,+.0025`, with intervals overlapping zero, and changes JS
by approximately `0,-.0013,-.0025,+.0002`, also interval-overlapping zero.
It does not create a reliable gold-versus-wrong forecastability separator.

Paper STP changes wrong-minus-gold JS by
`-.0052,-.0084,-.0064,-.0018` and top-1 agreement by
`+.0089,+.0151,+.0087,+.0001`. Through k=4, both directions mean the probe
more faithfully reproduces decoder behavior on wrong paths relative to gold
paths. Latent MSE shifts the same way at k=2 (`-.0083`) and k=8 (`-.0047`).
This is descriptive candidate-relative evidence, not proof of why paper STP
failed confirmation, but it provides no support for a mechanism in which STP
selectively makes correct trajectories more predictable.

## 9. Seed-1301 natural experiment

Report 01 established that all seven Native-only released-STP top-1 successes
retain gold in STP's aggregate top 10: three fail within beam ranking and four
at cross-view aggregation. On their 35 views, this audit compares gold against
the exact promoted wrong candidates.

Released-minus-Native changes in wrong-minus-gold latent normalized MSE are
`-.0039,-.0123,+.0071,-.0035` across horizons 1/2/4/8; decoder JS changes are
`+.0074,-.0123,-.0007,-.0084`; decoder top-1-agreement changes are
`-.0236,+.0227,-.0020,+.0191`. The signs are not coherent across horizons and
the sample is seven reactions. The audit does not identify a stable shift of
forecastability toward the promoted wrong candidate.

For the four aggregation-specific failures, final product atom-level
released-minus-Native change in within/between view variability is `+.00072`,
matched-view cosine is `-.00581`, and candidate Jaccard is `+.00321`. These are
small and do not support a large serialization-invariance change as the cause.
Motif/component ratios are numerically unstable in this four-reaction subset
because their matched between-identity denominators approach zero; they are
not interpreted.

## 10. Cross-view chemical invariance

Graph correspondence, not token position, aligns atoms, motifs, and eligible
components across one canonical and four randomized serializations. For Native
product atoms on held-out reactions:

| Layer | within/between variability | matched-view cosine | identity retrieval | centered CKA |
|---|---:|---:|---:|---:|
| 6 | .922 | .616 | .122 | .356 |
| 16 | .794 | .633 | .398 | .418 |
| 21 | .649 | .735 | .221 | .434 |
| final | .572 | .720 | .167 | .544 |

Lower within/between is more view invariant. Serialization agreement generally
emerges through depth, while identity retrieval peaks at layer 16 and declines
afterward. Increased view agreement and retained between-reaction
discrimination are therefore distinct properties.

Released STP changes the product-atom ratio by `-.0062` at layer 16,
`-.0188` at layer 21, and `-.0023` final; the final matched cosine changes
`-.0031` and retrieval `+.0003`. The layer-21 ratio change is the clearest
localized improvement, but it does not persist as a final-layer advantage.
Paper changes the final ratio by `+.0119`, cosine by `-.0081`, and retrieval by
`-.0065`, a small final-layer deterioration on all three directions. Seed
effects and other layers are mixed. Neither formulation consistently improves
serialization invariance while preserving identity discrimination.

## 11. Invariance coupled to five-view generation

On the reused 512-reaction development beams, final product atom-level
Spearman correlations are modest. For Native, matched-view cosine versus
aggregate gold score is `-.187`, within/between ratio versus gold score is
`+.128`, and centered CKA versus gold score is `-.180`. Released values are
`-.211,+.159,-.194`; paper values are `-.182,+.133,-.172`.

These signs do not say that view invariance causes worse generation: reaction
complexity and representation scale remain uncontrolled confounders. They do
show that more raw cross-view agreement is not, by itself, a reliable proxy
for official five-view success. Candidate-list Jaccard correlations are near
zero to small. The seed-1301 aggregation subset likewise has no large final
latent-agreement change.

## 12. Predictability x invariance x decoder relevance

Validation thresholds define the joint test subset; test reactions never set
those thresholds. In Native current-state probes, `11.1%,11.1%,13.5%,6.8%`
of test reactions fall in the joint subset at horizons 1/2/4/8. At k=1 its
decoder JS is `.123` versus `.168` outside, and top-1 agreement is `.795`
versus `.715`. Across reactions, forecast `R2` and decoder JS correlate about
`-.84`: more predictable state is strongly more decoder-faithful. The
invariance/JS correlation is about `+.22`; because lower variability ratio and
lower JS are favorable, the sign is directionally consistent with a jointly
useful subset.

Released subset prevalence is `11.3%,10.8%,11.5%,6.5%`; paper is
`10.2%,9.9%,11.5%,6.5%`. Raw-history prevalence is similarly variable. STP
does not consistently enlarge the subset or improve its decoder metrics.

This is the central positive mechanistic result: ChemFM contains a minority of
positions that are simultaneously causal-context predictable,
serialization-stable, and decoder-functional. The central negative result is
that neither tested STP objective selectively strengthens that property.

## 13. Conclusions and scope

What this experiment establishes:

1. Released STP has a small positive untouched-panel top-1 estimate in all
   four new seeds, but reaction-aware uncertainty still includes zero.
2. Paper STP does not replicate its development-panel top-1 gain.
3. Released top-1 improvement is not accompanied by top-3/5/10 improvement or
   increased candidate validity.
4. Future ChemFM states are substantially predictable only at short horizons
   and mainly in late layers; this predictable information affects the
   decoder.
5. Neither released nor paper STP increases that predictable fraction or its
   decoder preservation under the locked probes.
6. Cross-view chemical invariance emerges through depth, but neither STP
   formulation consistently improves invariance while preserving chemical
   identity discrimination.
7. Wrong generated trajectories can be as latent-predictable as gold ones;
   paper STP often shifts candidate-relative decoder forecastability toward
   wrong paths rather than selectively toward gold.

What remains unresolved:

1. The true released-STP top-1 effect may be small positive or zero; 640
   reactions x four seeds do not resolve that interval.
2. Frozen probes test specified function classes, not the maximum information
   theoretically extractable from hidden histories.
3. Candidate replay is observational and cannot prove that predictability or
   invariance causes beam ordering.
4. The audit does not test a learned future-state JEPA objective and does not
   establish that such an objective would improve generation.

The appropriate endpoint verdict is **INCONCLUSIVE for released STP** and
**not supported for paper STP at r8/lambda=.02**. The mechanistic verdict is
that generic trajectory straightening is not supported as an explanation of
the endpoint behavior.

No new JEPA training objective is justified directly by these data. If a later
objective is tested, the evidence narrows it to short-horizon, late-layer,
decoder-functional prediction with explicit cross-view identity preservation
and a candidate-relative control; generic hidden-state straightening or
unqualified future-state MSE is not supported. Before such method development,
the single clean continuation is a prespecified endpoint-only replication of
released r8/lambda=.02 with new paired Native seeds on the unused second 640
reactions from the already locked 1,280 manifest, with no further selection.

## 14. Reproducibility map

| Artifact | Path |
|---|---|
| Confirmation protocol | `docs/preregistrations/STP_UNTOUCHED_CONFIRMATION_PROTOCOL.md` |
| Audit protocol | `docs/preregistrations/LATENT_PREDICTABILITY_DECODER_COUPLING_PROTOCOL.md` |
| Confirmation manifest | `data/clm_jepa_uspto_mit_stp_confirmation/untouched_640.jsonl` |
| Confirmation analysis | `runs/stp_confirmation/a6000/analysis/confirmation.json` |
| Reaction-level top-1 | `runs/stp_confirmation/a6000/analysis/reaction_level_top1.jsonl` |
| Frozen-audit analysis | `runs/latent_predictability_audit/analysis.json` |
| Audit execution times | `runs/latent_predictability_audit/execution.json` |
| Absolute/paired probe tables | `runs/latent_predictability_audit/tables/probe_*.csv` |
| Absolute/paired decoder tables | `runs/latent_predictability_audit/tables/decoder_*.csv` |
| Candidate tables | `runs/latent_predictability_audit/tables/candidate_*.csv` |
| Cross-view tables | `runs/latent_predictability_audit/tables/invariance_*.csv` |
| Generation coupling | `runs/latent_predictability_audit/tables/generation_invariance_coupling.csv` |
| Joint analysis | `runs/latent_predictability_audit/tables/joint_predictability_invariance.csv` |
| Seed-1301 invariance subset | `runs/latent_predictability_audit/tables/seed1301_aggregation_invariance.csv` |

The compact archive contains the raw JSONL necessary to regenerate every
table, final confirmation checkpoints and ordered beams, protocols, manifests,
logs, and this report. Large immutable hidden-state caches and reconstructible
probe-weight directories are intentionally excluded.
