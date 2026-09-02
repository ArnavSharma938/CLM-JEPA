# Semantic Tube Prediction on ChemFM: consolidated Reports 07--09

## Executive conclusion

This is the authoritative report for the complete Semantic Tube Prediction
(STP) program formerly split across Reports 07, 08, and 09. It also contains
the preregistered frozen representation study of every trained Native/STP
checkpoint.

The generation verdict remains **INCONCLUSIVE**. Rank-8 released STP at
lambda=.02 changes official five-view exact top-1 by `+2.15`, `+0.98`, and
`-1.17` percentage points across seeds 533/917/1301: mean `+0.65`, crossed
95% interval `[-1.17,+2.54]`. Rank-8 paper STP changes top-1 by
`+1.37/+1.37` points at .02, `+1.17/+2.15` at .08, and `+1.95/+0.98` at
.12. The paper doses are unresolved, and paper versus released at their
selected r8/.02 setting is unresolved (`-0.20` points paper-minus-released,
`[-1.86,+1.37]`). Rank 128 does not improve either formulation's .02
treatment effect on the two tested seeds.

The new representation evidence supplies a stronger mechanistic conclusion.
STP reliably changes the geometry it targets, but “more STP-like” is not
monotonically “better generator”:

* released r8/.08 lowers a fixed released objective by `.297`, versus `.105`
  at r8/.02, but does not have the larger generation effect;
* released r128/.02 lowers it by `.255`, about 2.4 times the r8/.02 reduction,
  while its top-1 effect is only `+.49` points;
* harmful released seed 1301 lowers it by `.090`, within the favorable seeds'
  `.095--.129` range, and has final target CKA `.996` to Native;
* across 17 treatment checkpoints, released-loss and paper-loss changes have
  Spearman rho `-.097` and `-.152` with top-1 effect.

STP is active, Native ChemFM retains chemically structured bends, and there is
no gross STP-induced rank collapse. But objective fit, geometric straightness,
teacher-forced CE, and LoRA capacity do not determine exact free-running
generation. Early autoregressive path selection and multi-view candidate
ranking remain trajectory-noisy bottlenecks.

The 512-reaction panel is development data after repeated use. No finding here
is confirmation. The justified continuation is a locked three-arm r8/.02
confirmation on new seeds and an untouched panel—not another adaptive search.

## 1. Method fidelity and provenance

The paper is arXiv:2602.22617v1, *Semantic Tube Prediction: Beating LLM Data
Efficiency with JEPA*. The authoritative executable is
`galilai-group/llm-jepa@ea0017c654ad917066ff32afc88276bea8ca5f7e`.
PDF SHA-256 is
`bed646a5d7ab80c391a83d75535215bf85f9396e35506a24425bc3f126e773bc`;
source-tar SHA-256 is
`1f1218e0cee4afee1b40d257d03fb563917d9a1feaca538326687fc89ede6216`.

The paper equation and released experiment are not identical:

* **Released STP** directly ports `--linear=random_span`. It samples a
  non-full patch `[a,b)` over concatenated query/answer content, excludes
  framing tokens, and minimizes `1-cos(patch,before+after)`.
* **Paper STP** separately implements the literal `s<r<t` equation
  `1-cos(h_r-h_s,h_t-h_r)` on the same framing-excluded semantic path.

Released behavior is unchanged since upstream commit `313a7a7`; no upstream
revision supplied a direct three-index experiment. Tests cover released span,
loss, and all-gradient parity; paper-equation identity; boundary mapping; and
objective non-equivalence away from a straight path.

Official query content maps to reactant tokens and answer content to product
tokens in unchanged ChemFM serialization
`<rstart>reactants<eos><prostart>product<eos>`. Both conditions retain the
final post-RMSNorm states, one span per example, FP32 cosine, symmetric
gradients, native NTP, and total loss `L_NTP + lambda*L_STP`. Neither uses a
predictor, projection head, mask, EMA, stop-gradient, SIGReg, endpoint JEPA,
pair residual, or auxiliary dropout.

| Milestone | Commit/artifact |
|---|---|
| First official-STP implementation | `aaf6894a29c60a409c53b3d5390f66307757a9fe` |
| Report 08 experiment source | `acba1594f46356b2e097ad8ca8479a7df8a82a0a` |
| Report 08 final | `841f6e95635f88706b7fcdec14866e3c31932629` |
| Report 09 preregistration | `e58a45a1fe2622f8ae17e0c5cfca4813e5c5a560` |
| Report 09 experiment/analysis | `3b86ea70c2ad9663069ebe036487197e10326d35` |
| Official-rank analysis fixes | `8f8f1810`, `aef24562` |
| Report 09 completed repository | `717b076a183308127d279ca97d4faafdb23c499b` |
| Representation preregistration | `e1c5011` |
| Representation implementation | `49143710a6aa83e4f292d4520d8c0d4060ce5704`; exact outer-span correction `2a20459322e930f46250b11b4417cad7b82fbb04` |

ChemFM is pinned at revision
`f99dc2e89726539bb9cf31b2e2b43606650bac6a8`, weight SHA-256
`24686705d779db6876acc09c81d64d432262ef8b5dbfccc3852112587079ce419`.

## 2. Shared experimental system

| Setting | Value |
|---|---|
| Task | USPTO-MIT forward reaction prediction |
| Base | ChemFM-1B |
| Training data | frozen 1,280-row pilot |
| Budget | 4 epochs, 320 optimizer updates, epoch-4 endpoint |
| Batch | physical 4, accumulation 4, effective 16 |
| Optimizer | fused AdamW; LR `1e-4`; betas `.9/.999`; eps `1e-8`; WD `.01` |
| Schedule | cosine, 5% warmup, min LR `1e-5` |
| Precision | BF16 model, FP32 STP construction |
| LoRA | q/k/v/o and gate/up/down; dropout .1; embeddings/LM head saved |
| Rank | r8/alpha8 or r128/alpha128, so alpha/r=1 |
| Endpoint | five R-SMILES views, beam 10, official reciprocal-rank aggregation |
| Development panel | fixed 512 reactions, SHA-256 `a2e620...aa6057d` |

Every treatment uses a same-seed, same-rank Native control. Absolute r128
accuracy is not an STP effect. Report 07's 256 rows are the exact prefix of the
512 panel; Report 08 extended existing predictions without retraining.

## 3. Exact optimization record

STP needs only the final representation. Exact final-state capture eliminated
retention of all transformer layers during training.

| Rank | Reference step | Optimized | Speedup | Peak allocation | Tokens/s |
|---:|---:|---:|---:|---:|---:|
| 8 | .655873 s | .580892 s | 1.129x | 4.818 to 4.758 GB | 1,137 to 1,284 |
| 128 | .781460 s | .766025 s | 1.020x | 5.245 to 5.185 GB | 955 to 974 |

Identical batches produced exactly equal spans, losses, gradients, and updates.
Final-only checkpointing saved about 13.45 seconds and 3.6 GB per trajectory.
The four-worker evaluator already sustained 86--95% mean GPU utilization in
Report 08 and 98.0--98.8% on the later L40. Model loading was under .5% of
rank-128 endpoint time. Persistent workers, additional workers, dynamic
scheduling, and sampler rewrites did not clear the measured benefit/equivalence
bar; retained evaluation speedup was explicitly `1.000x`.

The final repository suite passes 114 tests with one expected skip.

## 4. Complete generation matrix

All values below are treatment-minus-same-rank-Native changes on 512 reactions.
Intervals are the staged crossed seed/reaction intervals, not confirmation
intervals.

| Condition | Seeds | Top-1 seed effects (pp) | Mean | Crossed 95% CI | Top-3/5/10 mean |
|---|---|---|---:|---:|---:|
| Released r8/.005 | 533,917 | +1.95,+.59 | +1.27 | [-.20,+2.93] | +2.25/+1.37/+3.61 |
| Released r8/.02 | 533,917,1301 | +2.15,+.98,-1.17 | +.65 | [-1.17,+2.54] | +1.37/+1.50/+2.15 |
| Released r8/.02 | 533,917 only | +2.15,+.98 | +1.56 | [+.20,+3.12] | screening subset |
| Released r8/.08 | 533,917 | +.78,+1.17 | +.98 | [-.20,+2.15] | +.49/+1.07/+1.17 |
| Released r128/.02 | 533,917 | -.39,+1.37 | +.49 | [-1.76,+2.73] | +.88/+1.17/-.59 |
| Paper r8/.02 | 533,917 | +1.37,+1.37 | +1.37 | [+.39,+2.54] | +2.05/+3.52/+2.83 |
| Paper r8/.08 | 533,917 | +1.17,+2.15 | +1.66 | [+.39,+3.12] | +3.61/+3.32/+2.64 |
| Paper r8/.12 | 533,917 | +1.95,+.98 | +1.46 | [+.20,+2.93] | +2.05/+3.52/+3.03 |
| Paper r128/.02 | 533,917 | +.59,-1.17 | -.29 | [-2.54,+1.95] | +.49/-.10/-.98 |

The locked decisions were applied without post-hoc additions:

1. rank 8 remained selected after released r128/.02 failed the .5-point,
   both-seeds-nonnegative rule;
2. released .02 remained selected because neither edge improved it by the
   one-point threshold;
3. paper .02 remained selected because .08 exceeded it by only .293 points,
   below the .5-point threshold; .12 did not trigger .16;
4. paper r128 was rejected at .02 because both r128-minus-r8 interactions were
   negative (`-.78`, `-2.54`);
5. formulation superiority required magnitude >=.5, same-signed seed effects,
   and an interval excluding zero. It was false.

No exact configuration was rerun. Conditional .16 and additional r128 paper
cells were not launched.

### View behavior

Released r8/.02 view effects for seeds 533, 917, and 1301 were respectively:

| Seed | View 1 | View 2 | View 3 | View 4 | View 5 |
|---:|---:|---:|---:|---:|---:|
| 533 | +.39 | +.98 | +1.17 | +1.95 | +1.56 |
| 917 | +.20 | +.59 | +1.95 | +.98 | +1.17 |
| 1301 | +.39 | 0 | -.78 | 0 | +.59 |

Paper r8 effects are distributed across views rather than confined to one:

| Lambda/seed | V1 | V2 | V3 | V4 | V5 |
|---|---:|---:|---:|---:|---:|
| .02/533 | +1.37 | +.98 | +1.37 | +1.56 | +1.56 |
| .02/917 | -.20 | -.20 | +.78 | +1.76 | +.78 |
| .08/533 | +1.17 | +.98 | +.98 | +2.73 | +.59 |
| .08/917 | +.78 | +1.37 | +1.56 | +2.54 | +2.34 |
| .12/533 | +.78 | +.78 | +.78 | +2.15 | +1.37 |
| .12/917 | +.39 | -.20 | +.59 | +.59 | +.98 |

Released r128-minus-r8 treatment effect is `-1.07` points at .02. Paper's is
`-1.66`. These are two-seed, formulation-specific .02 tests, not universal
capacity claims. At selected r8/.02, paper-minus-released effects are
`-.78/+.39`, mean `-.20`, interval `[-1.86,+1.37]`; no winner exists.

## 5. Teacher forcing and auxiliary pressure

| Condition | Seed | Top-1 pp | CE delta | Correct-token pp | Margin delta |
|---|---:|---:|---:|---:|---:|
| Released r8/.005 | 533/917 | +1.95/+.59 | -.00338/-.00389 | +.134/+.225 | +.278/+.347 |
| Released r8/.02 | 533/917/1301 | +2.15/+.98/-1.17 | -.00383/-.00255/-.00474 | +.147/+.134/+.056 | +.229/+.332/-.009 |
| Released r8/.08 | 533/917 | +.78/+1.17 | +.00362/-.00291 | +.020/+.152 | +.087/+.210 |
| Released r128/.02 | 533/917 | -.39/+1.37 | -.00641/-.00353 | -.020/+.106 | -.206/-.177 |
| Paper r8/.02 | 533/917 | +1.37/+1.37 | -.00168/-.00574 | +.076/+.234 | +.201/+.285 |
| Paper r8/.08 | 533/917 | +1.17/+2.15 | -.00258/-.00788 | +.103/+.307 | +.198/+.439 |
| Paper r8/.12 | 533/917 | +1.95/+.98 | -.00099/-.00846 | +.109/+.298 | +.210/+.282 |
| Paper r128/.02 | 533/917 | +.59/-1.17 | -.00920/-.00690 | -.052/-.025 | -.194/-.095 |

Released seed 1301 has its largest CE improvement but loses generation. Paper
r128/917 improves CE by `.00690` and loses `1.17` points. Paper's per-seed CE
ordering also fails to rank its lambdas. Teacher forcing verifies activity but
does not select the free-running generator.

Frozen initial-batch pressure differed substantially:

| Objective at .02 | Loss | Raw STP/NTP norm ratio | Weighted ratio | Gradient cosine |
|---|---:|---:|---:|---:|
| Released | 1.4735 | 2.252 | .0450 | -.397 |
| Paper | 1.4136 | .388 | .00776 | +.384 |

Paper used about 5.8 times less weighted pressure. At trained paper checkpoints,
weighted ratios were about `.041/.041` at r8/.02, `.136/.184` at .08,
`.194/.182` at .12, and `.033/.020` at r128/.02. Pressure rose with lambda,
but raw gradients shrank by .12; there is a flat noisy response, not a proven
collapse threshold.

## 6. Seed-1301 beam and aggregation mechanism

Seed 1301 falls from 20 to 14 top-1 reactions, while top-3/5/10 change by
`+.59/+.39/+.59` points. Seven reactions are Native-only top-1 and one is
STP-only. In all seven losses, gold remains in at least one STP view and in the
aggregate top 10:

* 4/7 are cross-view aggregation failures: gold tops a view but another
  candidate accumulates more reciprocal-rank score;
* 3/7 are within-beam ranking failures: gold remains returned but tops no view;
* 0/7 are beam-entry failures.

View-level gains usually occur on reactions that remain aggregate-wrong, while
losses concentrate among formerly aggregate-correct reactions. Candidate
diversity, view Jaccard, invalid rate, and consensus change only slightly.

Across the 35 view rows of the seven losses, teacher-forced correct-token rate
is exactly unchanged, token top-1 gains/losses are 7/7, margin changes `+.0033`,
and no correct-token rank changes by five. The directly observed failure is
early free-running trajectory and cross-view ordering, not gold disappearance
or systematic teacher-forced degradation.

## 7. All-checkpoint frozen representation supplement

### Protocol, scope, and integrity

The protocol was committed at `e1c5011` before inference. The exact executed
implementation is `2a20459`; it evaluates all 22 final checkpoints: five
Native controls and all 17 rank/formulation/lambda/seed treatments.

It uses the canonical first-256 development prefix (SHA-256
`250bc411...cef4fb32`) plus Report 06's fixed 64-reaction stereo supplement.
At all 23 representation depths it measures 320 reactions, 11,173 matched
chemical-event/control pairs, 64 semi-global anchors per event, and 32 fixed
spans per main reaction for each objective. Every checkpoint sees identical
tokens, events, controls, and spans.

The model is frozen in inference mode; no optimizer/backward exists; pre/post
parameter fingerprints match. Transformer states are BF16 and geometry is
FP32. The final RTX 4050 run took 1,608.8 seconds (26.8 minutes), peaked at 3.10 GB
allocated VRAM, and preserves 185.2 MB of compressed raw/derived output.
Parity tests prove exact released and paper calculations for identical spans.

Metrics include local curvature, literal semi-global alignment, both fixed STP
losses, activation/transition norms, path efficiency (displacement/path
length), effective and participation rank, spectral concentration,
mean-direction energy, source/product pairing and retrieval, and aligned
same-seed Native/STP CKA/displacement. Event effects are matched within reaction
and segment. BH covers all 3,910 event/treatment/layer tests.

### Native chemistry geometry

Final-layer Native event-minus-control means are:

| Rank | Event | Local | Semi-global |
|---:|---|---:|---:|
| 8 | ring | +.10814 | +.07218 |
| 8 | branch | +.07153 | +.00479 |
| 8 | stereo | -.23948 | -.02253 |
| 8 | motif | +.07206 | +.02992 |
| 8 | inferred center | +.04365 | -.01448 |
| 128 | ring | +.14759 | +.08750 |
| 128 | branch | +.07350 | +.00328 |
| 128 | stereo | -.22676 | -.02667 |
| 128 | motif | +.05721 | +.02548 |
| 128 | inferred center | +.03997 | -.01605 |

Ring and motif completions induce persistent bends; branch effects are mainly
local; inferred centers are not persistently positive; stereo tokens are
smoother than matched positions. The pattern survives fine-tuning, rank, and
seed. Vanilla local collinearity is not an adequate description of chemical
events.

### Objective response and trajectory geometry

Final-layer entries are STP-minus-Native means. Negative fixed loss indicates
successful fitting; positive efficiency indicates a straighter end-to-end
path.

| Condition | Top-1 pp | Released loss | Paper loss | Source curvature | Target curvature | Source/target efficiency |
|---|---:|---:|---:|---:|---:|---:|
| Released r8/.005 | +1.27 | -.0564 | -.0086 | +.00173 | +.00019 | +.00040/+.00132 |
| Released r8/.02 | +.65 | -.1046 | -.0168 | +.00493 | +.00171 | +.00096/+.00205 |
| Released r8/.08 | +.98 | -.2970 | -.0435 | +.00730 | +.00242 | +.00287/+.00536 |
| Released r128/.02 | +.49 | -.2554 | -.0376 | +.00984 | +.00539 | +.00311/+.00337 |
| Paper r8/.02 | +1.37 | -.0382 | -.0075 | +.00016 | -.00200 | +.00023/+.00068 |
| Paper r8/.08 | +1.66 | -.0572 | -.0183 | +.00161 | -.00409 | +.00068/+.00178 |
| Paper r8/.12 | +1.46 | -.1097 | -.0242 | +.00036 | -.00788 | +.00072/+.00187 |
| Paper r128/.02 | -.29 | -.0259 | -.0091 | +.00072 | -.00015 | +.00079/+0.00000 |

Both objectives learn their intended geometry. They cross-generalize: released
training lowers paper loss and paper training lowers released loss. Released
STP nevertheless raises immediate curvature while improving whole-path
efficiency; its patch/complement constraint is not local token-by-token
collinearity. Paper STP increasingly lowers target curvature, but this also
does not rank its generation results.

Greater capacity clearly expresses the auxiliary geometry. Released r128/.02
fits the released objective much more strongly than r8/.02. Thus the tested
r128 failure cannot be attributed to inability to realize STP.

### Non-monotonic depth

| Condition/layer | Released loss | Paper loss | Target efficiency | Target curvature | Target CKA |
|---|---:|---:|---:|---:|---:|
| Released r8/.02 L6 | -.006 | -.003 | +.00083 | -.00219 | .98697 |
| Released r8/.02 L16 | -.022 | -.004 | +.00077 | -.00348 | .98635 |
| Released r8/.02 L21 | +.076 | -.013 | +.00172 | -.00028 | .99468 |
| Released r8/.02 L22 | -.105 | -.017 | +.00205 | +.00171 | .99506 |
| Released r8/.08 L21 | +.058 | -.040 | +.00642 | -.00222 | .98849 |
| Released r8/.08 L22 | -.297 | -.044 | +.00536 | +.00242 | .98937 |
| Paper r8/.12 L16 | -.091 | -.006 | +.00058 | -.00666 | .98433 |
| Paper r8/.12 L21 | +.157 | -.017 | +.00129 | -.00613 | .99296 |
| Paper r8/.12 L22 | -.110 | -.024 | +.00187 | -.00788 | .99352 |
| Released r128/.02 L16 | +.208 | +.001 | -.00090 | +.00510 | .97979 |
| Released r128/.02 L22 | -.255 | -.038 | +.00337 | +.00539 | .98581 |

Intermediate released geometry can worsen, then improve sharply at the exact
final state where loss is applied. Final CKA can recover after larger early
reorganization: r128/.02 minimum source CKA is about `.49` near layers 3--4,
but final source CKA is `.970--.979`. STP induces distributed compensation,
not a rigid all-layer rotation.

### Chemical events remain structured

Final treatment changes in event-minus-control effects, averaged by config:

| Condition | Metric | Ring | Branch | Stereo | Motif | Center |
|---|---|---:|---:|---:|---:|---:|
| Released r8/.005 | local/semi | +.00764/+.00526 | -.00208/-.00101 | -.00344/-.00233 | +.00443/+.00264 | +.00769/+.00264 |
| Released r8/.02 | local/semi | +.00651/+.00455 | +.00463/-.00083 | -.00519/-.00153 | +.00794/+.00277 | +.00436/+.00222 |
| Released r8/.08 | local/semi | +.01510/+.01249 | +.02243/-.00060 | -.00582/-.00279 | +.00961/+.00965 | -.00435/+.00357 |
| Released r128/.02 | local/semi | -.00613/+.00587 | +.01041/-.00555 | -.01120/+.00085 | +.01568/+.00406 | -.00926/-.00107 |
| Paper r8/.02 | local/semi | +.00165/+.00169 | -.00423/-.00086 | -.00145/-.00147 | +.00326/+.00377 | +.00655/+.00097 |
| Paper r8/.08 | local/semi | +.00965/+.00080 | +.00717/-.00090 | -.00653/-.00199 | -.00170/-.00370 | -.00617/-.00014 |
| Paper r8/.12 | local/semi | +.01204/-.00065 | +.00422/-.00276 | -.00232/-.00259 | +.00215/+.00736 | -.00501/+.00160 |
| Paper r128/.02 | local/semi | -.00441/-.00488 | +.00414/-.00142 | -.01474/-.00202 | -.00025/-.00230 | -.01577/-.00225 |

BH detects 2,916/3,910 changes at q<.05 because thousands of fixed pairs make
small effects measurable. Magnitude and coherence matter more. Released STP
often increases ring/motif semi-global disruption and high-lambda branch local
curvature. Chemistry-conditioned bends coexist with more aligned whole paths;
they are not erased.

### Spectrum, relational coding, and drift

Native final pooled spaces are already anisotropic: r8 mean-direction energy
is about `.875--.877` for sources and `.830--.836` for targets; effective
ranks are about 19--20 of at most 255 centered sample dimensions. STP does not
cause gross new collapse:

| Condition | Pooled rank S/T delta | Transition rank S/T delta | Pairing-gap delta | Retrieval top-1 delta |
|---|---:|---:|---:|---:|
| Released r8/.005 | +.328/+.196 | +.031/-.266 | +.00185 | -.0254 |
| Released r8/.02 | +.234/+.312 | -.079/-.051 | +.00148 | -.0169 |
| Released r8/.08 | +.191/+.477 | -.156/-.183 | -.00253 | -.0273 |
| Released r128/.02 | -.129/+.505 | -.888/+.131 | -.00986 | -.0371 |
| Paper r8/.02 | +.056/-.267 | +.214/-.033 | +.00288 | -.0449 |
| Paper r8/.08 | +.034/-.120 | +.029/+.245 | +.00011 | -.0645 |
| Paper r8/.12 | -.010/-.968 | +.065/+.157 | +.00108 | -.0527 |
| Paper r128/.02 | -.408/-.215 | -.032/+.201 | -.00191 | -.0391 |

No STP condition improves this pooled source-to-product retrieval probe.
Released r128/.02 most clearly degrades pairing gap and increases target
anisotropy while fitting STP strongly. STP is a trajectory-shape constraint,
not a pair-discrimination objective.

Final same-seed drift is substantial despite high CKA:

| Condition | Source/target CKA | Source/target relative RMS displacement |
|---|---:|---:|
| Released r8/.005 | .99287/.99502 | .364/.430 |
| Released r8/.02 | .99191/.99506 | .352/.400 |
| Released r8/.08 | .98185/.98937 | .556/.563 |
| Released r128/.02 | .97011/.98581 | .602/.636 |
| Paper r8/.02 | .98979/.99484 | .367/.404 |
| Paper r8/.08 | .99180/.99391 | .364/.436 |
| Paper r8/.12 | .99080/.99352 | .382/.449 |
| Paper r128/.02 | .97916/.98493 | .448/.497 |

Higher lambda/rank generally allows more movement, especially for released
STP, without a larger generation benefit. The models are not capacity-starved
for representational change.

### Geometry does not select generation

| Final treatment metric | Pearson r | Spearman rho | Leave-one-config-out rho range |
|---|---:|---:|---:|
| Released fixed loss | +.008 | -.097 | [-.239,+.146] |
| Paper fixed loss | -.078 | -.152 | [-.289,+.022] |
| Source curvature | -.169 | -.260 | [-.439,-.156] |
| Target curvature | -.226 | -.276 | [-.362,-.171] |
| Source efficiency | -.022 | -.109 | [-.213,+.036] |
| Target efficiency | +.205 | +.303 | [+.056,+.467] |

These 17 points are dependent development results, so p-values are not causal
evidence. Reaction-level win/loss analyses on the first 256 are sparse and
reverse across seeds.

Seed 1301 is the strongest controlled counterexample:

| Seed | Top-1 | Released loss | Target efficiency | Target CKA | Target displacement |
|---:|---:|---:|---:|---:|---:|
| 533 | +2.15 | -.1286 | +.00214 | .99610 | .346 |
| 917 | +.98 | -.0949 | +.00239 | .99311 | .464 |
| 1301 | -1.17 | -.0904 | +.00163 | .99598 | .389 |

Its failure is not insufficient objective learning, excessive final drift,
rank collapse, or gold-beam loss. The directly evidenced mechanism is changed
early free-running choices and cross-view ranking.

### First-principles interpretation

1. Native ChemFM trajectories contain chemistry-specific bends; they are not
   locally straight tubes.
2. STP applies global final-state shape pressure. Released STP aligns a patch
   with its complement; paper STP aligns two adjacent sub-displacements.
3. The transformer realizes this through non-monotonic layer compensation.
4. Global alignment and local chemical curvature coexist.
5. Neither loss explicitly makes the correct source/product pair more
   discriminative; retrieval does not improve.
6. A few early autoregressive choices and cross-view candidate ranks determine
   exactness. Small state changes can improve average CE while promoting a
   wrong surviving sequence above gold.
7. Objective fit demonstrates an active intervention, but is not the limiting
   causal variable for generation. More pressure/capacity is not presumptively
   beneficial.

## 8. Runtime and artifact accounting

Report 07's three released trajectories trained concurrently on one A6000 in
about 26 minutes; each 256 endpoint took about 16 minutes. Report 08 paired
training took 555--704 seconds and each 512 endpoint 1,749--2,145 seconds.
Report 09 used one L40: paired training took 350--380 seconds; endpoint time
was about 1,400 seconds at r8 and 1,648--1,692 at r128. Evaluation dominated
wall time at high GPU utilization.

Report 09's compact archive is
`runs/stp_completion/stp_completion_l40_compact.tar.zst`, SHA-256
`fbaf3b44ffb4002ea7f0f7d481503cab2c58b101c68f76ceb22c2b7b823c8553`.
All Thunder instances, including a late orphan A6000, were deleted after local
integrity checks.

The frozen study preserves each checkpoint's compressed reaction/event arrays,
hash/fingerprint metadata, spectra, relationships, and drift under
`runs/stp_representation/frozen_all_checkpoints/`; derived JSON, CSV, and SVG
artifacts are under `runs/stp_representation/analysis/`.

## 9. What is established, falsified, and unresolved

Established:

1. Released and paper STP are faithful, distinct interventions.
2. Both persistently reshape the intended hidden geometry.
3. Rank-8 development effects are mostly positive, but released r8/.02 is not
   seed-uniform and neither formulation has confirmation.
4. r128/.02 does not improve either tested formulation's treatment effect.
5. Paper .02/.08/.12 and selected paper/released formulations remain
   unresolved within noise.
6. Better CE, margin, objective fit, path efficiency, or larger drift does not
   reliably imply better generation.
7. Native chemical bends survive STP; no gross representation collapse occurs.
8. Seed 1301 is a within-beam/cross-view ranking failure with gold preserved.

Falsified is the simple monotonic mechanism: more successful STP straightening
or more capacity to realize it should yield more accurate generation. Not
falsified is a small positive expected rank-8 STP effect.

Still unresolved are new-seed/untouched-panel efficacy, paper's apparent
two-seed consistency, formulation superiority, capacity outside .02, and all
claims about natural language, other budgets, full tuning, or architectures.

## 10. Single justified continuation

Run one locked confirmation:

* Native r8, released r8/.02, and paper r8/.02;
* at least four new paired seeds;
* a new prespecified untouched 512-reaction panel from the remaining test set;
* unchanged official five-view evaluation;
* prespecified two Native contrasts and a multiplicity-aware direct
  formulation contrast;
* no selection after outcomes.

If compute permits one STP arm, paper r8/.02 is most informative because it
lacks an independent third seed. This constraint must be declared in advance.

## 11. Reproducibility map

| Evidence | Path |
|---|---|
| Report 08 preregistration | `docs/reports/08_STP_CAPACITY_FORMULATION_LAMBDA_PREREGISTRATION.md` |
| Report 08 analysis/archive | `runs/stp_matrix/a6000/analysis.json`; `runs/stp_matrix/stp_matrix_a6000_compact.tar.zst` |
| Report 09 preregistration | `docs/reports/09_STP_COMPLETION_PREREGISTRATION.md` |
| Report 09 analysis/archive | `runs/stp_completion/a6000/analysis.json`; `runs/stp_completion/stp_completion_l40_compact.tar.zst` |
| Seed-1301 beam/token diagnoses | `runs/stp_completion/a6000/existing_diagnostics/` |
| Representation protocol | `docs/preregistrations/STP_REPRESENTATION_GEOMETRY_PROTOCOL.md` |
| Frozen extractor/analyzer | `src/stp_representation_analysis.py`; `scripts/analyze_stp_representations.py` |
| Raw frozen outputs | `runs/stp_representation/frozen_all_checkpoints/` |
| Derived analysis | `runs/stp_representation/analysis/analysis.json` |
| Final table/event tests | `runs/stp_representation/analysis/configuration_final_layer.csv`; `event_treatment_effects.csv` |
| Plots | `runs/stp_representation/analysis/*.svg` |

All positive, null, and adverse results are retained. Representation probes
remain mechanism diagnostics; generated exact top-1 remains the endpoint.
