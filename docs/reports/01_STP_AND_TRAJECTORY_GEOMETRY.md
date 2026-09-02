# Semantic Tube Prediction and trajectory geometry on ChemFM

## Executive measured summary

This is the authoritative report for the complete Semantic Tube Prediction
(STP) program formerly split across Reports 07, 08, and 09. It also contains
the no-training frozen base-ChemFM chemical-event assay formerly in Report 06
and the preregistered frozen representation study of every trained Native/STP
checkpoint.

Under the preregistered decision rules, the generation verdict is
**INCONCLUSIVE**. Rank-8 released STP at
lambda=.02 changes official five-view exact top-1 by `+2.15`, `+0.98`, and
`-1.17` percentage points across seeds 533/917/1301: mean `+0.65`, crossed
95% interval `[-1.17,+2.54]`. Rank-8 paper STP changes top-1 by
`+1.37/+1.37` points at .02, `+1.17/+2.15` at .08, and `+1.95/+0.98` at
.12. The paper doses were not separated by the preregistered selection
threshold, and paper versus released at their
selected r8/.02 setting is unresolved (`-0.20` points paper-minus-released,
`[-1.86,+1.37]`). At `.02`, the mean same-rank treatment effect was lower at
rank 128 than rank 8 for both formulations on the two tested seeds. No other
paper-STP rank-128 lambda was tested.

Every treatment lowered its corresponding fixed-span objective relative to
its matched Native checkpoint. Across the tested development checkpoints,
larger objective reduction did not order generated top-1:

* released r8/.08 lowers a fixed released objective by `.297`, versus `.105`
  at r8/.02, but does not have the larger generation effect;
* released r128/.02 lowers it by `.255`, about 2.4 times the r8/.02 reduction,
  while its top-1 effect is only `+.49` points;
* released seed 1301 lowers it by `.090`, within seeds 533/917's
  `.095--.129` range, and has final target CKA `.996` to Native;
* across 17 treatment checkpoints, released-loss and paper-loss changes have
  Spearman rho `-.097` and `-.152` with top-1 effect.

The measured rank statistics do not show a large uniform rank reduction under
STP. Native event/control curvature differences remain present in the trained
checkpoints. Teacher-forced CE, fixed-span loss, path-efficiency, and final
rank metrics did not reproduce the ordering of top-1 effects in this matrix.
For seed 1301 specifically, the saved beams locate the lost Native successes
in within-beam ranking and cross-view aggregation rather than beam entry.

The 512-reaction panel is development data after repeated use. No untouched-
panel confirmation was run.

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
| Representation implementation | `49143710a6aa83e4f292d4520d8c0d4060ce5704`; exact outer-span correction `2a20459322e930f46250b11b4417cad7b82fbb04`; final manifest analyzer `3325ddc` |

ChemFM is pinned at revision
`f99dc2e89726539bb9cf31b2e2b4360650bac6a8`, weight SHA-256
`24686705d779db6876acc09c81d64d432262ef8b5dbfccc385212587079ce419`.

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
r128/917 improves CE by `.00690` and loses `1.17` points. The per-seed CE
ordering differs from the top-1 ordering across the paper lambdas.

Frozen initial-batch pressure differed substantially:

| Objective at .02 | Loss | Raw STP/NTP norm ratio | Weighted ratio | Gradient cosine |
|---|---:|---:|---:|---:|
| Released | 1.4735 | 2.252 | .0450 | -.397 |
| Paper | 1.4136 | .388 | .00776 | +.384 |

Paper used about 5.8 times less weighted pressure. At trained paper checkpoints,
weighted ratios were about `.041/.041` at r8/.02, `.136/.184` at .08,
`.194/.182` at .12, and `.033/.020` at r128/.02. Pressure rose with lambda,
while raw gradients were lower at `.12` than `.08`. No collapse threshold was
tested.

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
and no correct-token rank changes by five. The saved outputs locate the seven
lost top-1 cases in free-running within-beam and cross-view ordering; gold does
not disappear from the aggregate top 10 in these cases.

## 7. Frozen base-ChemFM chemical-event geometry

### 7.1 Frozen model, sample, and serialization

This no-training assay asked whether chemically annotated SMILES positions
have different consecutive-token curvature and ordered-span alignment from
matched ordinary positions. ChemFM-1B revision
`f99dc2e89726539bb9cf31b2e2b4360650bac6a8` was loaded through
`src/chemfm.py`, put in evaluation mode, and frozen. No labels, loss, backward
pass, optimizer, or update were constructed. A sampled parameter fingerprint
was identical before and after inference:

```text
72e66b0397c1897a1ea1864e7420a6cb76f83cf7e25a7bc8e05d515054e2fc15
```

Every reaction used the maintained teacher-forced serialization:

```text
<rstart>{canonical source}<eos><prostart>{canonical product}<eos>
```

The main sample was the prespecified 256-reaction endpoint panel
`prespecified_stage1_256.jsonl`, SHA-256
`5b87bce1e75ed1ebf1a2a9091e0367aedaa8600a5621bc960352cf45b18e1865`.
That panel contains no `@`, `/`, or backslash stereochemical tokens. The stereo
stratum therefore used a separately fixed 64-reaction supplement: seed
`20260829`, sampled without replacement from canonical view zero of each
20-view reaction in the official USPTO-50K test file after restricting to
reactions with a stereo token. Supplement reactions contribute only stereo
pairs. The embedding output and every one of 22 transformer-block outputs were
measured. Fast-tokenizer offsets were checked against the exact IDs emitted by
the maintained `ReactionCollator`.

### 7.2 Event definitions and matching

- **Ring closure:** closing occurrence of a paired SMILES ring label.
- **Branch:** opening and closing parentheses, with subtype retained.
- **Stereochemistry:** `@`, `@@`, `/`, and backslash token positions.
- **Motif completion:** token containing the last serialized atom in an RDKit
  match to one of 14 disclosed SMARTS motifs: carbonyl, carboxyl, ester, amide,
  amine, alcohol/phenol, ether, nitrile, nitro, sulfonyl, phosphoryl,
  carbon-halogen, alkene, or alkyne.
- **Reaction center:** an approximate graph-difference label because endpoint
  strings have no atom maps. The source/target component pair with the largest
  element- and bond-order-aware MCS was selected; atoms on an unmatched or
  bond-order-changed frontier were marked. Median MCS coverage was `.769` of
  the selected target component. These are MCS-inferred, not atom-mapped,
  reaction centers.

Categories can overlap. Controls had none of the five event labels. Each event
was assignment-matched within reaction and source/product segment, prioritizing
normalized position, branch depth, component, and token length. Atom events
matched atom controls and syntax events matched syntax controls. All
`11,173/11,173` pairs had the same token class. Median control reuse within a
reaction/category/segment was one; maximum reuse was seven.

| Event | Matched positions | Reactions | Source/target positions |
|---|---:|---:|---:|
| Ring closure | 1,616 | 256 | 906/710 |
| Branch | 5,304 | 253 | 3,096/2,208 |
| Stereochemistry | 306 | 64 | 145/161 |
| Motif completion | 3,042 | 256 | 1,929/1,113 |
| MCS-inferred reaction center | 905 | 256 | 437/468 |

Branches comprise 2,652 opening and 2,652 closing parentheses. Stereo events
comprise 264 `@`, 35 `/`, and 7 backslash labels.

### 7.3 Geometry and inference

At every valid center token `r` and layer, local curvature was

```text
1 - cos(h_r-h_{r-1}, h_{r+1}-h_r).
```

For semi-global alignment, 64 independent left/right offsets were drawn for
every event/control pair, producing `s<r<t` and

```text
1 - cos(h_r-h_s, h_t-h_r).
```

The same offsets were applied to each event and its control. Spans were grouped
as adjacent (`t-s=2`), short (`3--8`), medium (`9--24`), and long (`>=25`).
Zero-norm increments were retained as missing. Layers 1--22 had all 11,173
local pairs and 715,072 anchors valid; the embedding layer had 9,446 local
pairs and 508,607 anchors valid.

The inferential unit was a reaction. Pair differences were averaged within
reaction and compared across reactions. Intervals use 5,000 reaction-cluster
bootstrap draws; two-sided p-values use 20,000 reaction-level sign flips.
Benjamini-Hochberg correction covers all 230 primary
metric/category/layer tests. Paired Cohen's `dz` is the reaction-level mean
difference divided by its reaction-level SD. Positive values mean the event is
less aligned or more curved than its matched control.

### 7.4 Final-layer distributions and effects

| Event | Local difference (95% CI) | dz | q | Semi-global difference (95% CI) | dz | q |
|---|---:|---:|---:|---:|---:|---:|
| Ring closure | +.0513 (+.0378,+.0652) | +.462 | <.0001 | +.0746 (+.0679,+.0812) | +1.392 | <.0001 |
| Branch | +.0387 (+.0271,+.0499) | +.424 | <.0001 | +.0030 (-.0001,+.0063) | +.116 | .0784 |
| Stereochemistry | -.2053 (-.2516,-.1561) | -1.073 | <.0001 | -.0161 (-.0226,-.0096) | -.602 | <.0001 |
| Motif completion | +.1018 (+.0869,+.1169) | +.832 | <.0001 | +.0362 (+.0319,+.0406) | +.975 | <.0001 |
| MCS reaction center | +.0724 (+.0488,+.0961) | +.369 | <.0001 | -.0095 (-.0178,-.0018) | -.146 | .0233 |

Final-layer event/control medians were respectively `1.503/1.411` local and
`1.517/1.447` semi-global for rings; `1.472/1.396` and `1.463/1.464` for
branches; `1.222/1.312` and `1.575/1.589` for stereo; `1.591/1.456` and
`1.484/1.451` for motifs; and `1.461/1.388` and `1.448/1.455` for inferred
centers. Full layer-wise means, SDs, quartiles, medians, reaction counts,
effects, intervals, p-values, and q-values are retained in the CSV and JSON
artifacts.

Final-layer semi-global event-minus-control differences by span were:

| Event | Adjacent | Short 3--8 | Medium 9--24 | Long >=25 (95% CI) |
|---|---:|---:|---:|---:|
| Ring closure | +.0736 | +.1069 | +.0813 | +.0719 (+.0647,+.0789) |
| Branch | +.0678 | +.0132 | +.0038 | +.0033 (-.0003,+.0066) |
| Stereochemistry | -.2004 | -.0602 | -.0253 | -.0145 (-.0221,-.0075) |
| Motif completion | +.1388 | +.0919 | +.0470 | +.0330 (+.0283,+.0376) |
| MCS reaction center | +.0631 | +.0132 | +.0004 | -.0107 (-.0187,-.0031) |

Adjacent bins have fewer reaction clusters because an exactly adjacent draw is
rare among the 64 sampled anchors; their intervals remain in `summary.json`.
The long bin includes all 256 main reactions or all 64 stereo reactions.

Layer zero is a lexical embedding baseline. From embedding to layer 22, the
event/control contrast changed by:

| Event | Depth-added local difference (95% CI) | Depth-added semi-global difference (95% CI) |
|---|---:|---:|
| Ring closure | +.4069 (+.3850,+.4270) | +.1821 (+.1728,+.1913) |
| Branch | +.2911 (+.2741,+.3079) | +.1410 (+.1358,+.1467) |
| Stereochemistry | -.5341 (-.6294,-.4278) | -.2580 (-.2836,-.2336) |
| Motif completion | +.1090 (+.0902,+.1285) | -.0161 (-.0255,-.0076) |
| MCS reaction center | +.0206 (-.0183,+.0594) | -.0346 (-.0457,-.0237) |

Ring semi-global contrast becomes positive by layer 8 and remains positive at
layer 22. Motif semi-global difference is positive at both embedding and final
layers while the depth-added difference is negative, so matching does not
remove its lexical baseline. Final source/target signs were concordant for
ring semi-global (`+.047/+.108`), motif semi-global (`+.040/+.033`), stereo
local (`-.213/-.211`), motif local (`+.110/+.088`), and center local
(`+.066/+.061`). Branch semi-global was `-.007/+.019`.

Within this assay, rings and motifs have positive local and long-span
differences; branches have a positive local difference and a long-span
interval containing zero; inferred centers have positive local and negative
long-span differences; stereo differences are negative. These estimates do
not establish a training consequence or a causal chemical interpretation.

### 7.5 Limits, execution, and artifacts

Controls match class and several positional properties, not exact token
identity or the complete neighboring-token tuple. Events can overlap and
controls can be reused. Ordered anchors may cross components or source/product
markers. Reaction centers are an MCS-frontier proxy. Stereo comes from a
separate enriched sample. Results do not extend beyond those estimands.

The run used an RTX 4050 Laptop GPU, BF16 SDPA, batch 8, PyTorch
`2.3.0+cu121`, Transformers `4.45.2`, RDKit `2024.09.1`, and SciPy `1.14.1`.
Peak CUDA allocation was 2,695,863,296 bytes. Annotation took 6.4 seconds,
inference 43.9 seconds, and the complete run 115.2 seconds.

Artifacts are under `runs/diagnostics/frozen_chemfm_stp_geometry/`:
`summary.json`, `layerwise_primary_tests.csv`, `pair_geometry.npz`,
`matched_pairs.jsonl`, `layerwise_distributions.svg`, `paired_effects.svg`,
`span_persistence.svg`, and `artifact_manifest.json`. The implementation is
`src/frozen_geometry.py`; focused tests are in
`tests/test_frozen_geometry.py`.

## 8. All-checkpoint frozen representation supplement

### Protocol, scope, and integrity

The protocol was committed at `e1c5011` before inference. The exact executed
implementation is `2a20459`; it evaluates all 22 final checkpoints: five
Native controls and all 17 rank/formulation/lambda/seed treatments.

It uses the canonical first-256 development prefix (SHA-256
`250bc411...cef4fb32`) plus Section 7's fixed 64-reaction stereo supplement.
At all 23 representation depths it measures 320 reactions, 11,173 matched
chemical-event/control pairs, 64 semi-global anchors per event, and 32 fixed
spans per main reaction for each objective. Every checkpoint sees identical
tokens, events, controls, and spans.

The model is frozen in inference mode; no optimizer/backward exists; pre/post
parameter fingerprints match. Transformer states are BF16 and geometry is
FP32. The final RTX 4050 run took 1,608.8 seconds (26.8 minutes), peaked at 3.10 GB
allocated VRAM, and preserves 185.2 MB of compressed raw/derived output.
Parity tests matched released and paper calculations for identical spans.

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

Ring and motif event/control differences are positive locally and
semi-globally; branch differences are primarily local; inferred-center
semi-global differences are negative; and stereo differences are negative.
These signs recur across the evaluated Native ranks and seeds. The estimates
are descriptive of the disclosed annotations and matching procedure.

### Objective response and trajectory geometry

Final-layer entries are STP-minus-Native means. Negative fixed loss means the
treatment checkpoint has lower fixed-span loss; positive efficiency means a
higher displacement-to-path-length ratio.

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

All treatment rows lower their corresponding fixed objective; released
training also lowers the paper loss and paper training lowers the released
loss in these final-layer measurements. Released STP rows have positive mean
local-curvature changes and positive path-efficiency changes. Paper-STP target
curvature becomes more negative as lambda rises from `.02` to `.12`. Neither
set of geometry values orders the reported generation effects.

Released r128/.02 lowers the fixed released loss by `.2554`, versus `.1046`
for released r8/.02. This rules out zero objective response at r128/.02; it
does not identify why its generation treatment effect was smaller.

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

The released fixed loss is positive at layer 21 and negative at layer 22 for
the displayed released conditions; the training loss is applied at layer 22.
For r128/.02, source CKA is about `.49` near layers 3--4 and `.970--.979` at
the final layer. These depth profiles are non-monotonic; the assay does not
assign a causal mechanism to them.

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

BH marks 2,916/3,910 changes at `q<.05`. The table reports magnitudes because
many effects are small despite the adjusted result. Ring/motif semi-global
differences remain nonzero in the final checkpoints while whole-path
efficiency also changes; neither measurement is a generation endpoint.

### Spectrum, relational coding, and drift

Native final pooled spaces are anisotropic under these metrics: r8 mean-direction energy
is about `.875--.877` for sources and `.830--.836` for targets; effective
ranks are about 19--20 of at most 255 centered sample dimensions. The table
reports the treatment-minus-Native changes:

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

Every displayed pooled source-to-product retrieval delta is negative. Released
r128/.02 has pairing-gap delta `-.00986` and target pooled-rank delta `+.505`;
the table does not establish why those values changed.

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

The largest displayed final displacements occur for released r8/.08 and
released r128/.02. Those two conditions do not have the largest mean top-1
treatment effects.

### Association with generation in the development matrix

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

Seed 1301 provides the following within-configuration contrast:

| Seed | Top-1 | Released loss | Target efficiency | Target CKA | Target displacement |
|---:|---:|---:|---:|---:|---:|
| 533 | +2.15 | -.1286 | +.00214 | .99610 | .346 |
| 917 | +.98 | -.0949 | +.00239 | .99311 | .464 |
| 1301 | -1.17 | -.0904 | +.00163 | .99598 | .389 |

Its fixed-loss reduction, final CKA, and displacement overlap the ranges of the
other two seeds, so those measurements do not distinguish its adverse top-1
effect. Gold remained in the aggregate top 10 for all seven Native-only losses;
the observed losses were three within-beam ranking and four cross-view
aggregation changes.

### Descriptive synthesis

1. Native event/control geometry differs by chemical-event category under the
   fixed annotation and matching assay.
2. Released STP aligns a sampled patch displacement with its complement;
   paper STP aligns adjacent sub-displacements for sampled `s<r<t`.
3. Layer-wise fixed-loss and CKA profiles are non-monotonic.
4. Fixed-span alignment changes and local event/control differences coexist in
   the final states.
5. Pooled retrieval deltas are negative for all displayed treatments.
6. In seed 1301, all seven lost top-1 gold products survive in the aggregate
   top 10; three lose within-view rank and four lose at aggregation.
7. The reported development-matrix associations do not identify a scalar
   representation diagnostic that predicts treatment effect.

## 9. Runtime and artifact accounting

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

## 10. Measurements supported by the completed design

1. Released and paper STP implementations are numerically distinct and passed
   their recorded upstream/equation parity tests.
2. Every treatment lowers its matched fixed objective at the final layer.
3. Most rank-8 seed effects are positive, but released r8/.02 has one negative
   seed and its crossed interval includes zero.
4. For both formulations at `.02`, the two-seed mean treatment effect is lower
   at r128 than r8. Other paper r128 lambdas were not tested.
5. The preregistered paper-lambda and formulation-comparison thresholds were
   not met; the compared intervals do not separate those choices.
6. CE, margin, fixed objective, path-efficiency, drift, and rank do not share a
   monotonic ordering with the reported configuration means.
7. The recorded event/control differences remain present after STP, and the
   pooled-rank changes are small relative to their approximately 19--20 Native
   effective ranks.
8. Seed 1301's seven Native-only top-1 cases retain gold in the STP aggregate
   top 10 and split into three within-beam and four aggregation losses.

These data are inconsistent with a monotonic rule that more fixed-objective
reduction or rank capacity necessarily gives a larger generation effect in
this matrix. They do not estimate the effect on new training seeds and an
untouched panel. Formulation superiority, paper-STP capacity away from `.02`,
other budgets, natural-language tasks, and other architectures remain outside
the completed design.

## 11. Reproducibility map

| Evidence | Path |
|---|---|
| Report 08 preregistration | `runs/stp_matrix/a6000/preregistration.md`; `runs/stp_matrix/a6000/stage_*/preregistration.json` |
| Report 08 analysis/archive | `runs/stp_matrix/a6000/analysis.json`; `runs/stp_matrix/stp_matrix_a6000_compact.tar.zst` |
| Report 09 preregistration | `runs/stp_completion/a6000/required/preregistration.json` |
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
