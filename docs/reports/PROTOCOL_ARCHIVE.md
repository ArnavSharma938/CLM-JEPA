# Archived scientific protocols

Historical protocol text retained for provenance, not authorization to run experiments.
Original paths were under docs/preregistrations/. Hashes identify original files before consolidation.

## GEODESIC_MECHANISM_AUDIT_PROTOCOL.md

Original SHA-256: `8dbca0a25ace30a7642c5773e36c9b0f25271b9d156ec72430a71c329f055f49`.

# Geodesic Mechanism Audit: locked protocol

Status: preregistered before any new checkpoint inference or outcome analysis.

Repository state at protocol lock: `3e1d87197529dd2b51dcd53c8193b73fdd85afd9`.
This is a frozen, falsification-oriented mechanism audit.  It trains no model,
does not alter official generation, and does not select a new STP condition.

## Questions and decision criteria

The audit tests whether the ambient-Euclidean straightness optimized by STP is
a defensible proxy for intrinsic or behaviorally meaningful geodesicity.  The
Geodesic Hypothesis receives joint support only if the prespecified results
show: (i) a reproducible finite local-linear regime, (ii) reduced tube radius
inside that regime, (iii) corresponding improvement in output Fisher geometry,
(iv) little gold-token sensitivity to the nominally perpendicular component,
and (v) lower geodesic violation for gold than wrong beam trajectories.  A
failure of any component is reported separately; no composite significance
claim or post-hoc threshold determines success.

Primary contrasts are paired within reaction and seed.  Confidence intervals
use reaction-cluster bootstrap resampling (10,000 replicates, seed 20260902).
P-values, where useful, use paired sign-flip permutation tests and are reported
with effect sizes; this exploratory audit does not use them as a discovery
gate.  Results are descriptive for the repeatedly used development panel.

## Frozen inputs

The gold-trajectory panel is the existing prespecified 256-reaction manifest:
`data/clm_jepa_uspto_mit_official_endpoint/prespecified_stage1_256.jsonl`.
Gold-versus-wrong analyses use existing ordered five-view beam files from the
512-reaction development panel; no candidates are regenerated.  The seed-1301
seven Native-only aggregate successes in the archived beam diagnostic form a
prespecified natural-experiment stratum.

The complete 22-checkpoint census and exact paths are inherited unchanged from
`src/stp_representation_analysis.py`.  To bound GPU cost without selecting on
outcomes, work is divided as follows:

* **Primary matched trajectories (full 256):** rank-8 Native/released-STP at
  lambda .02 for seeds 533, 917, and 1301; rank-8 Native/paper-STP at lambda
  .02 for seeds 533 and 917.  Modules A, C, D, released-objective anatomy, and
  matched Native-to-STP displacement use this panel.
* **Capacity/strength robustness (fixed first 64):** every one of the 22 final
  checkpoints.  Module A summaries, tangent/acceleration summaries, and the
  relevant formulation anatomy are computed here.  This panel is fixed by
  manifest order, not checkpoint behavior.
* **Intrinsic-manifold robustness:** primary rank-8 seed 533 Native/released/
  paper plus rank-128 seed 533 Native/released/paper, and the seed-1301
  Native/released pair.  Queries are a fixed hash-selected set of at most 2,048
  valid interior transitions per layer from the 256 reactions.  The reference
  cloud excludes all states from the query reaction.
* **Gold-versus-wrong trajectories:** all 256 reactions common to the gold
  panel and archived five-view beams for rank-8 lambda-.02 Native/released
  seeds 533, 917, 1301 and Native/paper seeds 533, 917.  Per reaction/view the
  candidates are gold, aggregate top-1, highest-ranked wrong candidate, and up
  to three additional ordered aggregate-beam candidates.  Duplicate canonical
  strings are evaluated once per source serialization.  The seed-1301 seven
  losses additionally use every distinct candidate in the aggregate top 10.
* **Inference cone:** 64 hash-selected product-prefix positions stratified by
  ordinary/event positions, for seed-1301 Native/released and seed-533 Native/
  paper.  Use K=10 next tokens and probability-weighted rollouts through
  horizons 1--5.  This bounded module is secondary to A/C/D/E.

## Representation depths

Primary analyses use embedding output, layers 6, 16, 21, and the final
post-RMSNorm state.  On a fixed 64-reaction subset, hooks additionally capture
the input to the last transformer block, the last-block output before final
RMSNorm, and the final post-RMSNorm state.  Existing all-23-depth Report 01
statistics remain the broad depth census; this audit does not repeat them.

## Module A: exact tube scale-space and persistence

For every contiguous outer span `[s,t]` of each source, product, and
source-to-product-crossing semantic trajectory, and for every interior `r`,
compute the chord coefficient `alpha`, normalized perpendicular distance
`rho=||q||/||h_t-h_s||`, mean/RMS/max/p90/p95 radius, fractions above
`.05,.10,.20,.50`, and alpha outside `[0,1]`.  Report every feasible integer
span length L from 2 through the maximum observed length; zero-length chords
are flagged rather than divided.

Estimate persistence scale independently for segment, layer, and checkpoint
using a two-line continuous piecewise regression over log median-RMS-radius,
with candidate breakpoints requiring at least four L values on each side.
Report the breakpoint and bootstrap interval, not a binary claim that it is
the theorem's unknown tau.  Also report tangent autocorrelation C(k), k=1..32,
multiscale turning angles for k=1..min(32,boundary), optimal-ray residual by
horizon, and tangential/normal acceleration relative to the path velocity.

## Module B: intrinsic versus extrinsic curvature

Estimate local tangent spaces using states from other reactions only.  Repeat
with k=32,64,128 neighbors; tangent dimensions 8,16,32 (capped by k-1); raw
Euclidean and globally whitened neighbor search; and same-segment-only versus
pooled-segment reference clouds.  Decompose acceleration into tangent and
ambient-normal components and define geodesic violation as the component of
tangent acceleration perpendicular to projected velocity.  Conclusions must
be stable in sign across neighbor counts, dimensions, and both search metrics;
otherwise this module is inconclusive.

## Module C: decoder and Fisher--Rao geometry

At the final layer compute logits and categorical probabilities using the exact
checkpoint LM head.  Evaluate displacement norms in the local decoder Fisher
metric without materializing G.  In probability space use
`d_FR(p,q)=2*acos(sum_i sqrt(p_i*q_i))`, triple triangle excess, and endpoint
distance divided by summed adjacent distance.  Compare matched changes in
hidden tube/ray metrics against Fisher metrics.

## Module D: signal/noise test and intervention

For fixed deterministic triples at every feasible span scale, decompose the
middle displacement into chord-parallel and perpendicular parts.  Compute the
analytic final-state gold log-probability gradient and its signed and cosine
sensitivity to both components.  Then replace `h_r` by
`h_r-gamma*u_perp`, gamma in `.1,.25,.5`, through the unchanged LM head, both
raw and after restoring the original hidden-state norm.  Record gold log
probability/rank/margin, entropy, and top-k identity changes.  This noncausal
analysis tests the representation decomposition, not an inference method.

## Module E: gold versus wrong trajectories

Teacher-force archived candidate token paths with their original reaction view
and compare exact tube scale curves, paper and released losses, tangent
persistence, Euclidean and Fisher path excess, optimal-ray residual, and the
intrinsic estimate.  Primary effects are within reaction/view
`G(wrong)-G(gold)`, stratified by model correctness.  For the seven seed-1301
losses report the Native and released-STP separation and its change; geometry
is not used to rescore beams.

## Module F: released anatomy and inference cone

For released spans report cos(P,B), cos(P,A), cos(B,A), and
`kappa=||B+A||/(||B||+||A||)` alongside the released objective.  Matched state
displacement is decomposed relative to the Native chord, including
`cos(delta_h_r,-q_native)`, endpoint/chord rotation and magnitude change, and
endpoint-to-middle displacement ratio.

The inference-cone module measures probability-weighted angular spread,
perpendicular variance, Fisher dispersion, and axis rotation relative to the
gold continuation across horizons 1--5.  It does not change official decoding.

## Compute and preservation

Use one Thunder Compute A6000, 6 vCPU, and 100 GB disk, with no template.  A
single checkpoint load feeds batched hidden-state extraction and all invariant
preprocessing is cached.  Geometry uses FP32; transformer inference remains
BF16.  Cache shards are content-addressed and resumable.  Retain compact raw
tables, metadata, plots, tests, and logs; remove downloaded base weights and
disposable tensor caches after verified archive transfer.  Record wall time,
peak VRAM, throughput, package/CUDA versions, checkpoint hashes, input hashes,
and the exact execution commit.


## LATENT_PREDICTABILITY_DECODER_COUPLING_PROTOCOL.md

Original SHA-256: `68bb36e87c09973e758ed10f7fa055b15f9f2fefb7848bb1547174e9a7560065`.

# Frozen Latent Predictability, Decoder Coupling, and Chemical-View Audit

Status: frozen without access to untouched-confirmation outcomes.

The locked reaction split is stored at
`data/clm_jepa_uspto_mit_latent_audit/splits.json`, SHA-256
`315a5012d79f8704b3119b03e3bfdc96f9618f189aa5ca1d1a602cfeed9ff560`.

This audit is a frozen diagnostic. It trains no ChemFM checkpoint, changes no
generation result, and cannot select the arms, seeds, panel, stopping rule, or
interpretation of the independent untouched-panel confirmation.

## Frozen checkpoints and isolation

The checkpoint census is exactly rank-8 Native and released STP lambda .02 at
seeds 533, 917, and 1301, plus paper STP lambda .02 at seeds 533 and 917. The
canonical paths, rank/alpha checks, and same-seed pairings come from
`src/stp_representation_analysis.py`. Models are loaded in evaluation mode;
all parameters are frozen; no optimizer or backward pass involving ChemFM is
allowed. Transformer inference is BF16 and probe targets/statistics are FP32.

Before any expensive stage, the runner requires the untouched confirmation
manifest, records its SHA-256, and aborts if its reaction identities overlap
any audit reaction. It reads no confirmation predictions or outcomes. Audit
splits and hyperparameters are unaffected by the confirmation track.

## Reaction-disjoint probe data

The input is the frozen 1,024-identity USPTO-MIT validation manifest. Sort
unique identities by `sha256("latent-decoder-audit-v1|" + identity)` and assign
the first 640 to probe train, next 192 to validation, and final 192 to test.
Split before token positions or alternate serializations are expanded. Every
view and candidate from one reaction stays in that reaction's split.

The previously used 512-reaction official-test development panel is an
external application set only. It is never used to fit a PCA, standardizer,
probe, regularization coefficient, early-stopping decision, or support rule.
The seven seed-1301 losses are explicitly exploratory.

## Forecasting task and probes

At layers 6, 16, 21, and final post-RMSNorm, predict `h[t+k]` for
`k in {1,2,4,8}` separately within source and product. Product is primary.
Compare `h[t]` with the raw short history
`concat(h[t-3],h[t-2],h[t-1],h[t])`. Their primary comparison uses the common
history-eligible set. A matched-horizon analysis uses only starts eligible at
`k=8`.

Every checkpoint/cell receives:

1. the train-target mean;
2. an L2-regularized linear predictor;
3. that frozen linear prediction plus a width-128, two-layer GELU residual.

A train-only fixed-rank 256 PCA compresses the 2,048-dimensional target for
probe fitting. Predictions are reconstructed and every scientific metric is
computed in full hidden space. PCA coverage is reported. Training positions
are reaction-balanced and capped at 16,384 per segment/cell. Fixed ridge and
MLP regularization coefficients are shared across checkpoints; early stopping
uses validation only. PCA and ridge are deterministic. Three MLP seeds are
used for final-layer product cells and one elsewhere.

Report per-dimension MSE, target-variance-normalized MSE, raw and centered
cosine, and untruncated `R2=1-SSE/SST`. Predictable fraction means this
untruncated R2; negative values are retained. Nonlinear improvement is
MLP-minus-ridge R2. Primary reductions average positions within reaction and
then reactions; token-weighted values are secondary.

## Decoder-preserved predictability

At final post-norm, project true and predicted states through the checkpoint's
unchanged saved LM head. At layers 6, 16, and 21, substitute the predicted
future state and process it through the remaining frozen blocks. Reuse cached
prefix K/V because causal prefix states are unchanged. True-state injection
must reproduce full-forward logits before predicted-state results are valid.

Report `KL(true || predicted)`, JS, actual next teacher-token log probability
and probability, rank, margin against the best other token, top-1 agreement,
and top-5/top-10 overlap. Final-layer metrics use all test positions.

**Runtime amendment made before any decoder metric existed.** Intermediate
block injection is an exact but deliberately representative diagnostic rather
than an exhaustive census. It uses the same 64 audit-test reactions selected
only by a fixed identity hash, a reaction-balanced base cap of 96 product and
48 source positions per cell, and up to 32 reaction-balanced positions for
each sparse support (event completion, component boundary, and reaction-center
window). The latent probes and final-layer decoder analysis remain exhaustive
over all audit-test positions. This amendment replaces the impractical
4,096/2,048-plus-all-rare replay plan; it was locked after profiling exposed
full-matrix materialization and suffix replay—not model inference or probe
fitting—as the bottleneck, and before inspecting any decoder result.
Because BF16 SDPA uses different kernels for the original batched extraction
and one-position suffix replay, parity is gated on RMS error <=0.02 and cosine
>=0.999 rather than an outlier-sensitive absolute element threshold. Maximum
absolute error and LM-head top-1 mismatches are still recorded without hiding
them; a near-tied argmax mismatch alone is not treated as state-replay failure.

## Semantic supports

The same locked probes are evaluated on:

1. arbitrary valid content positions;
2. atom-bearing to future atom-bearing pairs;
3. an annotated event completion to the next annotated event completion,
   with no intervening event;
4. the last pre-component-boundary state to the first post-boundary atom;
5. positions within two serialized tokens of existing MCS-inferred
   reaction-center annotations.

Ring, branch, stereo, motif, and reaction-center labels reuse
`src/frozen_geometry.py`. Full-product graph information may define a held-out
evaluation stratum but is never a probe feature. Sparse cell counts and
missing horizons are reported without pooling them post hoc.

## Archived gold and wrong trajectories

Reuse archived five-view beam rows and the existing candidate workload logic.
Per eligible reaction/view replay gold, view top-1, highest-ranked wrong, and
already retained robustness candidates. No candidates are generated or
rescored. Locked validation-trained probes consume each candidate's own causal
past.

Primary separations are `error(wrong)-error(gold)` for normalized latent MSE
and decoder KL/JS. For wrong paths, next-token metrics refer to that path's
actual teacher-forced token. Control candidate length with same-horizon paired
comparisons, normalized-position strata, linear length adjustment, and a
near-equal-token-length subset.

For the seven seed-1301 Native-only successes, compare Native and released STP
gold, promoted-wrong, horizon, decoder, and support effects. The four known
aggregation failures are reported individually. This stratum is exploratory.

## Cross-view chemical invariance

On audit-test reactions create the canonical serialization and four
deterministically seeded, validity-checked RDKit randomized serializations,
including component-order randomization. Align through molecular-graph
correspondence, never token position. Pool tokenizer pieces corresponding to
matched atoms, motif atom sets, and unambiguous components. Ambiguous repeated
components are marked or assigned by graph-isomorphism matching.

At every selected layer report within-identity view variation,
between-identity centroid variation, their ratio, matched-view cosine,
centered CKA, and cross-view identity retrieval. Between-identity controls
match atom element, aromaticity, charge, degree, segment, normalized position,
and reaction-length bin.

For coupling to actual generation, teacher-force the canonical gold product
after each of the five archived source views and join latent agreement with
candidate Jaccard, per-view gold rank, reciprocal-rank aggregate score,
aggregate correctness, within-view failures, and aggregation failures.

## Joint analysis

At aligned held-out atoms and motifs, combine forecast R2/error, true and
predicted cross-view invariance, and decoder divergence/agreement. Validation-
fixed tertiles define a predictability-by-invariance table. A strict subset is
defined on validation as positive R2, within/between below one, and decoder
agreement better than the constant probe; prevalence and chemical composition
are then reported only on test.

## Controls, inference, and integrity

- Derange complete future-state donor reactions within split, prohibiting
  self-matches and matching segment, horizon, length, position, and token class.
- Replace suffix tokens after `t` and assert every `h[<=t]` and probe prediction
  is unchanged within recorded BF16 tolerance.
- Assert no input index exceeds `t`; scalers/PCA/probes see train only.
- Use identical positions, supports, views, and shuffled assignments across
  same-seed Native/STP checkpoints.
- Preserve model/tokenizer/checkpoint/input hashes, package versions, wall
  time, peak VRAM, and pre/post frozen-parameter fingerprints.
- Use reaction-cluster bootstrap intervals with positions and views nested in
  reaction. Display all seeds. Apply BH within prespecified
  layer-by-horizon-by-support families while retaining raw effects.

The audit may motivate a subsequent JEPA objective only if information is
jointly causal-context predictable, serialization invariant, and decoder
functional. No new JEPA formulation is trained in this run.


## STP_REPRESENTATION_GEOMETRY_PROTOCOL.md

Original SHA-256: `84d01212afab515311b283a3775108e153f7e342321de4a1a07e6efe1a87f201`.

# Frozen STP checkpoint representation study: locked protocol

Status: preregistered before checkpoint inference.

Repository state at protocol lock: `717b076a183308127d279ca97d4faafdb23c499b`.
This is a frozen, exploratory mechanism study. It trains no model and cannot
replace the prespecified generation endpoint in Reports 07--09.

## Scope and checkpoint census

The study includes every final epoch-4 Native or STP checkpoint used by the
Report 07--09 experiment family: 22 checkpoints in total.

| rank | formulation | lambda | seeds | checkpoints |
|---:|---|---:|---|---:|
| 8 | Native | 0 | 533, 917, 1301 | 3 |
| 8 | released STP | .005 | 533, 917 | 2 |
| 8 | released STP | .02 | 533, 917, 1301 | 3 |
| 8 | released STP | .08 | 533, 917 | 2 |
| 8 | paper STP | .02 | 533, 917 | 2 |
| 8 | paper STP | .08 | 533, 917 | 2 |
| 8 | paper STP | .12 | 533, 917 | 2 |
| 128 | Native | 0 | 533, 917 | 2 |
| 128 | released STP | .02 | 533, 917 | 2 |
| 128 | paper STP | .02 | 533, 917 | 2 |

The Report 05 residual-JEPA checkpoints are outside this STP checkpoint
family. No checkpoint will be selected or omitted using its representation
result.

## Frozen data and serialization

The main sample is the exact 256-unique-reaction development manifest used by
the generation experiments:
`data/clm_jepa_uspto_mit_official_endpoint/prespecified_stage1_256.jsonl`
(SHA-256
`250bc411efa06ac543cf5bd037b166fc4d48e89562401dfa148dbbb2cef4fb32`).
Its canonical view is serialized exactly as
`<REACTANT>{source}<eos><PRODUCT>{target}<eos>`.

For stereochemistry event coverage only, use the same deterministic 64-reaction
supplement, selection seed `20260829`, and source file as Report 06:
`data/uspto_50k/test_r_smiles.csv` (SHA-256
`ea0d90b44018314392af149de540507cac2bb66c4f6805f7239910d57195b39b`).
The supplement is excluded from source--product retrieval, spectra, and links
to generation outcomes.

All inference is frozen (`eval`, inference mode, no optimizer, no loss
backward), BF16 in the transformer and FP32 for geometry/statistics. Event
labels, matched ordinary controls, and semi-global anchors are exactly those
of `src/frozen_geometry.py`, including the same matching seed and 64 anchors
per event. Every checkpoint receives the same examples, matches, and spans.

## Locked measurements

### 1. Token trajectory geometry at all 23 representation depths

For ring closures, branches, stereochemistry, functional-group/motif
completions, and inferred reaction-center events, measure the existing exact
quantities:

* local curvature, `1-cos(h_t-h_(t-1), h_(t+1)-h_t)`;
* semi-global alignment disruption, `1-cos(h_r-h_s, h_t-h_r)`, split into
  adjacent, 3--8, 9--24, and 25+ token outer-span bins;
* event-minus-within-reaction matched-control effects.

Also measure whole-source and whole-product activation norm, transition norm,
mean local curvature, end-to-end displacement/path-length ratio, and fixed-span
released and paper STP losses per reaction and layer. The diagnostic STP span
set is sampled once from a fixed independent seed, is held constant across all
checkpoints, and is not claimed to reproduce any trajectory's training RNG.
Given a fixed span, its released patch-versus-complement and literal paper
three-point calculations must exactly match the implementations in
`src/stp.py`.

### 2. Representation-space structure

At every layer and separately for source/product pooled states and sampled
source/product token-transition vectors, report:

* activation/transition variance and norm;
* covariance effective rank and participation ratio;
* leading-eigenvalue energy and top-8 cumulative energy;
* mean-direction energy (anisotropy);
* source--product true-pair cosine, matched-shuffle gap, retrieval top-1 and
  mean reciprocal rank.

These are descriptive frozen probes, not measures of generation quality.

### 3. Treatment-induced representation drift

For every STP checkpoint, compare aligned main-panel states with its same-seed,
same-rank Native checkpoint at every layer. Report centered linear CKA,
aligned-state cosine, relative RMS displacement, displacement effective rank,
and changes in event/control geometry. Rank effects are differences of
treatment effects; absolute rank-8/rank-128 differences are not evidence that
capacity changes STP.

### 4. Links to trained objectives and generation

Join the frozen measures to the already archived per-seed five-view generation
and teacher-forced outcomes. Report configuration/seed associations only as
descriptive, leave-one-configuration-out sensitivity where feasible, and
reaction-level associations between geometry change and paired exact-generation
wins/losses. With only two or three training seeds, no correlation is to be
presented as causal or confirmatory.

## Inference and multiplicity

Checkpoint-level event effects use reaction-cluster means and paired bootstrap
intervals. Treatment-minus-Native geometry effects use the same reaction,
event/control match, seed, and rank. Layer/event families receive global
Benjamini--Hochberg adjustment; both raw effect sizes and adjusted values are
retained. Seed-level treatment summaries show every seed and use no
large-sample significance claim.

The primary mechanistic contrasts, fixed before inference, are:

1. whether STP reduces its own frozen objective and increases path straightness;
2. whether any reduction is semi-global rather than merely local;
3. whether it is source-, product-, or reaction-event localized;
4. whether released and paper STP make distinguishable geometric changes at
   comparable lambda;
5. whether rank 128 changes the STP-minus-Native representation effect;
6. whether stronger geometric straightening tracks, fails to track, or
   anti-tracks generated exact top-1 across the completed matrix.

No representation result changes which checkpoints are included. All null and
adverse findings will be preserved. The final write-up will consolidate Reports
07--09 into one authoritative report; the former reports will become concise
provenance pointers so their duplicated narrative cannot be mistaken for
independent evidence.


## STP_UNTOUCHED_CONFIRMATION_PROTOCOL.md

Original SHA-256: `8083f67cba085ddccc4ff973cce6e17d5aa64e23816ce9361c3924143a4fb0a7`.

# Untouched-panel ChemFM STP confirmation protocol

**Frozen:** 2026-09-04, before any new checkpoint training or inference.  The
repository state immediately before this protocol was `57bdcd1`.  The JSON
protocol in `data/clm_jepa_uspto_mit_stp_confirmation/preregistration.json` is
the machine-readable authority.

**Outcome-blind size amendment:** after the six primary-seed trajectories had
trained but before any confirmation-panel inference or outcome inspection, the
user reduced the endpoint from 1,280 to 640 reactions. The retained panel is
exactly rows 0--639 of the original panel's already-locked salted SHA-256
selection order; checkpoint behavior played no role. The original panel and
hash remain retained for provenance. No inference had begun when this
amendment was committed.

## Panel and independence

The amended endpoint contains 640 unique official USPTO-MIT test reactions and five
official R-SMILES views per reaction.  Its SHA-256 is
`3655e58404c3509c04b15cd4ffcdf15723f9be62e76c48c676ccb7decf9e2945`.
It is an exact prefix of the original 1,280-reaction manifest at SHA-256
`17aba3335a60985580b77cb9e89947f1d26e2d7164bf4a9115817e9212477ac7`.
Selection was outcome-blind from the hydrated official test object at SHA-256
`c2f4a3b731c4ed0a35b1c38fbff9563aee0e61064bcedeca555f335f69964945`.

All 3,300 reactions ever frozen for the prior official endpoint, the separate
24-reaction equivalence panel, the 256 training reaction identities, and the
1,024/256 validation and probe identities were excluded by canonical directed
source-product pair.  The exclusion-ledger SHA-256 is
`85a945af68e02a5970227ae21726e3af5ef2c1c5632962daded9cedef6231fb8`.
The panel has 640 distinct official groups, 640 distinct chemical pairs,
3,200 distinct example IDs, and zero overlap with either exclusion namespace.

The frozen Latent Predictability audit uses validation and already-developed
beam data only.  Its splits and code are locked before confirmation outcomes
are opened.  It must abort on confirmation-panel overlap and cannot alter this
experiment's arms, seeds, stopping, or endpoint.

## Arms and training

For each paired seed, train exactly three rank-8/alpha-8 arms: Native,
released STP at lambda .02, and paper-equation STP at lambda .02.  Preserve
the existing 1,280-row (256 reaction x five-view) training CSV, four epochs,
320 optimizer steps, paired order, BF16 model/FP32 STP reduction, LoRA targets
and saved modules, fused AdamW, LR/scheduler, dropout, and serialization from
the completed STP experiment.  No existing configuration is substituted.

The first seed pair is `2027,3163`.  The contingent pair is `4211,5393`.
All three arms are completed and evaluated for both first seeds before the
futility decision is read.

## Endpoint, stopping, and inference

The primary endpoint is exact top-1 under unchanged official five-view,
beam-10 reciprocal-rank aggregation.  Released-minus-Native and
Paper-minus-Native are the two prespecified treatment contrasts.  If both
treatments are nonpositive in both first seeds, stop for futility.  Otherwise
run both contingent paired seeds.  There is no early stop for success.

Report top-3/5/10, each view, validity, paired wins/losses, per-seed reaction
bootstrap intervals, exact McNemar tests, a crossed seed-by-reaction bootstrap,
and a seed-t interval explicitly labeled fragile at two or four seeds.  Report
raw inference and Holm adjustment across the two treatment contrasts.  Prior
512-reaction development results remain separate and are not pooled.

## Pre-endpoint evaluator gate

Only deterministic length-balanced assignment of whole reactions to the four
existing evaluator workers is considered.  Benchmark it against current
round-robin assignment on a locked 64-reaction, length-stratified subset of the
already-used development panel (SHA-256
`3a53520e3bef26e9e06f9ca869696a8693bfadd86595da3c8302a46caebafe17`),
using representative old Native, Released, and Paper checkpoints.  Retain it
only if median end-to-end wall speedup is at least 3% and every raw/canonical
ordered beam, aggregate rank/score, validity count, and exact flag is identical.
Otherwise use the established four-worker round-robin evaluator.  The
confirmation panel is forbidden for this benchmark.

The gate rejected length-balanced assignment: its three-checkpoint median
speedup was `1.0195x`, below the locked `1.03x` threshold, despite exact
ordered-output equality. A second outcome-blind worker-scaling check then
compared five and six processes with the four-worker reference. On the Native
screen, five workers gave `1.0128x` and six gave `.9833x`; both were exactly
equivalent, but neither cleared the `1.05x` continuation gate. Four workers
therefore remain locked. STP adds no inference-time module, so Released and
Paper use precisely this same generation computation; there is no separate
STP inference fast path to approximate.
