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
