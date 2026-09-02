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
