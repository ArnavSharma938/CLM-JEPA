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
