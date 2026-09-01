# STP capacity, formulation, and coefficient experiment preregistration

This document was frozen before any new 512-reaction endpoint was run. The
implementation starts from repository commit `29b64fc`; the final experiment
commit will be recorded in the completed report. All new training uses seeds
533 and 917, the existing 1,280-row training manifest, four epochs/320 optimizer
steps, effective batch size 16, the native ChemFM serialization and NTP loss,
and the official five-view/beam-10 endpoint. The 512 endpoint is the first 512
rows of the already frozen `prespecified_stage1_1280.jsonl`; its first 256 rows
are byte-for-byte the prior endpoint panel.

## Stage A

- Reuse, without retraining, native rank-8/alpha-8 and released-STP rank-8/
  alpha-8, lambda 0.02 checkpoints for seeds 533, 917, and 1301.
- Extend each endpoint from 256 to 512 by resuming the exact four-worker output
  partition. Previously generated rows are immutable; only rows 257--512 are
  generated.
- Primary comparison: paired released STP minus native exact top-1, per seed and
  pooled descriptively. Five-view results and paired reaction uncertainty are
  reported. Teacher-forced CE, correct-token rate/margin, fixed-span STP and
  gradient diagnostics are secondary.

## Stage B

- Train two paired rank-128/alpha-128 native and released-STP lambda-0.02 seeds.
- Primary comparison: released STP minus native within each rank, never the
  absolute rank-128 score.
- Stage-C rank rule, fixed before Stage A: choose rank 128 only if its mean
  treatment effect exceeds rank 8 by at least 0.005 absolute top-1 and both
  rank-128 seed effects are nonnegative. Otherwise choose rank 8, which is the
  practical/reuse default.

## Stage C

- On the selected rank, compare released and literal-paper objectives on frozen
  batches for raw loss, gradient norm, STP/NTP gradient cosine and norm ratio,
  and sampled-span-length dependence.
- Train paper-equation STP at lambda 0.02 for seeds 533 and 917. Compare native,
  released STP, and paper STP on the same endpoint.
- Stage-D formulation rule: choose paper STP only if its mean treatment effect
  exceeds released STP by at least 0.005 and both paper seed effects are
  nonnegative. Otherwise retain released STP.

## Stage D

- At the selected rank/formulation, evaluate lambda 0.005, 0.02, and 0.08,
  reusing the exact lambda-0.02 runs. Train two seeds for each newly required
  lambda.
- Select by mean paired official exact top-1 treatment effect, with all seed
  effects shown. No dense refinement is permitted. At most one outside-edge
  lambda may be run, and only if the edge advantage is at least 0.01 with both
  seed effects nonnegative; this screen does not otherwise launch it.

## Conditional final rank check

A final comparison at both ranks is run only if a non-0.02 lambda improves mean
treatment effect over lambda 0.02 by at least 0.01 and neither selected-lambda
seed is harmed. Exact-matching runs are reused; at most two seeds are newly
trained per missing rank. Each STP model is compared only with its same-rank
native control.

All negative runs are retained. These two-seed/512-reaction screens are
exploratory and are not upgraded to confirmatory significance claims.
