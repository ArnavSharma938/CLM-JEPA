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
