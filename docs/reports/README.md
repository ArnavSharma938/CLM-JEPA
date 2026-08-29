# cLM-JEPA experiment reports

This directory contains objective records of the completed experiments and
frozen diagnostic audits. Each report records its protocol, measurements,
limitations, and artifact paths. Interpretive recommendations are intentionally
excluded.

## Current measurements

| Condition or audit | Recorded endpoint |
|---|---|
| Native NTP | Official five-view top-1 `50/1,280`; fixed-256 one-view top-1 `6/256` |
| Symmetric cosine endpoint JEPA | Top-1 `17/512` versus native `24/512`; target variance `495x` below native |
| Endpoint MSE+SIGReg | Official five-view top-1 `40/1,280`; fixed-256 one-view top-1 `6/256`; target-token CE `0.248779`; raw pair retrieval `85.9%` |
| PCSF | Floor ratio `0.535` at epoch 4 for `rho=0.80`; top-1 `4/256`; CE `0.246664` |
| Projection-space MSE+SIGReg | Projected rank `3.22/3.29`; raw rank `12.58/5.22`; CE `0.256497`; top-1 `4/512` |
| Gradient-interaction methods | PCGrad/CAGrad/Du top-1 `3.91%/2.34%/3.91%` on 256; CE deltas `+6.82%/+22.67%/+1.33%` versus native |
| Dense causal V-JEPA 2.1-style | Target-token CE `0.253547`; top-1 `6/256`; top-10 `39/256`; global raw retrieval `33.59%` |
| Frozen ChemFM event geometry | Final-layer ring/motif local deltas `+0.051/+0.102`; semi-global deltas `+0.075/+0.036`; branch and inferred-center effects do not persist at long spans |

The official native-versus-endpoint-MSE+SIGReg top-1 difference was `-0.781`
percentage points, bootstrap 95% CI `[-1.719,+0.156]`, exact McNemar
`p=0.1433`. The frozen futility rule stopped evaluation at 1,280 reactions.

The generation-pathway audit additionally recorded:

- final-layer cLM source-to-target CKA `0.547` and source-to-AR-product CKA
  `0.278`;
- native-recipient activation-patch CE changes of `+0.00834`, `+0.00690`, and
  `+0.03069` after layers 11, 16, and 21;
- raw NTP/JEPA gradient cosine `-0.0223` and saved-state AdamW update cosine
  `+0.8143`;
- cLM hard-four-way retrieval `76.95%`, and `72.27%` after independent
  canonicalization;
- cLM-minus-native best-top-3/5/10 Morgan Tanimoto changes of `-0.00911`,
  `-0.01823`, and `-0.01285` across 1,280 existing predictions.

## Experiment inventory

| Experiment | Main recorded comparison |
|---|---|
| Native NTP | ChemFM autoregressive fine-tuning reference |
| Symmetric cosine JEPA, k=1 | `17/512` top-1 versus native `24/512` |
| Target stop-gradient, k=1 | `26/512` versus native `24/512`, `p=0.8318`; CE `7.69%` above native |
| Cosine+SIGReg batch 2, k=0/k=1 | `2/256` and `3/256` versus native `7/256` |
| Cosine+SIGReg batch 128, k=0 | `0/256`; 20 rather than 320 updates |
| SIGReg batch-16 calibration | Direct/streamed objective and gradient parity measured |
| Cadence-matched cosine+SIGReg-16, k=0 | `3/256` versus native `6/256`; source/target variance `16.9x/14.8x` below native |
| Raw MSE, k=0 | Epoch-2 variance about `4x` below native; stopped at the frozen gate |
| Endpoint MSE+SIGReg-16, k=0 | Epoch-4 top-1 `6/256`; CE `3.36%` above native; raw retrieval `85.9%` |
| MSE+PCSF-16, k=0 | Floor not held; top-1 `4/256`; CE `2.49%` above native |
| Projection-space MSE+SIGReg, k=0 | CE `6.57%` above native; top-1 `4/512` |
| Gradient-interaction methods | Weight controls, PCGrad, CAGrad, and Du auxiliary similarity completed |
| Dense causal V-JEPA 2.1-style | CE `3.98%` above native; top-1 `6/256`; top-10 `39/256` |
| Reduced GSM8K reference | NTP `36/300`; JEPA `28/300` |

## Reports

Read in this order:

1. [00 — Method and protocol fidelity](00_METHOD_AND_PROTOCOL_FIDELITY.md): exact objectives and evaluation semantics.
2. [01 — Endpoint objective development and official evaluation](01_ENDPOINT_OBJECTIVES_AND_OFFICIAL_EVALUATION.md): early cosine controls, MSE/MSE+SIGReg training, representation measurements, and the later 1,280-reaction official five-view endpoint. Its context table distinguishes the ChemFM experiments from the reduced DeepSeek/GSM8K reference.
3. [02 — Endpoint frozen mechanism audits](02_ENDPOINT_MECHANISM_AUDITS.md): four chronologically separated frozen phases—gradient/block swaps, SIGReg pair specificity, contraction/NTP directions, and the final generation-pathway/AdamW/chemistry audit. Each phase identifies its own checkpoints, parameter scope, panel size, and execution date relative to the trained interventions.
4. [03 — Endpoint trained intervention experiments](03_ENDPOINT_INTERVENTION_EXPERIMENTS.md): three separate later training programs—PCSF, projection-space MSE+SIGReg, and the gradient-interaction matrix. Its context table records that these are independent ChemFM-1B trajectories rather than sequential fine-tunes and identifies their different 256/512-reaction endpoints.
5. [04 — Dense causal V-JEPA 2.1-style experiment](04_DENSE_CAUSAL_VJEPA2_1_EXPERIMENT.md): separate architecture/state family with causal suffix/context prediction, deep supervision, EMA target, super-mini, pilot, and frozen evaluation.
6. [05 — Persistent pair-specific residual trajectory](05_PAIR_RESIDUAL_TRAJECTORY_EXPERIMENT.md): matched native versus persistent residual-JEPA training and fixed five-view generation endpoint.
7. [06 — Frozen ChemFM chemical-event trajectory geometry](06_FROZEN_CHEMFM_TRAJECTORY_GEOMETRY.md): no-training all-layer local-curvature and ordered-span STP-alignment assay with matched chemical-event controls.

Consolidation history:

- former reports 01 and 03 are contained in current report 01;
- former reports 05, 07, and 11 are contained in current report 02;
- former reports 08, 09, and 10 are contained in current report 03.

No recorded protocol, measurement, limitation, or artifact provenance was
deleted. Bulky reproducible intermediates were removed after consolidation as
documented in reports 01–03; compact summaries and selected comparison
checkpoints remain local, and the removed tracked files are recoverable from
commit `61fbc74`.
