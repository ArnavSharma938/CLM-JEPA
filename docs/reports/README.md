# cLM-JEPA experiment results

## Current answer

MSE and MSE+SIGReg were tested. MSE alone reduced but did not eliminate ChemFM's JEPA variance contraction. MSE+exact-SIGReg-16 restored source/target variance to 90.3%/55.2% of native at epoch 4 and produced 85.9% raw four-way pair retrieval (25% chance), but it did not improve generation. The frozen mechanistic audit localizes the adverse update to mostly SIGReg-driven pressure in layers 17-21 rather than broad model-wide gradient opposition.

The final benchmark-faithful comparison used 1,280 unique USPTO-MIT reactions and all five official R-SMILES views:

| Primary endpoint | Native epoch 4 | MSE+SIGReg cLM-JEPA epoch 4 |
|---|---:|---:|
| Exact top-1 | 50/1,280 (3.906%) | 40/1,280 (3.125%) |

The paired difference was `-0.781` percentage points, 95% CI `[-1.719,+0.156]`, exact McNemar `p=0.1433`. A frozen futility rule excluded the prespecified `+1` pp benefit and stopped evaluation at 1,280. This result applies to the fixed seed-533 pilot endpoints; it is not a universal conclusion about JEPA.

## What was tested

| Experiment | Main result | Why it mattered |
|---|---|---|
| Native NTP | Final reference endpoint | Preserved ChemFM autoregressive fine-tuning |
| Symmetric cosine JEPA, k=1 | 17/512 vs native 24/512; target variance 495x below native | Identified extreme common-direction concentration with useful residual pair information |
| Target stop-gradient, k=1 | 26/512 vs native 24/512, `p=0.8318`; CE 7.69% worse | Partially relaxed geometry without an established behavioral gain |
| Cosine+SIGReg batch 2, k=0/k=1 | 2/256 and 3/256 vs native 7/256 | Neither k value restored geometry; batch-2 statistic was inadequate |
| Cosine+SIGReg batch 128, k=0 | Geometry restored; 0/256 top-1 | Cadence-confounded because updates fell from 320 to 20 |
| SIGReg batch-16 calibration | Exact streamed/direct gradients; SIGReg gradient did not vanish under contraction | Qualified a controlled batch-16 run |
| Cadence-matched cosine+SIGReg-16, k=0 | 3/256 vs native 6/256; variance remained 16.9x/14.8x lower | Showed the fixed cosine+SIGReg configuration did not stop contraction |
| Raw MSE, k=0 | Variance remained about 4x below native at epoch 2 | MSE alone was insufficient and stopped at the gate |
| **MSE+SIGReg-16, k=0** | Geometry restored; epoch-4 top-1 tied 6/256; CE 3.36% worse | Selected endpoint; separated geometry repair from generation benefit |
| Official five-view endpoint | Native 50/1,280 vs cLM-JEPA 40/1,280 | Final benchmark-faithful behavioral result |
| Reduced GSM8K LLM-JEPA reference | NTP 36/300 vs JEPA 28/300 | Not a successful control; its contraction was much less extreme than ChemFM's |

## Reports

Read in this order:

1. [Method and protocol fidelity](00_METHOD_AND_PROTOCOL_FIDELITY.md): what is faithful to LLM-JEPA and ChemFM, exact objectives, and evaluation semantics.
2. [Why the project moved from cosine to MSE+SIGReg](01_COSINE_TO_MSE_SIGREG_DIAGNOSIS.md): consolidated cosine failure, stop-gradient, SIGReg batch/cadence studies, gradient assay, and GSM8K comparison.
3. [MSE and MSE+SIGReg experiment](02_MSE_SIGREG_EXPERIMENT.md): the decisive objective ablation and representation result.
4. [Official five-view endpoint evaluation](03_OFFICIAL_ENDPOINT_EVALUATION.md): powered endpoint design, exact inference parity, statistics, and stopping decision.
5. [Mechanistic gradient and block-swap audit](04_MECHANISTIC_GRADIENT_AND_BLOCK_SWAP_AUDIT.md): auxiliary/NTP gradient decomposition, pair specificity, token-position compatibility, and causal depth localization.

## Research status

The current bottleneck is not unresolved global representation contraction. The strongest mechanistic evidence is localized: the trained auxiliary gradient is globally near-orthogonal to NTP, but SIGReg pressure in layers 17-21 conflicts most with early-token NTP, and bidirectional block swaps localize CE harm to the same region. Report 04 specifies the single controlled follow-up.
