# cLM-JEPA experiment results

## Current answer

MSE and MSE+SIGReg were tested. MSE alone reduced but did not eliminate ChemFM's JEPA variance contraction. MSE+exact-SIGReg-16 restored source/target variance to 90.3%/55.2% of native at epoch 4 and produced 85.9% raw four-way pair retrieval (25% chance), but it did not improve generation. The frozen mechanistic audit localizes the adverse update to mostly SIGReg-driven pressure in layers 17-21 rather than broad model-wide gradient opposition. A temporal fresh-slice audit then rejected direct SIGReg pair destruction: SIGReg improves pair discrimination, but becomes 4.06x larger than applied MSE by epoch 4 and remains mildly anti-NTP.

A subsequent Pair-Center Spread Floor (PCSF) test attempted the smallest anti-contraction constraint: a reference-relative one-sided standard-deviation floor on positive-pair centers. Its prespecified `rho=0.80`, `beta=4.2` configuration did not hold the floor (`0.535x` native-reference pair-center sigma at epoch 4), scored 4/256 top-1 versus native 6/256, and had 2.49% worse target CE. This tested calibration therefore failed both its mechanism and downstream criteria; it does not establish that a successfully enforced minimal floor would fail.

The final frozen directional audit separated contraction from task utility. MSE contracted pair-center spread in all 33 checkpoint-batch measurements, while the correct-pair-specific residual expanded it in all 33 and modestly improved held-out NTP at the important trained states. The contraction therefore comes from MSE's pair-blind alignment component. SIGReg genuinely reverses that direction but becomes adverse to held-out NTP; PCSF is directionally restorative and NTP-compatible where active, but too weak to overcome total MSE. Anti-contraction tuning alone is not the next bottleneck.

The requested projection-space test then placed both MSE and SIGReg behind one shared `2048->2048->2048->64` BN/ReLU head. It did not insulate ChemFM: projected `z` concentrated into about three effective dimensions, while raw `h` overshot native variance and lost rank. Target-token CE was `0.256497`, worse than direct MSE+SIGReg `0.248779`; on a matched budget-bounded 512-reaction five-view panel, native/direct/projected top-1 was `18/15/4`. Projected versus direct was `-2.148` pp, 95% CI `[-3.516,-0.781]`, McNemar `p=0.00342`.

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
| MSE+PCSF-16, k=0 | Floor not held; epoch-4 top-1 4/256 vs native 6/256; CE 2.49% worse | Falsified the tested PCSF calibration without resolving the broader minimal-floor hypothesis |
| Projection-space MSE+SIGReg, k=0 | z became centered but ~3-rank; h was not native-like; CE 6.57% worse than native; 4/512 top-1 | Rejected projector placement as sufficient insulation for this objective |
| Official five-view endpoint | Native 50/1,280 vs cLM-JEPA 40/1,280 | Final benchmark-faithful behavioral result |
| Reduced GSM8K LLM-JEPA reference | NTP 36/300 vs JEPA 28/300 | Not a successful control; its contraction was much less extreme than ChemFM's |

## Reports

Read in this order:

1. [Method and protocol fidelity](00_METHOD_AND_PROTOCOL_FIDELITY.md): what is faithful to LLM-JEPA and ChemFM, exact objectives, and evaluation semantics.
2. [Why the project moved from cosine to MSE+SIGReg](01_COSINE_TO_MSE_SIGREG_DIAGNOSIS.md): consolidated cosine failure, stop-gradient, SIGReg batch/cadence studies, gradient assay, and GSM8K comparison.
3. [MSE and MSE+SIGReg experiment](02_MSE_SIGREG_EXPERIMENT.md): the decisive objective ablation and representation result.
4. [Official five-view endpoint evaluation](03_OFFICIAL_ENDPOINT_EVALUATION.md): powered endpoint design, exact inference parity, statistics, and stopping decision.
5. [Mechanistic gradient and block-swap audit](04_MECHANISTIC_GRADIENT_AND_BLOCK_SWAP_AUDIT.md): auxiliary/NTP gradient decomposition, pair specificity, token-position compatibility, and causal depth localization.
6. [SIGReg pair-specificity audit](05_SIGREG_PAIR_SPECIFICITY_AUDIT.md): temporal true/shuffled and fresh-slice gradient responses at MSE+SIGReg epochs 1, 2, and 4.
7. [Pair-Center Spread Floor experiment](06_PCSF_EXPERIMENT.md): derivation, implementation, frozen calibration, A6000 optimization, four-epoch run, geometry, and downstream verdict.
8. [Contraction and held-out NTP directional audit](07_CONTRACTION_AND_NTP_DIRECTIONAL_AUDIT.md): time-matched spread trajectories, objective-induced spread velocity, and disjoint-batch NTP effects.
9. [Projection-space MSE+SIGReg experiment](08_PROJECTION_SPACE_MSE_SIGREG_EXPERIMENT.md): primary-source design, PCSF production cleanup, logical-batch projector training, two-space geometry, gradient alignment, CE, and generation verdict.
10. [JEPA–NTP gradient-interaction experiment](09_GRADIENT_INTERACTION_EXPERIMENT.md): clean direct MSE+SIGReg baseline, weight controls, PCGrad, CAGrad, published auxiliary-gradient similarity, optimized diagnostics, and autoregressive verdict.

## Research status

Global contraction remains a reproducible property of raw MSE, but fixing geometry alone has not produced decoder gains. Report 07 identifies the specific split: the reaction-pair-specific MSE residual is spread-preserving and modestly NTP-compatible, while the larger pair-blind alignment component contracts the representation and dilutes that benefit. Report 08 shows that a conventional train-only projector does not automatically isolate that pressure: gradients through the shared backbone can distort raw h even when both losses are computed only on z. Further work should target this objective/coupling decomposition rather than another anti-collapse increase or projector sweep.
