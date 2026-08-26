# Pair-Center Spread Floor experiment

## Measured summary

- Frozen pair-center standard deviation fell from `1.192x` the matched-native reference at epoch 1 to `0.688x`, `0.560x`, and `0.535x` at epochs 2–4; the fixed floor was `rho=0.80`.
- At epoch 4, source/target effective rank was `56.31 / 25.90`, versus native `49.50 / 26.27`.
- On the fixed 256-reaction panel, epoch-4 PCSF scored `4/256` exact top-1 versus native `6/256`; the paired difference was `-0.781 pp` (bootstrap 95% CI `[-2.344, +0.781] pp`, exact McNemar `p=0.625`).
- PCSF target-token CE was `0.246664`, `2.49%` above native `0.240683` and below MSE+SIGReg `0.248779`.
- Residual PC2-removed retrieval reached `84.0%`. Pair margin correlations with CE or beam-rank changes ranged from `-0.057` to `+0.095`, with all reported `p >= 0.128`.

The run records the trajectory produced by `rho=0.80` and `beta=4.2`; it is not a floor-held comparison because the measured spread fell below `rho`.

## Method

For source and target EOS states `s_i,t_i in R^D`, define

```text
d_i = s_i - t_i
m_i = (s_i + t_i) / 2
L_MSE = (1 / BD) sum_i ||d_i||^2
V_PC = (1 / ((B - 1)D)) sum_i ||m_i - mean(m)||^2
sigma_PC = sqrt(V_PC + epsilon)
L_PCSF = relu(rho * sigma_ref,B - sigma_PC)^2
L_total = L_NTP + active * 2 * (L_MSE + beta * L_PCSF)
```

The reference is the frozen matched-native epoch-4 representation for the exact same reaction identities in every logical batch. Reference states are detached; current source and target states both receive PCSF gradients. PCSF is computed exactly over the established logical batch of 16 using the same physical-4/accumulation-4 cadence as the matched controls. The auxiliary branch is active on one logical batch with probability `0.5`, exactly as in the batch-16 MSE+SIGReg protocol.

No SIGReg, projector, stop-gradient, covariance/rank penalty, whitening, alternative readout, data change, or hyperparameter sweep was added.

### Variance identity

Since `s_i = m_i + d_i/2` and `t_i = m_i - d_i/2`, and the joint mean is `mean(m)`, the cross terms cancel:

```text
(1 / (2BD)) sum_i [||s_i - mean(m)||^2 + ||t_i - mean(m)||^2]
  = (1 / BD) sum_i ||m_i - mean(m)||^2 + (1/4) L_MSE.
```

Thus the exact population identity is `V_joint = V_center + L_MSE/4`. With the implemented unbiased center denominator, the corresponding paired-unbiased identity is `V_joint,pair = V_center,unbiased + B L_MSE / [4(B-1)]`. Both forms are tested numerically.

## Literature grounding and scope

- [VICReg](https://arxiv.org/abs/2105.04906) uses a one-sided **per-dimension** standard-deviation hinge, plus covariance regularization. Its paper and official implementation justify standard deviation rather than raw variance because the raw-variance gradient vanishes near contraction. PCSF retains the standard-deviation hinge but applies one scalar floor to positive-pair centers.
- [LeJEPA](https://arxiv.org/abs/2511.08544) uses SIGReg to constrain the marginal toward an isotropic Gaussian. PCSF deliberately makes no Gaussian, whitening, or decorrelation demand.
- [Temporally Centered SIGReg](https://arxiv.org/abs/2607.26924) reports that marginal Gaussianization can compress task-cluster separation and instead regularizes temporally centered residuals. [Sub-JEPA](https://arxiv.org/abs/2605.09241) weakens full ambient Gaussianization through random subspaces. Neither uses the positive-pair midpoint and a reference-relative one-sided spread floor.
- [Barlow Twins](https://arxiv.org/abs/2103.03230) is pair-coupled but drives a normalized cross-correlation matrix toward identity. [Whitening-MSE](https://arxiv.org/abs/2007.06346) explicitly whitens before positive-pair MSE. Both impose stronger spectral structure than PCSF.

A targeted search of arXiv and the cited official repositories for objectives on positive-pair centers, midpoints, paired barycenters, and reference-relative spread floors found no directly equivalent objective. This is a bounded search result, not a novelty claim.

## Implementation and validation

Implementation paths at the time of the experiment:

- `src/jepa.py`: pair centers, unbiased spread, one-sided PCSF, exact streamed statistic, and endpoint VJP.
- `src/train.py`: logical-batch PCSF training, exact reference lookup, activity cadence, metrics, and checkpoint serialization.
- `src/chemfm.py`: reference-index collation and attention-backend loading.
- `scripts/pcsf_experiment.py`: frozen reference extraction, trajectory measurement, and gradient calibration.
- `scripts/benchmark_pcsf_training.py`: A6000 parity and throughput benchmarks.

PCSF was subsequently removed from the maintained training path. These scripts, artifacts, and report remain historical records.

Tests cover center/spread arithmetic, both variance identities, zero/positive hinge behavior, restorative gradients, translation invariance, uniform scaling, exact identity lookup, streamed-versus-materialized loss/representation/parameter gradients, and endpoint-only forward parity. The complete local suite passed: `80 passed, 1 skipped`.

The compact LM-head candidate produced BF16 logit differences up to `0.125`, an NTP-loss difference of `4.30e-4`, and LoRA-gradient relative L2 difference of `0.805%`; it was not used. The endpoint-only no-grad statistics pass produced identical endpoint states and was used for the experiment.

## Calibration

### Reference geometry and `rho`

All values below use the same 1,280 training reactions and k=0 EOS readout.

| Frozen state | Pair-center sigma | Ratio to native e4 | Source / target variance | Source / target rank |
|---|---:|---:|---:|---:|
| Base ChemFM | 0.114865 | 3.304 | `2.444e-2 / 2.270e-2` | `28.33 / 25.04` |
| Native e4 reference | 0.034769 | 1.000 | `1.378e-3 / 2.359e-3` | `49.50 / 26.27` |
| MSE e1 | 0.041024 | 1.180 | `1.548e-3 / 4.022e-3` | `47.12 / 6.96` |
| Failed MSE e2 | 0.020744 | 0.597 | `5.252e-4 / 7.918e-4` | `51.74 / 27.23` |
| MSE+SIGReg e2 | 0.035121 | 1.010 | `1.672e-3 / 1.583e-3` | `41.70 / 35.38` |
| MSE+SIGReg e4 | 0.032526 | 0.935 | `1.373e-3 / 1.361e-3` | `42.73 / 36.29` |

For the actual shuffled logical batches, every failed-MSE-e2 ratio was at most `0.695`; every MSE+SIGReg-e4 ratio was at least `0.808`; matched native is exactly `1.0` by construction. `rho=0.80` was fixed from that separation. It permits 20% standard-deviation contraction relative to normal NTP while activating before the known failed MSE state.

### Gradient calibration and `beta`

At failed MSE epoch 2, four fixed real batches gave a median equal-norm coefficient of `4.205` over all trainable parameters and `7.309` over LoRA A/B alone. The single prespecified coefficient was rounded to `beta=4.2`, matching the requested all-trainable calibration. At that state, the median applied PCSF/MSE norm ratio was approximately `1.05`; the raw PCSF/NTP cosine averaged `-0.092` globally and `-0.030` on LoRA A/B.

No outcome-based coefficient tuning was performed.

## A6000 execution optimization

Hardware was one RTX A6000 (48 GB), 6 vCPUs, and 200 GB storage. Short candidates used identical weights, examples, seed, objective, and logical batch.

| Candidate | Examples/s | Step s | Peak allocated | Decision |
|---|---:|---:|---:|---|
| Eager, checkpointing, physical 4 | 3.55 | 4.510 | 12.43 GiB | reference |
| SDPA, checkpointing, physical 4 | 4.08 | 3.922 | 10.54 GiB | SDPA retained |
| SDPA, no checkpointing, physical 4 | 4.01 | 3.991 | 10.54 GiB | matched prior setting |
| SDPA, no checkpointing, fused, workers 2, physical 4 | 3.95 | 4.047 | 10.54 GiB | workers did not help |
| SDPA, physical 8 | 6.88 | 2.325 | 19.14 GiB | not used: changed control computation |
| SDPA, physical 16 | 10.11 | 1.582 | 36.31 GiB | not used: changed control computation |

Physical 8/16 changed the established microbatch token-loss weighting, padding shapes, and dropout RNG grouping, so their speed was not purchased at the cost of a mismatched native control. The final production configuration preserved physical 4, accumulation 4, fused AdamW, SDPA, BF16, no checkpointing, and the original data order. A 20-update unsynchronized benchmark measured `4.33 examples/s`, `3.691 s/update`, and `13.85 GiB` peak allocation. Training plus the two prespecified two-reaction validation passes took `1,746.1 s` (`29.1 min`). Four exact batch-1 model replicas completed the 256-reaction beam evaluation in `1,103.3 s` (`18.4 min`).

## Training trajectory

| Epoch | Active steps | PCSF above floor on active steps | Mean active spread ratio | NTP | MSE | PCSF |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 47/80 | 0/47 (0.0%) | 2.590 | 1.22342 | 0.033673 | 0 |
| 2 | 43/80 | 11/43 (25.6%) | 0.980 | 0.32981 | 0.004480 | `3.40e-6` |
| 3 | 42/80 | 36/42 (85.7%) | 0.696 | 0.21091 | 0.001967 | `2.11e-5` |
| 4 | 40/80 | 40/40 (100%) | 0.617 | 0.17388 | 0.001481 | `4.48e-5` |

The frozen trajectory was `1.192 -> 0.688 -> 0.560 -> 0.535` times reference sigma. At frozen PCSF epoch 2, the applied PCSF/MSE gradient norm ratio was `0.300` globally (`0.231` LoRA A/B); at epoch 4 it was `0.717` (`0.770` LoRA A/B). Applied PCSF was about `1.25%` of the NTP gradient norm. PCSF was active on all 40 active auxiliary steps at epoch 4, while the frozen spread ratio was `0.535`.

## Frozen representation results

### Global 1,280-reaction endpoint

| State | Pair-center sigma ratio | Source / target variance | Mean energy source / target | Rank source / target |
|---|---:|---:|---:|---:|
| Native e4 | 1.000 | `1.378e-3 / 2.359e-3` | `0.9825 / 0.9691` | `49.50 / 26.27` |
| MSE e2 | 0.597 | `5.252e-4 / 7.918e-4` | `0.9934 / 0.9901` | `51.74 / 27.23` |
| MSE+SIGReg e4 | 0.935 | `1.373e-3 / 1.361e-3` | `0.9692 / 0.9681` | `42.73 / 36.29` |
| PCSF e2 | 0.688 | `6.654e-4 / 1.100e-3` | `0.9916 / 0.9861` | `50.62 / 15.74` |
| PCSF e4 | 0.535 | `4.215e-4 / 6.143e-4` | `0.9946 / 0.9922` | `56.31 / 25.90` |

PCSF epoch 4 source/target variance was `30.6% / 26.0%` of native. Its source/target effective rank was `56.31 / 25.90`, versus `51.74 / 27.23` for MSE epoch 2.

### Fixed 256-reaction panel

| State | Raw margin | Raw retrieval | Residual PC2 margin | Residual PC2 retrieval |
|---|---:|---:|---:|---:|
| Native e4 | 0.005479 | 43.4% | 0.258816 | 76.6% |
| MSE e2 | 0.001650 | 64.5% | 0.288993 | 83.6% |
| MSE+SIGReg e4 | 0.011634 | 85.9% | 0.435986 | 93.8% |
| PCSF e2 | 0.002213 | 67.6% | 0.301297 | 80.1% |
| PCSF e4 | 0.001550 | 69.5% | 0.309533 | 84.0% |

PCSF epoch-4 raw/residual retrieval was `69.5% / 84.0%`, versus `85.9% / 93.8%` for MSE+SIGReg epoch 4.

## Autoregressive evaluation

The endpoint comparison used the same frozen 256 unique validation reactions, one canonical view per identity, exact beam width 10, and the maintained ranking/canonicalization path.

| Epoch-4 endpoint | Top-1 | Top-3 | Top-5 | Top-10 | Valid-candidate rate | Target CE |
|---|---:|---:|---:|---:|---:|---:|
| Native | 2.34% (6/256) | 10.16% | 15.63% | 20.31% | 78.20% | 0.240683 |
| MSE+SIGReg | 2.34% (6/256) | 9.38% | 13.28% | 19.14% | 86.60% | 0.248779 |
| PCSF | 1.56% (4/256) | 8.59% | 13.28% | 17.58% | 85.00% | 0.246664 |

For native versus PCSF top-1: both correct `3`, native-only `3`, PCSF-only `1`, neither `249`. Rank improved on `14`, worsened on `31`, and tied on `211` reactions. Mean rank improvement was `-0.211` (95% CI `[-0.457, 0.027]`).

PCSF CE worsened for `53.9%` of reactions. The mean per-reaction native-minus-PCSF CE difference was `-0.00616` with bootstrap 95% CI `[-0.01241, -0.00011]`; Wilcoxon `p=0.245`. At epoch 2, CE was native `0.251414`, MSE `0.257006`, MSE+SIGReg `0.261471`, and PCSF `0.260526`.

## Decoder coupling and source sensitivity

At PCSF epoch 4:

- raw pair margin versus native-to-PCSF CE improvement: Spearman `rho=0.054`, `p=0.386`;
- residual PC2 pair margin versus CE improvement: `rho=0.095`, `p=0.128`;
- raw pair margin versus beam-rank improvement: `rho=0.004`, `p=0.952`;
- residual PC2 pair margin versus beam-rank improvement: `rho=-0.057`, `p=0.365`.

All four reported correlations were between `-0.057` and `+0.095`, with `p >= 0.128`.

| Intervention | Raw endpoint sensitivity native / PCSF | Residual PC2 sensitivity native / PCSF | Decoder CE change native / PCSF |
|---|---:|---:|---:|
| Contributor removed | `0.0255 / 0.0175` | `0.4437 / 0.4097` | `0.5633 / 0.5706` |
| Contributor replaced | `0.00560 / 0.00172` | `0.3390 / 0.3568` | `0.7431 / 0.7454` |
| Unrelated source | `0.0159 / 0.00461` | `0.8864 / 0.8882` | `0.9071 / 0.8982` |

Raw endpoint sensitivity was lower for PCSF than native under all three interventions. Residual PC2 sensitivity differed by `-0.0340`, `+0.0178`, and `+0.0018`; decoder CE changes differed by `+0.0073`, `+0.0023`, and `-0.0089`.

## Recorded comparisons

- Statistic, reference lookup, hinge, restorative-gradient, and direct/streamed-equivalence tests passed.
- The calibration state had an applied PCSF/MSE norm ratio near `1.05`. The trained checkpoint ratios were `0.30x` at epoch 2 and `0.72x` at epoch 4; the PCSF/NTP ratio was about `0.0125x`.
- The measured spread was below the fixed floor from epoch 2 onward.
- Epoch-4 source/target rank was `56.31 / 25.90`, residual retrieval was `84.0%`, top-1 was `4/256`, and target-token CE was `0.246664`.

## Evidence paths

- Training/config/curves: `runs/pcsf/training/result.json`
- Epoch checkpoints: `runs/pcsf/training/checkpoints/epoch_{1,2,3,4}`
- Reference cache: `runs/pcsf/reference/native_epoch4_pair_centers.pt`
- Reference and PCSF trajectories: `runs/pcsf/calibration/spread_trajectory.json`, `runs/pcsf/analysis/spread_trajectory.json`
- Beta and post-training gradients: `runs/pcsf/calibration/beta_at_mse_e2.json`, `runs/pcsf/analysis/gradients/epoch_{1,2,4}.json`
- A6000 benchmarks/utilization: `runs/pcsf/benchmark/`
- Fixed-panel geometry: `runs/pcsf/analysis/geometry_256.json`
- Per-reaction CE/interventions: `runs/pcsf/evaluation/pcsf_epoch{2,4}_diagnostics.json`
- Complete ordered beam candidates: `runs/pcsf/evaluation/generation_epoch4/pcsf_epoch4_generation.jsonl`
- Paired statistics/coupling: `runs/pcsf/evaluation/summary_epoch4.json`
- Verified transfer archive: `runs/thunder_pcsf_transfer/pcsf_artifacts.tar.gz`, SHA-256 `f14134bcc0d072201525d25f8098feaaa3598605c7555469e08ff71ad87b4fec`
