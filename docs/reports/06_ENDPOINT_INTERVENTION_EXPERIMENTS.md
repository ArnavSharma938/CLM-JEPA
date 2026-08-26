# Endpoint cLM-JEPA trained intervention experiments

This record consolidates three later trained intervention programs around the direct endpoint MSE+SIGReg result. Each part is a separate training trajectory. Metrics are not pooled across parts, and each section preserves its original evaluation panel and artifact paths.

## Chronology and experimental context

| Part | Repository record date | Relative time | Model and initialization context | Objective change | Main evaluation scope |
|---|---|---|---|---|---|
| I. Pair-Center Spread Floor | 2026-08-16, commit `fd04c0d` | First of these interventions, after the direct endpoint MSE+SIGReg and SIGReg-specificity work | ChemFM-1B, seed 533, separately trained LoRA trajectory; native and earlier MSE/MSE+SIGReg checkpoints used as frozen references | Replaced SIGReg with historical PCSF while retaining endpoint MSE | Four epochs on the fixed 1,280-row training manifest; frozen 256-reaction representation and generation panels |
| II. Projection-space MSE+SIGReg | 2026-08-22, commit `3db02fa` | Recorded six days after Part I; the PCSF branch was removed before this run | ChemFM-1B, seed 533, new training trajectory from the controlled starting model; shared train-only 2048→2048→2048→64 projector | Moved both MSE and SIGReg from raw endpoint states to projected states | Four epochs on the same 1,280 rows; 256-reaction CE/geometry and a budget-bounded matched 512-reaction five-view prefix |
| III. Gradient-interaction matrix | 2026-08-23, commit `07fac75` | Recorded the day after Part II; the projector branch was removed and direct endpoint MSE+SIGReg restored | ChemFM-1B, seed 533, seven separately trained four-epoch trajectories with identical manifests and optimizer controls | Kept direct raw-endpoint MSE+SIGReg and changed only auxiliary weighting or LoRA gradient combination | All seven representation/gradient audits; user-revised 256-reaction behavioral endpoint for the retained conditions |

The dates above are the first repository commits containing the original reports. These are not sequential fine-tunes of one checkpoint into the next. Each intervention has its own optimizer state and checkpoint trajectory. They share the ChemFM-1B base and controlled USPTO-MIT training manifest, but differ in auxiliary parameterization, active production code at the time, and behavioral-panel size.

## Part I — Pair-Center Spread Floor

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

## Part II — Projection-space MSE+SIGReg

## Measured summary

The projected space had source/target variance `0.243582 / 0.263016`, effective
rank `3.22 / 3.29`, and pair retrieval `76.56%`. Raw LM space had variance
`0.004884 / 0.012639`, effective rank `12.58 / 5.22`, and pair retrieval
`85.55%`. On the matched
256-reaction teacher-forced panel, projected cLM-JEPA CE was `0.256497`, worse
than direct MSE+SIGReg (`0.248779`) and native (`0.240683`). On the first 512
reactions of the frozen official five-view manifest, exact top-1 was 4/512,
versus 15/512 for direct MSE+SIGReg and 18/512 for native.

## Primary-source basis

The architecture was fixed before training from the requested primary sources.

- [LeJEPA](https://arxiv.org/pdf/2511.08544) reports its best tested projector
  dimension at 64 in Table 1d and generally favors a three-layer projector in
  its depth ablation.
- The pinned official LeJEPA
  [`MINIMAL.md`](https://raw.githubusercontent.com/galilai-group/lejepa/c293d291ca87cd4fddee9d3fffe4e914c7272052/MINIMAL.md)
  separates backbone `emb` from shared-MLP `proj`, applies invariance and
  SIGReg to `proj`, and evaluates `emb`. Its MLP has two wide hidden layers with
  BatchNorm.
- [SimCLR Section 4.2](https://arxiv.org/pdf/2002.05709) reports its nonlinear,
  linear, and no-projection comparisons and evaluates the representation before
  the head for downstream use.
- The official SimCLR
  [`model.py`](https://github.com/google-research/simclr/blob/master/model.py)
  and
  [`model_util.py`](https://github.com/google-research/simclr/blob/master/model_util.py)
  pass projected hidden states to the SSL loss, use hidden BN/nonlinear layers,
  and omit a final ReLU.

The frozen shared head was therefore
`2048 -> 2048 -> 2048 -> 64`: hidden Linear-BatchNorm-ReLU blocks and a final
Linear layer, with no final nonlinearity and no L2 normalization. It has
8,532,032 trainable parameters.

## Objective and implementation

For the unchanged k=0 source and target EOS states `h_s,h_t`, training used one
shared projector `P` and only the following auxiliary:

`z_s=P(h_s), z_t=P(h_t)`

`L_aux = MSE(z_s,z_t) + [4 * 0.01 / 0.99] SIGReg({z_s,z_t})`

`L_total = L_NTP + active * 2 * L_aux`

`active` retained the established 50% logical-batch cadence. The physical batch
was 4, accumulation/logical JEPA batch was 16, and the projector received the
concatenated 32 source-plus-target rows in one call. Both BatchNorm layers thus
normalized the logical JEPA batch rather than individual microbatches.

The active update uses a no-grad endpoint pass over the four physical chunks,
one exact logical-batch projector/SIGReg backward pass, and RNG-replayed ChemFM
passes that inject the endpoint VJP alongside unchanged NTP. Gradients flow
`z -> P -> h -> ChemFM`; no endpoint is detached. A compact native-logit path
was enabled after executable state/loss/gradient parity tests to avoid
projecting unused token logits during active replay.

There was no raw-endpoint MSE or SIGReg in this condition. The projection
condition was subsequently removed from the maintained trainer. Historical
scripts, checkpoints, and evidence remain under `scripts/` and `runs/`.

## Frozen training protocol and execution

The run matched the prior direct MSE+SIGReg condition: ChemFM-1B, USPTO-MIT
forward prediction, seed 533, the identical 1,280-row training manifest, LoRA,
LR `1e-4`, four epochs, 80 optimizer updates per epoch, BF16, SDPA, fused AdamW,
no gradient checkpointing, physical batch 4, accumulation 4, k=0, outer
coefficient 2, SIGReg tradeoff `0.01`, and the same auxiliary activity RNG.
Manifest hashes matched the control exactly:

- train: `b5900bc7e4f1a858ecf3fdf3732da63e08fc0f955f1cb9ccf90534e2273c8dba`
- validation: `41db8068aa3c4b6faf7149ce8ee5645e5ce3794f64d52c0e56552456721e5013`

The A6000 preflight exercised real optimizer state and exact logical BatchNorm.
The full run completed 320 updates, 172 active logical updates, in 32.33 minutes
with 14.74 GiB peak tensor allocation.

| Epoch | Mean NTP | Projected MSE | Projected SIGReg | Active updates |
|---:|---:|---:|---:|---:|
| 1 | 1.302790 | 0.151014 | 5.078280 | 47/80 |
| 2 | 0.422565 | 0.091792 | 4.835297 | 43/80 |
| 3 | 0.307564 | 0.082210 | 3.753002 | 42/80 |
| 4 | 0.253846 | 0.085376 | 3.128446 | 40/80 |

NTP and both projected losses remained finite. The fixed two-row selector CE at
epoch 4 was `0.172400`; the larger matched CE panel below is the meaningful
autoregressive diagnostic.

## Two-space geometry

The epoch-4 diagnostic used 256 unique validation identities. Projector
BatchNorm used saved running statistics at evaluation. SIGReg's raw statistic
scales with evaluation sample count, so the 256-row value is reported as a
geometry diagnostic and is not compared numerically with the logical-N=16
training curve.

| Metric | Raw LM space h | Projected JEPA space z |
|---|---:|---:|
| Dimensions | 2048 | 64 |
| Source / target variance | `0.004884 / 0.012639` | `0.243582 / 0.263016` |
| Pair-center spread | `0.080046` | `0.491247` |
| Source / target effective rank | `12.58 / 5.22` | `3.22 / 3.29` |
| Source / target mean-direction energy | `0.9391 / 0.8389` | `0.00147 / 0.00392` |
| Correct-minus-random cosine margin | `0.04971` | `0.81588` |
| Pair retrieval top-1 / MRR | `85.55% / 90.53%` | `76.56% / 86.52%` |
| MSE alignment | n/a | `0.048751` |
| SIGReg statistic, N=256 | n/a | `49.7792` |

Across the four logical audit batches, projected source effective rank fell from
`9.59` at epoch 1 to `3.23` at epoch 2 and `2.82` at epoch 4 while variance
rose from `0.0178` to `0.3096`.

The matched epoch-4 native source/target variances were `0.001431/0.002320`, effective ranks
`41.00/22.61`, and mean-direction energies `0.9818/0.9695`. Direct MSE+SIGReg
was `0.001293/0.001281`, ranks `38.36/34.06`, and energies `0.9709/0.9698`.
The projected condition's raw values were `0.004884/0.012639`, ranks
`12.58/5.22`, energies `0.9391/0.8389`, and retrieval `85.55%`; direct
MSE+SIGReg retrieval was `85.9%`.

## Held-out NTP gradient alignment

Four disjoint logical batches were audited at epochs 1, 2, and 4. Projected
MSE, SIGReg, and their full applied combination were mapped through the
projector into the 308 LoRA gradient tensors and compared with held-out NTP.

| Epoch | Held-out NTP | z MSE / SIGReg | MSE cosine / norm ratio | SIGReg cosine / norm ratio | Full cosine / norm ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.48309 | `0.10642 / 5.21062` | `-0.0057 / 2.18x` | `+0.0087 / 0.55x` | `-0.0025 / 2.02x` |
| 2 | 0.34753 | `0.08278 / 3.76448` | `+0.0447 / 2.89x` | `-0.0385 / 1.14x` | `+0.0330 / 2.88x` |
| 4 | 0.28381 | `0.06437 / 2.78183` | `-0.0243 / 2.45x` | `-0.0079 / 0.81x` | `-0.0300 / 2.36x` |

At epoch 4, the full projected auxiliary/NTP cosine was `-0.0300` and its norm
ratio was `2.36x`. The training implementation propagated its VJP through `h`
into ChemFM.

## NTP and generation

On the same frozen 256-reaction decoder-coupling panel used by prior reports:

| Condition | Aggregate target-token CE | Relative to native |
|---|---:|---:|
| Native | 0.240683 | reference |
| Direct MSE+SIGReg | 0.248779 | +3.36% |
| **Projected MSE+SIGReg** | **0.256497** | **+6.57%** |

Projected CE was 3.10% above direct MSE+SIGReg.

Generation used exact ChemFM five-view, beam-10 scoring. The projected model's
longer decodes made all 1,280 reactions incompatible with the authorized cloud
budget, so evaluation was fixed during execution—before inspecting outcomes—to
the first 512 reactions of the already frozen sequential manifest. The same
prefix was extracted from existing native and direct prediction artifacts. It
is a matched, budget-bounded descriptive panel, not the original 1,280-reaction
confirmatory endpoint.

| Endpoint on matched 512 | Native | Direct MSE+SIGReg | Projected MSE+SIGReg |
|---|---:|---:|---:|
| Exact top-1 | 18 (3.52%) | 15 (2.93%) | **4 (0.78%)** |
| Exact top-3 | 88 (17.19%) | 83 (16.21%) | **33 (6.45%)** |
| Exact top-5 | 130 (25.39%) | 119 (23.24%) | **70 (13.67%)** |
| Exact top-10 | 187 (36.52%) | 176 (34.38%) | **135 (26.37%)** |
| Official view-candidate validity | 99.09% | 97.45% | **96.68%** |
| Mean unique valid per view | 7.24 | 8.59 | **8.31** |

Against direct MSE+SIGReg, projected top-1 changed by `-2.148` percentage
points, paired bootstrap 95% CI `[-3.516,-0.781]`, exact McNemar `p=0.00342`
(3 both correct, 12 direct-only, 1 projected-only). Against native, the change
was `-2.734` points, CI `[-4.297,-1.367]`, `p=0.000122`. Direct versus native
on this prefix was not significant (`p=0.629`).

## Recorded comparisons

- Projected `z` effective rank was `3.22 / 3.29`; raw `h` effective rank was
  `12.58 / 5.22`; native raw rank was `41.00 / 22.61`.
- Raw `h` variance was `3.41x / 5.45x` native source/target variance.
- Epoch-4 projected full-auxiliary/NTP cosine and norm ratio were `-0.0300` and
  `2.36x`.
- Target-token CE was `0.256497`, versus `0.248779` direct and `0.240683`
  native. Exact top-1 on the matched 512 was `4`, versus `15` direct and `18`
  native.

## Evidence

- Training result and curves: `runs/projected_mse_sigreg/training/result.json`
- Epoch 1-4 adapters, projectors, and optimizer states:
  `runs/projected_mse_sigreg/training/checkpoints/`
- Epoch-4 projector SHA-256:
  `6c96e4660ee6225f15119d7fdcebcfe9bfe15e33ba0b35c6e6dd662d8762aedd`
- Two-space 256-row geometry:
  `runs/projected_mse_sigreg/evaluation/two_space_epoch4.json`
- Epoch 1/2/4 gradient audit:
  `runs/projected_mse_sigreg/evaluation/gradient_audit.json`
- Matched 256-row CE/interventions:
  `runs/projected_mse_sigreg/evaluation/projected_decoder_coupling_256.json`
- Ordered 512-row manifest, predictions, and paired summaries:
  `runs/projected_mse_sigreg/evaluation/panel_512/`
- Projected 512 prediction SHA-256:
  `2cd5dcb5b6a7cb3eb144e656ad293abd9c31f1489337317437927961034629fc`

The real ChemFM smoke test reached all projector parameters and 154/308 LoRA
gradient tensors on the local RTX 4050. The full local suite and the controlled
A6000 environment each passed 77 tests with one skip before training; final
post-report verification is recorded in the repository handoff.

## Part III — JEPA–NTP gradient interaction

## Scope

This experiment measures interaction between direct raw-endpoint MSE+SIGReg and
ChemFM's next-token-prediction (NTP) gradient. The controlled test keeps the data, ChemFM checkpoint, LoRA
configuration, optimizer, schedule, logical batch, JEPA statistic, cadence,
and evaluation pipeline fixed.  It changes only the way the NTP and JEPA
gradients are combined on trainable LoRA parameters.

## Primary-source implementation decisions

The implementations were checked against the primary papers and official
code before integration:

- [Yu et al., *Gradient Surgery for Multi-Task Learning*](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)
  (NeurIPS 2020), and the [official PCGrad repository](https://github.com/tianheyu927/PCGrad)
  plus the [named PyTorch implementation](https://github.com/WeiChengTseng/Pytorch-PCGrad).
  The experiment
  uses the requested asymmetric form: if the JEPA and NTP gradients conflict,
  only the conflicting component of JEPA is removed.
- [Liu et al., *Conflict-Averse Gradient Descent for Multi-task Learning*](https://openreview.net/pdf?id=_61Qh8tULj_)
  (NeurIPS 2021), and the [official CAGrad repository](https://github.com/Cranial-XIX/CAGrad).
  The two-objective bounded dual uses
  the official `c=0.5`, `1e-4` numerical constants, and final `1/(1+c)`
  rescaling.
- [Du et al., *Adapting Auxiliary Losses Using Gradient Similarity*](https://www.gatsby.ucl.ac.uk/~balaji/CL-NeurIPS2018-adapt.pdf)
  (NeurIPS 2018).  The implemented weighted published rule is
  `g_N + max(0, cosine(g_N,g_J)) g_J`; this is not a project-designed cosine
  threshold or binary gate.

The implementation computes a three-scalar Gram matrix over the aligned LoRA
gradient tensors instead of allocating flattened multi-million-element
vectors.  This is algebraically identical to applying the published vector
formulas to a concatenated parameter gradient.

## Endpoint objective and maintained organization

The endpoint cLM-JEPA branch uses:

\[
L_{JEPA}=\operatorname{MSE}(h_s,h_t)
+ \frac{4(0.01)}{0.99}\operatorname{SIGReg}(\{h_s,h_t\}),
\]

\[
L=L_{NTP}+\lambda_{active}L_{JEPA}.
\]

The following were removed from `src/`:

- PCSF loss/statistic, reference caching, collation, VJP hooks, configuration,
  and metrics;
- projector construction, optimization, checkpointing, evaluation, and the
  projection-space active condition;
- experimental gradient hooks unrelated to the four reported combination
  rules.

Historical PCSF and projection scripts, reports, checkpoints, and artifacts
remain available. The projection-head definition was moved to
`scripts/historical_projection.py`; it is not imported by production code.
A case-insensitive active-path scan found no PCSF/projector/rho/reference-spread
terms under `src/`.

The later dense causal V-JEPA-2.1-style branch is implemented separately in
`src/vjepa2_1.py` and selected as its own training family. It does not combine
with this endpoint objective or its gradient-interaction rules.

## Exact gradient construction

For each logical batch, one cadence draw is shared by all conditions.  On an
active step:

1. A no-gradient endpoint-only pass collects all 16 source and target EOS
   states without invoking the vocabulary projection.
2. Raw MSE and the exact 16-sample-per-view SIGReg statistic are evaluated on
   that complete logical batch.  Autograd obtains their endpoint VJPs.
3. The physical four-example chunks are replayed with their exact RNG states.
   Separate `torch.autograd.grad` calls accumulate
   `g_N = grad(L_NTP)` and `g_J = grad(L_JEPA)`.
4. Frozen parameters are never included.  The selected interaction rule is
   applied only to 308 LoRA A/B tensors (6,307,840 parameters), as required.
   ChemFM's two trainable PEFT `modules_to_save` token-I/O tensors retain the
   ordinary weighted-sum gradient.
5. The established global-norm clipping, fused AdamW update, and cosine
   schedule run unchanged.

The no-gradient statistics pass and gradient-bearing replay use the same
dropout RNG stream.  Checkpoints preserve Python, NumPy, CPU/CUDA, dataloader,
JEPA-cadence, and SIGReg-slice states.

`lambda_eff` is the cadence-adjusted expected coefficient used throughout the
earlier controlled reports.  With 50% JEPA activity,
`lambda_active=lambda_eff/0.5`; therefore the four controls have active-step
coefficients 0.5, 1, 2, and 4, respectively.  PCGrad, CAGrad, and auxiliary
similarity use the historical `lambda_eff=1` control, hence active coefficient
2.  Both the raw unweighted and applied weighted JEPA/NTP norm ratios are
logged so this convention cannot be mistaken for an unreported scale change.

## Frozen protocol

| Setting | Value |
|---|---|
| Model | ChemFM-1B |
| Dataset | USPTO-MIT synthesis |
| Train / internal training-time validation | 1,280 / 2 frozen rows, matching the historical controlled jobs |
| External diagnostic / official generation panels | 256 / 256 frozen reactions |
| Seed | 533 |
| LoRA / trainable model state | unchanged controlled configuration |
| Epochs / optimizer updates | 4 / 320 |
| Physical / accumulation / logical batch | 4 / 4 / 16 |
| JEPA readout | k=0 source and target EOS, final transformer state |
| JEPA cadence | Bernoulli 0.5, one draw per logical update |
| SIGReg | exact 16 samples/view, 1,024 slices, 17 knots on [0,3] |
| Optimizer | fused AdamW, lr 1e-4, betas (0.9,0.999), eps 1e-8, wd 0.01 |
| Scheduler | cosine with 5% warmup, min lr 1e-5 |
| Attention | SDPA |
| Hardware | one retained RTX A6000 48 GB / six-vCPU instance, pinned PyTorch 2.3.0+cu121 |

The seven training conditions are four raw weighted sums (`lambda_eff` 0.25,
0.5, 1, 2), then PCGrad, CAGrad, and Du auxiliary similarity at
`lambda_eff=1`. Following the explicit cost constraint, the retained instance
executes the matrix sequentially. A completed `lambda_eff=1`
artifact from a briefly provisioned snapshot-identical A6000 was copied onto
the retained instance before that clone and the reusable snapshot were
deleted; all unfinished conditions were discarded and restarted sequentially.
Every condition runs 320 serial optimizer updates. The full test suite passed
on the retained instance before its first optimizer step.

After all training and representation/gradient diagnostics completed, the user
explicitly narrowed the costly behavioral endpoint: do not run further weight
conditions and limit every retained official comparison to 256 reactions. The
behavioral set is therefore native, historical direct MSE+SIGReg,
`lambda_eff=0.25` as a low-weight control, PCGrad, CAGrad, and auxiliary
similarity. The 256 reactions are rows 0-255 of the already frozen
`prespecified_stage1_1280` manifest, selected without reference to any model
outcome. Existing native, direct, and lambda-0.25 predictions are identity-
checked and sliced to that exact order; the three gradient-interaction methods
are generated directly on the same manifest. The lambda-0.25 1,280-reaction
run completed just before this scope change and is retained as an out-of-scope
artifact, not mixed into the 256-reaction primary table. A lambda-0.5 official
run was stopped after 52 reactions and is likewise excluded. This user-directed
mid-execution scope revision reduces endpoint power and is reported explicitly.

### Retained execution record

The completed weighted-sum controls each contain 320 updates and the same 172
active JEPA updates. Their mean active-update `(cosine, conflict fraction, raw
JEPA/NTP norm ratio, applied JEPA/NTP norm ratio)` values were:

| `lambda_eff` | Mean tuple |
|---:|---:|
| 0.25 | `(0.00352, 0.5407, 0.09829, 0.04914)` |
| 0.5 | `(0.00466, 0.6453, 0.09914, 0.09914)` |
| 1.0 | `(-0.02637, 0.8081, 0.08985, 0.17970)` |
| 2.0 | `(-0.03769, 0.8372, 0.06889, 0.27556)` |

The implementation checkpoint recorded during execution was commit `07fac75`.
The completed consolidated artifact is
`runs/gradient_interaction/a6000/endpoint_256/summary.json`. The superseded
execution-handoff report was removed after these nonduplicated details were
incorporated here.

## Verification and evaluation

Before full training:

- local and A6000 suites passed 84 tests with one intentional skip;
- the A6000 two-update physical-4 preflight exercised one inactive and one
  active step, peaked at 10.27 GB, and emitted finite diagnostics;
- its active PCGrad step observed cosine -0.4952, raw unweighted JEPA/NTP norm
  ratio 0.04112, and a 4.23% modification relative to the raw summed gradient;
- direct/streamed logical-batch objective and VJP tests remained green.

Representation evaluation avoids repeated tokenization inside a condition,
batches endpoint extraction, requests the backbone endpoint without the LM
vocabulary projection, vectorizes prototype/retrieval scoring into one matrix
multiply, and computes effective rank from the equivalent smaller sample Gram
eigenspectrum.  On a 256x2048 RTX 4050 benchmark it ran in 0.289 s versus
1.139 s (3.94x).  Retrieval outputs were exact; other scalar differences were
at most 5.7e-6, except effective rank's 1.07e-4 SVD/eigendecomposition
roundoff.

Official beam-10 generation was reprofiled after its original four-worker
path produced only about 0.11 complete five-view reactions/s early in the
first endpoint.  The optimization followed NVIDIA's
[profile-first guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#application-profiling),
[CUDA Graph guidance](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/best-practices.html),
and Hugging Face's description of the
[dynamic/static KV-cache tradeoff](https://github.com/huggingface/transformers/blob/main/docs/source/en/cache_explanation.md).
One warmed baseline view took 3.0835 s: 2.0575 s of model-forward CUDA work
and an upper bound of 1.0260 s between forwards.  CPU profiling exposed about
10,000 tiny graph replays per view plus scalar-synchronizing beam scoring.

The exact final evaluator therefore:

- keeps dynamic sequence-length views but appends into a double-buffered,
  preallocated KV cache instead of concatenating the full prefix each layer;
- transfers each beam candidate vector in one operation instead of calling
  `.item()` for each candidate;
- captures each decoder layer as two graphs around the only dynamic-shape
  operation (cache update/repeat and SDPA), preserving the original operation
  order inside both segments; and
- uses three independent batch-1 workers, the fastest exact configuration
  after three-versus-four tie-breaking.

The final warmed view took 1.6014 s (1.93x faster), with model-forward CUDA
time 1.0776 s (1.91x faster).  On the same 24-reaction parity panel used by
the prior endpoint optimization, three-worker steady-state throughput was
0.28181 reactions/s and end-to-end throughput was 0.25230 reactions/s;
mean GPU utilization was 76.7%.  Every ordered raw candidate list, canonical
candidate list, ranked list, and exact flag matched the original evaluator for
all 24 reactions, and the merged prediction SHA-256 was identical.  This is
1.82x the prior report's selected 0.15496 reactions/s benchmark rate.

Rejected candidates were not used: static cache, merged LoRA, left-padded
batch-2, and equal-length batch-2 all changed raw candidate lists; static
cache and merged LoRA were also slower on the frozen eight-reaction screen.
The final generation code path changes evaluation runtime only, not prompts,
beam semantics, candidate canonicalization, or scoring.

Training received a separate A6000 profile rather than assuming that decoder
optimizations transferred.  A synchronized 80-update epoch spent 419.69 s
(86.8%) in the separate NTP/JEPA forward-backward path, 58.92 s (12.2%) in the
logical-batch endpoint-statistics/VJP pass, 4.13 s (0.85%) in optimizer work,
and only 0.66 s (0.14%) loading data.  Thirty-one half-second samples averaged
35.4% GPU utilization while retaining 22.47 GB, with 77% peak utilization.
This rules out tokenization, loader workers, and host-to-device transfer as
meaningful remedies.  The one-active-update PyTorch trace also showed
`aten::_scaled_dot_product_efficient_attention`; SDPA was already selecting a
fused memory-efficient backend rather than mathematical attention.  This
matches NVIDIA's advice to identify launch gaps before applying graphs
([performance troubleshooting](https://docs.nvidia.com/dl-cuda-graph/latest/troubleshooting/performance-issues.html)).

Five-update, cadence-matched training candidates were then tested sequentially
and compared at every saved adapter tensor.  Removing five inadvertent
synchronization barriers per update was bit exact but throughput-neutral
(35.37 s synchronized versus 35.44 s asynchronous).  Reusing the first or last
gradient-bearing microbatch as one endpoint pass was also bit exact but gained
less than 1% (35.25/35.06 s), so neither reuse branch remains in production.
PyTorch's documented
[batched VJP](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html)
was materially faster (28.25 s, 1.252x), but changed adapter elements by as much
as 1.831e-4 and changed the tiny validation generation panel after only five
updates; it was rejected under the no-correctness-trade rule.  The production
trainer therefore keeps sequential objective VJPs, makes synchronization
conditional on explicit profiling, avoids a redundant preliminary write of all
308 LoRA gradients, and applies final coefficients with exact multi-tensor CUDA
operations.  No rejected VJP/reuse switch remains in the active trainer.

Each epoch-4 training condition receives:

- source/target variance, effective rank, pair-center spread,
  mean-direction energy, cosine margins, retrieval/MRR, PCA spectrum, and
  top-two-PC residual retrieval;
- held-out LoRA-gradient audits at epochs 1, 2, and 4 for raw MSE, raw SIGReg,
  full active-weighted auxiliary, and the selected combiner.

The retained behavioral conditions additionally receive one-view 256-reaction
beam-10 generation with per-reaction normalized target-token CE/rank and a
256-reaction, five-official-R-SMILES-view, beam-10 endpoint. Paired native and
direct-MSE+SIGReg comparisons use the identical ordered reaction identities.

## Results

All seven four-epoch training runs, all seven representation evaluations, and
all 21 held-out-gradient audits completed.  The cost-revised behavioral
endpoint also completed for every retained condition.  Each of the six
official prediction files contains 256 unique reactions in exact manifest
order (`panel_index` 0 through 255).  The consolidated, machine-readable table
is `runs/gradient_interaction/a6000/endpoint_256/summary.json`.

### Training-time gradient interaction

These statistics cover the same 172 JEPA-active optimizer updates in each
run.  `raw ratio` is the unweighted JEPA/NTP LoRA-gradient norm ratio;
`modification` is the final combiner's change relative to ordinary summed
gradients.

| Condition | Mean cosine | Conflict | Raw ratio | Mean modification | Mean Du gate |
|---|---:|---:|---:|---:|---:|
| lambda_eff 0.25 | +0.0035 | 54.1% | 0.0983 | 0.0% | - |
| lambda_eff 0.5 | +0.0047 | 64.5% | 0.0991 | 0.0% | - |
| lambda_eff 1.0 | -0.0264 | 80.8% | 0.0899 | 0.0% | - |
| lambda_eff 2.0 | -0.0377 | 83.7% | 0.0689 | 0.0% | - |
| PCGrad | -0.0300 | 82.6% | 0.0912 | 1.14% | - |
| CAGrad | -0.0706 | 86.6% | 0.0593 | 67.86% | - |
| Du auxiliary similarity | -0.0050 | 52.3% | 0.1188 | 21.63% | 0.0153 |

PCGrad changed the summed update by `1.14%` on average while conflict was
recorded on `82.6%` of active updates. CAGrad changed it by `67.86%`. Du
adaptation's mean gate was `0.0153` and it set the gate to zero on conflicting
steps.

In the independently replayed held-out NTP audit, the full auxiliary cosine was
negative in `20/21` epoch-state audits. At epoch 4, the cosine triplets below are
`MSE / SIGReg / full auxiliary`; the last column is the active-weighted full
auxiliary norm relative to held-out NTP.

| Condition | Epoch-4 held-out cosines | Full-aux/NTP norm | Combiner change |
|---|---:|---:|---:|
| lambda_eff 0.25 | +0.020 / -0.015 / -0.011 | 0.058 | 0.0% |
| lambda_eff 0.5 | +0.013 / -0.014 / -0.011 | 0.085 | 0.0% |
| lambda_eff 1.0 | -0.014 / -0.039 / -0.048 | 0.186 | 0.0% |
| lambda_eff 2.0 | -0.327 / -0.022 / -0.092 | 0.202 | 0.0% |
| PCGrad | +0.039 / -0.032 / -0.019 | 0.164 | 0.31% |
| CAGrad | -0.040 / -0.144 / -0.145 | 0.106 | 69.26% |
| Du auxiliary similarity | +0.009 / -0.008 / -0.005 | 0.207 | 20.27% (gate 0) |

Across epochs 1, 2, and 4, SIGReg was negatively aligned in 19 of 21 audits;
the two positive measurements occurred on the Du trajectory. The full
auxiliary was negative in 20 of 21 audits.

### Representation geometry

All entries use the same frozen 256-reaction diagnostic.  Variance and
pair-center spread are in raw final-transformer space.  `Mean energy` is the
source/target fraction in the mean direction; lower is less common-direction
dominated.  Retrieval is four-way raw pair retrieval.

| Condition | Variance S / T | Center spread | Eff. rank S / T | Mean energy S / T | Cosine margin | Retrieval |
|---|---:|---:|---:|---:|---:|---:|
| Native ChemFM | .02488 / .02233 | .11622 | 25.79 / 23.04 | .6238 / .6625 | .06211 | 40.6% |
| lambda_eff 0.25 | .00167 / .00201 | .03549 | 43.18 / 25.17 | .9766 / .9716 | .00792 | 74.2% |
| lambda_eff 0.5 | .00133 / .00338 | .03888 | 38.61 / 11.56 | .9788 / .9437 | .00870 | 73.8% |
| lambda_eff 1.0 | .00188 / .00192 | .03854 | 35.92 / 29.35 | .9605 / .9576 | .01881 | 85.2% |
| lambda_eff 2.0 | .00098 / .00082 | .02585 | 44.08 / 38.49 | .9672 / .9702 | .01237 | 85.2% |
| PCGrad | .00188 / .00197 | .03817 | 40.27 / 34.48 | .9617 / .9583 | .01813 | 84.0% |
| CAGrad | .00071 / .00064 | .02226 | 50.60 / 41.13 | .9498 / .9499 | .01751 | 82.4% |
| Du auxiliary similarity | .00166 / .00153 | .03342 | 39.54 / 33.11 | .9789 / .9804 | .00628 | 66.4% |

All trained source variances were between `.00071` and `.00188`, versus native
`.02488`; target variances were between `.00064` and `.00338`, versus native
`.02233`. Retrieval was `66.4%–85.2%`, versus native `40.6%`. CAGrad had the
lowest variance and pair-center spread; Du had the lowest trained-condition
retrieval.

### One-view autoregressive behavior

This endpoint directly measures the ordinary decoder, normalized target-token
CE, and beam-10 product rank.  CE delta is relative to native, so positive is
worse.  The historical direct condition is the validated epoch-4 MSE+SIGReg
baseline from report 02 on the same 256 reactions.

| Condition | Top-1 | Top-3 | Top-5 | Top-10 | Valid candidates | Target-token CE | CE delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native | 2.34% | 10.16% | 15.63% | 20.31% | 78.20% | .240683 | - |
| Direct MSE+SIGReg | 2.34% | 9.38% | 13.28% | 19.14% | 86.60% | .248779 | +3.36% |
| PCGrad | 2.73% | 8.59% | 14.06% | 17.58% | 80.39% | .257099 | +6.82% |
| CAGrad | 2.34% | 5.86% | 12.11% | 19.14% | 86.76% | .295247 | +22.67% |
| Du auxiliary similarity | 1.17% | 8.98% | 13.67% | 18.75% | 86.76% | .243893 | +1.33% |

PCGrad mean per-reaction native-minus-
PCGrad CE was -0.01681, bootstrap 95% CI [-0.02333,-0.01051], Wilcoxon
`p=6.44e-8`. CAGrad was -0.06015,
[-0.06650,-0.05405], `p=3.89e-39`).  Du filtering came closest to native CE:
-0.00168, [-0.00743,+0.00388], `p=0.962`.

### Official five-view generation

The table is the final frozen 256-reaction comparison.  `Delta` and the paired
bootstrap CI are against native top-1.  Exact McNemar tests use the same
reaction identities.

| Condition | Top-1 | Top-3 | Top-5 | Top-10 | Top-1 delta [95% CI] | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| Native | 4.30% | 19.53% | 26.95% | 39.06% | - | - |
| Historical direct MSE+SIGReg | 3.52% | 18.75% | 26.17% | 37.11% | -0.78 pp [-3.13,+1.56] | .754 |
| lambda_eff 0.25 | 1.95% | 15.23% | 24.22% | 36.72% | -2.34 pp [-4.69,-0.39] | .070 |
| PCGrad | 3.91% | 16.41% | 25.00% | 35.55% | -0.39 pp [-2.73,+1.95] | 1.000 |
| CAGrad | 2.34% | 13.67% | 25.78% | 35.94% | -1.95 pp [-4.30,0.00] | .125 |
| Du auxiliary similarity | 3.91% | 19.53% | 26.56% | 35.94% | -0.39 pp [-2.73,+1.95] | 1.000 |

PCGrad and Du were each +0.39 pp versus historical direct top-1, with the same
paired CI [-1.17,+1.95] pp and McNemar `p=1.0`. CAGrad was -1.17 pp versus
direct (`p=.453`). Du equaled native top-3 and was below native at top-1,
top-5, and top-10. All trained conditions had 100% valid aggregated ranked
candidates.

The three newly generated official endpoints took 954.9-982.6 seconds each
(15.9-16.4 minutes), including model load.  End-to-end throughput was
0.2605-0.2681 reactions/s, mean GPU utilization 91.1-91.6%, and mean power
193.1-194.1 W.  Native, historical direct, and lambda-0.25 were identity-sliced
from completed exact five-view runs; they were not wastefully regenerated.

## Recorded comparisons

- The retained lambda-0.25 control recorded top-1 `1.95%`; native top-1 was
  `4.30%`.
- PCGrad recorded `82.6%` conflicting active updates, `1.14%` mean update
  modification, top-1 `3.91%`, and CE delta `+6.82%`.
- CAGrad recorded `67.86%` mean update modification, top-1 `2.34%`, and CE
  delta `+22.67%`.
- Du auxiliary similarity recorded mean gate `0.0153`, top-1 `3.91%`, and CE
  delta `+1.33%`.
- Historical direct MSE+SIGReg recorded top-1 `3.52%` and CE delta `+3.36%` on
  the same 256-reaction comparison.
