# USPTO-MIT frozen SIGReg gradient-response assay (RTX 4050)

## Decision

**A. Proceed with batch-16 SIGReg**, using the already-proposed symmetric k=0 objective, LeJEPA two-view `lambda_sig=0.01`, and no target detachment. This recommendation is conditional on pairing the future run with a native control that uses the same effective batch 16 and 160-update/two-epoch schedule, because exact SIGReg-16 necessarily changes the established optimizer cadence.

The frozen assay does **not** reproduce the hypothesis that SIGReg gradients vanish in cLM-JEPA's empirically contracted state. At the most contracted symmetric checkpoint, the absolute source/target SIGReg endpoint-gradient norms remain `0.0391/0.0392`, close to base ChemFM's `0.0427/0.0419`. With the proposed coefficient, SIGReg has 42.9% of the full parameter-space cosine-gradient norm and 50.1%/50.6% of the source/target endpoint cosine-gradient norms. This is meaningful residual force, is reproduced by three fixed projection draws, and is nearly symmetric across the two branches.

The qualification is directional: at the symmetric checkpoint SIGReg is only weakly opposed to cosine in parameter space (`cos=-0.069`), whereas the explicit VISReg scale term is strongly opposed (`cos=-0.942`). After coefficient mapping, the components opposing cosine are approximately 2.97% and 4.41% of the cosine norm, respectively. Explicit scale regularization is therefore more directionally efficient at this single state, but SIGReg has not lost its gradient, does not materially conflict with NTP, and is stronger at the partially contracted target-stop-gradient checkpoint. The evidence does not satisfy decision B's premise that SIGReg materially weakens while scale remains strong, nor decision C's premise that neither regularizer addresses the conflict.

No optimizer was constructed, no parameter was updated, and no training, generation, retrieval, interventions, PCA, decoder coupling, or checkpoint selection was run. Raw evidence is in [`runs/diagnostics/sigreg_gradient_response.json`](../../runs/diagnostics/sigreg_gradient_response.json); implementation is [`src/diagnose_sigreg_gradients_rtx4050.py`](../../src/diagnose_sigreg_gradients_rtx4050.py).

## Frozen protocol

The assay reused the first deterministic 16-reaction group from the batch-16 preflight: seed-533 permutation of the fixed 1,280-row Gate 4 USPTO-MIT training manifest. Every checkpoint received identical ChemFM serialization, token IDs, masks, physical recomputation batches of two, source k=0 final `<eos>`, and target final `<eos>`. All `Dropout` probabilities and attention-dropout fields were set to zero. Gradient checkpointing remained enabled only to fit the frozen backwards on the RTX 4050.

| # | Manifest row (zero-based) | Source SHA-256 prefix | Target SHA-256 prefix | Source/target characters |
|---:|---:|---|---|---:|
| 1 | 708 | `3a6e4f8b2502` | `6b3fbe656dcb` | 98 / 74 |
| 2 | 936 | `5a3070d9d0b0` | `25289b1d0a42` | 35 / 21 |
| 3 | 620 | `4dd6dacac645` | `badc3550a0a9` | 82 / 56 |
| 4 | 543 | `a8a18ca954d2` | `98631dde8275` | 87 / 47 |
| 5 | 431 | `1226674d4ce2` | `868aa24c1540` | 50 / 41 |
| 6 | 1188 | `5789539a520c` | `94d3923abb21` | 254 / 62 |
| 7 | 1270 | `980920ab3925` | `dafb57e9c301` | 34 / 32 |
| 8 | 1106 | `687d0d7585ac` | `a9973ccd5387` | 51 / 31 |
| 9 | 930 | `e05062961c50` | `a966fa3146f3` | 75 / 57 |
| 10 | 887 | `9afa8584d5d9` | `0afc1de43d62` | 116 / 61 |
| 11 | 1020 | `d9eba22ece8f` | `6a3bb0655198` | 53 / 29 |
| 12 | 963 | `47c82c23f925` | `e1dd3ecaaa0a` | 91 / 46 |
| 13 | 1228 | `eb67f963354f` | `05b28619dd4f` | 82 / 62 |
| 14 | 234 | `2ee516126539` | `cd7ef91eedcd` | 82 / 35 |
| 15 | 1207 | `1dcd4b6fc3d8` | `264853014c14` | 127 / 82 |
| 16 | 454 | `60b35b4f487c` | `30983217c726` | 61 / 28 |

Full SMILES and hashes are retained in the JSON artifact.

| Label | Existing state | Historical updates | Historical readout | Assay readout |
|---|---|---:|---|---|
| Base | pretrained ChemFM-1B | 0 | none | k=0 source EOS |
| Native e2 | seed-533 native epoch 2 | 320 | none | k=0 source EOS |
| Symmetric e2 | original seed-533 symmetric cLM-JEPA epoch 2 | 320 | k=1 `[PRED]` | k=0 source EOS |
| Target-SG e2 | seed-533 target-stop-gradient epoch 2 | 320 | k=1 `[PRED]` | k=0 source EOS |
| SIGReg-128 e2 | seed-533 batch-128 SIGReg endpoint | 20 | k=0 source EOS | k=0 source EOS |

Using k=0 for every frozen state avoids a readout-definition confound. It does not recreate the k=1 predictor used to train two historical checkpoints; it asks how the proposed future k=0 representation behaves under each existing shared-model state. SIGReg-128 is a healthy-geometry anchor, not an exposure-matched training comparison.

## Objective definitions and literature check

The [LeJEPA paper](https://arxiv.org/abs/2511.08544) and [official LeJEPA code](https://github.com/galilai-group/lejepa) define SIGReg through the Epps-Pulley empirical-characteristic-function statistic over random one-dimensional projections. This assay exactly reuses the repository's verified implementation: 17 trapezoidal knots on `[0,3]`, 1,024 normalized Gaussian projections, source and target statistics computed independently and averaged. The primary projections use seed 533 and are identical across checkpoints (direction SHA-256 `7ee375aa…f9724cddf`); seeds 917 and 1907 provide two cheap endpoint-gradient repetitions.

Losses are:

- `L_NTP`: ordinary target-token ChemFM cross-entropy;
- `L_cos = mean_i [1 - cos(z_s,i, z_t,i)]`;
- `L_SIGReg`: the two-view mean Epps-Pulley statistic at `N=16` per view;
- `L_scale = mean_(view,dimension) [1 - (||z - mean(z)||_2 / sqrt(16) + 1e-6)]^2`.

The scale loss is the scale component in the [VISReg paper and official implementation](https://arxiv.org/abs/2606.02572), without its centering or sliced-Wasserstein shape terms. VISReg explicitly separates scale and shape, and its published algorithm uses the squared distance of each centered feature standard deviation from one. The small `1e-6` is the official implementation's numerical stabilizer. For comparison, [VICReg](https://arxiv.org/abs/2105.04906) and its [official code](https://github.com/facebookresearch/vicreg) use a different one-sided floor, `mean ReLU(1-sqrt(var+1e-4))`; that hinge was verified but not blended into `g_variance`.

LeJEPA combines `(1-lambda_sig)L_cos + lambda_sig L_SIGReg`. To preserve this project's fixed cosine strength:

`L_aux = L_cos + (0.01/0.99)L_SIGReg`.

With 50% auxiliary dropout and active outer coefficient two, expected weights are `1.0` for cosine and `0.010101` for SIGReg. The scale term is shown with the same hypothetical regularizer-slot coefficient solely for coefficient-normalized comparison; it was not trained or selected. “Weighted” below means expected over dropout; an active update doubles both cosine and regularizer values, leaving their ratio unchanged.

## Geometry along the observed trajectory

Effective rank is sample-limited to at most 15 and is included to distinguish contraction from rank-one collapse.

| Checkpoint | Source variance | Target variance | Source mean energy | Target mean energy | Source rank | Target rank |
|---|---:|---:|---:|---:|---:|---:|
| Base | 2.113e-2 | 1.966e-2 | 0.7013 | 0.7157 | 11.03 | 9.06 |
| Native e2 | 1.210e-3 | 1.950e-2 | 0.9845 | 0.6709 | 10.61 | 1.65 |
| Symmetric e2 | **7.106e-5** | **4.958e-5** | **0.99917** | **0.99942** | 11.42 | 9.45 |
| Target-SG e2 | 2.946e-4 | 2.880e-4 | 0.99626 | 0.99636 | 11.80 | 10.20 |
| SIGReg-128 e2 | 1.394e-2 | 1.384e-2 | 0.8087 | 0.8201 | 9.45 | 8.72 |

The fixed batch reproduces the established trajectory: symmetric JEPA has 297-fold/397-fold lower source/target variance than base while retaining multidimensional rank; target stop-gradient partially relaxes it; SIGReg-128 restores raw scale. This is extreme variance contraction/common-direction concentration, not evidence of classical rank collapse.

## Losses and full parameter gradients

Scalar losses are reported for completeness, not used as substitutes for gradients.

| Checkpoint | NTP loss | Cosine loss | SIGReg loss | Scale loss |
|---|---:|---:|---:|---:|
| Base | 1.18284 | 0.30469 | 6.63997 | 0.84982 |
| Native e2 | 0.12107 | 0.48633 | 7.04742 | 0.91551 |
| Symmetric e2 | 0.15020 | 0.001709 | 7.64725 | 0.99079 |
| Target-SG e2 | 0.12628 | 0.004395 | 7.53215 | 0.97967 |
| SIGReg-128 e2 | 1.02954 | 0.15723 | 6.95989 | 0.87411 |

| Checkpoint | `||g_NTP||` | `||g_cos||` | `||g_SIG||` raw | `||g_scale||` raw | `||g_SIG||` weighted | `||g_scale||` weighted |
|---|---:|---:|---:|---:|---:|---:|
| Base | 2.167e15 | 1.197e15 | 3.737e15 | 3.285e14 | 3.775e13 | 3.318e12 |
| Native e2 | 1.0876 | 38.7616 | 71.5788 | 14.2429 | 0.7230 | 0.1439 |
| Symmetric e2 | 1.2699 | 0.03681 | 1.5633 | 0.1708 | 0.01579 | 0.001725 |
| Target-SG e2 | 0.7290 | 0.4387 | 2.1986 | 0.4003 | 0.02221 | 0.004043 |
| SIGReg-128 e2 | 11.0316 | 13.4469 | 31.7932 | 3.1716 | 0.3211 | 0.03204 |

Base ChemFM's enormous shared-parameter gradients are the already-observed initial embedding/language-head regime and make absolute base parameter norms unsuitable as a contraction trend. Endpoint gradients below remove that Jacobian-scale confound.

| Checkpoint | SIG/cos raw | SIG/NTP raw | Scale/cos raw | Scale/NTP raw | SIG/cos weighted | SIG/NTP weighted | Scale/cos weighted | Scale/NTP weighted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 3.122 | 1.724 | 0.274 | 0.152 | 0.0315 | 0.0174 | 0.00277 | 0.00153 |
| Native e2 | 1.847 | 65.815 | 0.367 | 13.096 | 0.0187 | 0.6648 | 0.00371 | 0.1323 |
| Symmetric e2 | **42.463** | 1.231 | 4.638 | 0.134 | **0.4289** | 0.0124 | **0.04685** | 0.00136 |
| Target-SG e2 | 5.011 | 3.016 | 0.912 | 0.549 | 0.0506 | 0.0305 | 0.00922 | 0.00555 |
| SIGReg-128 e2 | 2.364 | 2.882 | 0.236 | 0.288 | 0.0239 | 0.0291 | 0.00238 | 0.00290 |

At severe contraction, both regularizers become larger relative to cosine primarily because cosine itself is close to its minimum and its gradient shrinks. The absolute endpoint results establish whether corrective gradients themselves disappear.

## Gradient directions

Negative cosine means the regularizer gradient opposes the other loss gradient; positive means alignment.

| Checkpoint | SIG vs cosine | SIG vs NTP | Scale vs cosine | Scale vs NTP | Cosine vs NTP |
|---|---:|---:|---:|---:|---:|
| Base | -0.772 | -0.873 | -0.716 | -0.844 | +0.589 |
| Native e2 | +0.904 | +0.003 | +0.926 | -0.001 | -0.012 |
| Symmetric e2 | **-0.069** | -0.048 | **-0.942** | -0.766 | +0.761 |
| Target-SG e2 | **-0.727** | -0.013 | **-0.863** | -0.026 | +0.028 |
| SIGReg-128 e2 | -0.579 | +0.105 | -0.638 | +0.195 | -0.001 |

The native state is not on a cosine-optimized trajectory and has highly asymmetric geometry, so its positive regularizer/cosine alignment is a useful control rather than evidence about contraction caused by JEPA. In the two contracted JEPA checkpoints, both regularizers oppose cosine. At the symmetric endpoint, SIGReg's large norm is mostly orthogonal; scale is much more directly opposed. Multiplying weighted norm ratio by the opposing cosine component gives:

| Checkpoint | SIGReg opposing projection / `||g_cos||` | Scale opposing projection / `||g_cos||` |
|---|---:|---:|
| Base | 2.43% | 0.20% |
| Native e2 | 0% | 0% |
| Symmetric e2 | **2.97%** | **4.41%** |
| Target-SG e2 | **3.68%** | 0.80% |
| SIGReg-128 e2 | 1.38% | 0.15% |

Neither regularizer substantially conflicts with NTP after its proposed coefficient is applied. At symmetric e2, the opposing projection onto NTP is approximately 0.060% of `||g_NTP||` for SIGReg and 0.104% for scale.

## Source/target endpoint decomposition

These are exact representation-gradient norms before the shared ChemFM Jacobian. They cleanly separate source from target; inventing additive source/target parameter norms would be misleading because both branches share parameters.

| Checkpoint/view | `||g_cos||` | `||g_SIG||` | `||g_scale||` | SIG/cos raw | Scale/cos raw | SIG/cos weighted | Scale/cos weighted | SIG vs cos | Scale vs cos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base source | 1.481e-2 | 4.269e-2 | 5.083e-3 | 2.884 | 0.343 | 0.0291 | 0.00347 | -0.245 | -0.176 |
| Base target | 1.509e-2 | 4.185e-2 | 5.102e-3 | 2.773 | 0.338 | 0.0280 | 0.00341 | -0.226 | -0.169 |
| Native source | 1.573e-2 | 3.789e-2 | 5.412e-3 | 2.408 | 0.344 | 0.0243 | 0.00347 | -0.041 | -0.034 |
| Native target | 1.957e-2 | 3.973e-2 | 5.157e-3 | 2.030 | 0.264 | 0.0205 | 0.00266 | -0.331 | -0.197 |
| Symmetric source | 7.878e-4 | **3.907e-2** | **5.496e-3** | **49.59** | **6.977** | **0.5009** | **0.0705** | -0.028 | -0.283 |
| Symmetric target | 7.827e-4 | **3.919e-2** | **5.501e-3** | **50.07** | **7.029** | **0.5058** | **0.0710** | -0.043 | -0.181 |
| Target-SG source | 1.751e-3 | 3.762e-2 | 5.467e-3 | 21.48 | 3.121 | 0.2170 | 0.0315 | -0.024 | -0.265 |
| Target-SG target | 1.748e-3 | 3.756e-2 | 5.468e-3 | 21.48 | 3.128 | 0.2170 | 0.0316 | -0.050 | -0.252 |
| SIGReg-128 source | 1.098e-2 | 4.137e-2 | 5.164e-3 | 3.767 | 0.470 | 0.0380 | 0.00475 | -0.256 | -0.225 |
| SIGReg-128 target | 1.061e-2 | 4.183e-2 | 5.166e-3 | 3.944 | 0.487 | 0.0398 | 0.00492 | -0.238 | -0.211 |

SIGReg's absolute endpoint norm stays in the narrow range `0.0376–0.0427` from healthy through severely contracted states; the scale gradient stays `0.00508–0.00550`. There is no source-only or target-only failure. For symmetric e2, the raw SIGReg/cosine endpoint ratios across direction seeds 533/917/1907 are `47.40–49.59` on source and `47.79–50.07` on target, so the result is not a favorable single projection draw.

Plots:

- [Source variance versus corrective-gradient ratio](../../runs/diagnostics/sigreg_gradient_response_source.svg)
- [Target variance versus corrective-gradient ratio](../../runs/diagnostics/sigreg_gradient_response_target.svg)

## Interpretation relative to VISReg's claim

VISReg reports that Epps-Pulley/SIGReg gradients diminish in a synthetic trajectory where the feature norm itself is scaled toward zero. That primary-paper claim was inspected before this result and is not dismissed. It does not describe the exact cLM-JEPA pathology observed here: the centered reaction-to-reaction variance contracts around a large shared direction, with 99.9% mean-direction energy and nonzero representation norm. SIGReg penalizes the full mismatch from a zero-mean isotropic Gaussian, so it can retain gradient through the mean/scale mismatch even when centered variance is tiny.

The empirical assay therefore distinguishes the hypotheses:

1. **SIGReg does not lose absolute gradient under this common-direction contraction.** Its source and target endpoint norms are essentially invariant across the trajectory.
2. **Explicit scale is more directionally targeted at the most contracted symmetric state.** Its parameter gradient strongly opposes cosine, but it is about nine times smaller in norm than SIGReg before the common coefficient.
3. **Neither has a material NTP conflict at the proposed coefficient.** The scale gradient's high negative cosine with NTP at symmetric e2 is offset by its very small weighted norm.
4. **The earlier 16-step smoke's flat/increasing SIGReg scalar cannot be interpreted as vanishing corrective force.** Frozen gradients directly show that the force remains; different batches and random slices made that short scalar trajectory a weak diagnostic.

## Precisely specified next experiment

Run the previously proposed **symmetric k=0 cosine-JEPA + exact SIGReg-16** condition for exactly two epochs with LeJEPA `lambda_sig=0.01`, mapped as

`L = L_NTP + active * 2 * [L_cos + (0.01/0.99)L_SIGReg]`,

using 50% auxiliary dropout and otherwise unchanged reliable seed-533 settings. Keep JEPA gradients through both source and target: this assay finds nearly identical corrective SIGReg force on both branches, while the prior target-stop-gradient experiment improved geometry only partially and did not establish better decoder coupling. Run an exposure-matched native batch-16 control under the same 80-updates-per-epoch scheduler semantics so the result is interpretable. Do not add the explicit scale term, VISReg shape loss, projector, EMA, or another objective in this experiment, and do not tune `lambda_sig` from these frozen outcomes.
