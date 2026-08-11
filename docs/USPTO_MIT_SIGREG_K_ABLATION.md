# USPTO-MIT SIGReg + symmetric-JEPA k ablation

## Question and controlled design

This experiment asks two separate questions:

1. Does standard SIGReg materially repair the common-direction/variance-contraction pathology without damaging native generation?
2. Under the same SIGReg-regularized symmetric JEPA objective, is LLM-JEPA k=0 or k=1 better for generation and decoder coupling?

Exactly two new seed-533 conditions were trained:

- **k=0:** no predictor token; the existing source `<eos>` final-layer state is used directly.
- **k=1:** append `[PRED]`; use its final-layer state.

Both use symmetric gradients through source and target. The target is not detached or frozen, and both branches continue to share the ordinary native next-token supervision. All other ChemFM/LoRA data, optimizer, scheduler, learning rate, batch/exposure, JEPA-loss dropout, serialization, and evaluation settings are matched to the existing reliable setup. Both runs stop after epoch 2 (320 optimizer steps); no outcome-based tuning, additional seed, or later epoch was run.

### SIGReg implementation

The implementation follows the official [LeJEPA repository](https://github.com/galilai-group/lejepa) at commit `c293d291ca87cd4fddee9d3fffe4e914c7272052` and the [LeJEPA paper](https://arxiv.org/abs/2511.08544): an Epps-Pulley empirical-characteristic-function statistic with 17 trapezoidal knots on `[0,3]` and 1,024 normalized Gaussian random slices. The source and target endpoint distributions are tested as separate views using the same random directions. SIGReg is applied directly to the JEPA representation space; no projector, normalization, stop-gradient, EMA target, or other objective modification is added.

LeJEPA's standard 0.05 trade-off is outer-rescaled to preserve the experiment's frozen cosine coefficient:

`L = L_native + lambda_actual * active * [L_cos + (0.05 / 0.95) * L_SIGReg]`

where `active` is the existing 50% JEPA-loss dropout and `lambda_actual=2`, preserving expected cosine weight 1. The two runs used the same independently seeded dropout sequence: 621/1,280 JEPA-active microbatches in each. SIGReg's effective sample count is the preserved physical batch size of two per view; gradient accumulation does not enlarge that statistic. This is an important scope limitation when interpreting a negative result, but changing the physical batch would violate the requested matched setup.

Focused tests verify the Epps-Pulley computation against an independent executable reconstruction, k=0 EOS selection, identical native behavior when disabled, and nonzero SIGReg/JEPA gradients through both source and target states. A real ChemFM-1B batch was finite for both k values. Peak allocated memory during full training was 5.14 GiB.

## Training behavior

| Condition | Epoch | Mean native loss | Mean cosine JEPA loss | Mean SIGReg loss | Mean combined JEPA objective |
|---|---:|---:|---:|---:|---:|
| SIGReg k=0 | 1 | 0.8830 | 0.1108 | 0.9232 | 0.1593 |
| SIGReg k=0 | 2 | 0.2446 | 0.00210 | 0.9420 | 0.05168 |
| SIGReg k=1 | 1 | 0.7612 | 0.1094 | 0.9300 | 0.1584 |
| SIGReg k=1 | 2 | 0.2469 | 0.00403 | 0.9485 | 0.05395 |

Cosine alignment again approaches its near-zero shortcut, while the SIGReg statistic does not decrease. Training therefore did not achieve the intended Gaussian-distribution constraint within the frozen two-epoch, physical-batch-two protocol.

Training wall time was 22.3 minutes for k=0 and 21.2 minutes for k=1. The W&B offline runs are `offline-run-20260810_222558-dtktw4iw` and `offline-run-20260810_224831-sdzo0unu`, respectively.

## Fixed 256-reaction generative evaluation

The evaluation panel is the first 256 identities by frozen panel index from the existing 512-identity decoder-coupling reference: one unique canonical reaction/product identity, one official enumeration, beam 10, and identical identities for every checkpoint. All primary comparisons below use epoch-2 checkpoints, including the existing references, so training exposure is matched.

| Epoch-2 condition | Exact top-1 | Top-3 | Top-5 | Top-10 | Valid-candidate rate | Target-token CE |
|---|---:|---:|---:|---:|---:|---:|
| Native | **0.027344 (7/256)** | 0.117188 | 0.175781 | 0.226562 | 0.809375 | **0.236778** |
| Original symmetric cLM-JEPA k=1 | 0.015625 (4/256) | 0.109375 | 0.160156 | 0.246094 | 0.852344 | 0.238418 |
| Target-stop-gradient k=1 | **0.035156 (9/256)** | **0.152344** | **0.222656** | **0.277344** | 0.780078 | **0.232881** |
| Symmetric SIGReg k=0 | 0.007812 (2/256) | 0.093750 | 0.136719 | 0.210938 | 0.871094 | 0.247966 |
| Symmetric SIGReg k=1 | 0.011719 (3/256) | 0.070312 | 0.125000 | 0.199219 | **0.880469** | 0.249483 |

Against native, k=0 has two shared top-1 successes, five native-only successes, and no k=0-only success. Its top-1 difference is -1.95 percentage points (paired bootstrap 95% interval `[-3.91,-0.39]`; exact McNemar p=0.0625). Its aggregate CE is 4.72% worse; mean per-reaction `native CE - k0 CE` is -0.01157 with 95% interval `[-0.01800,-0.00513]` and Wilcoxon p=0.000250.

Against native, k=1 has two shared top-1 successes, five native-only successes, and one k=1-only success. Its top-1 difference is -1.56 points (95% interval `[-3.52,0.00]`; McNemar p=0.2188). Its aggregate CE is 5.37% worse; mean per-reaction improvement is -0.01376 with 95% interval `[-0.02042,-0.00749]` and Wilcoxon p=0.0000689.

The higher valid-candidate fractions do not compensate for the loss of correct products. Both SIGReg conditions damage the primary metric and every reported top-k metric relative to exposure-matched native fine-tuning.

### Direct k=0 versus k=1

k=1 has one additional top-1 success: two reactions are correct under both, zero are k=0-only, and one is k=1-only (McNemar p=1.0). This is not meaningful evidence for k=1. At wider cutoffs the direction reverses:

| Cutoff | k=0-only correct | k=1-only correct |
|---|---:|---:|
| Top-1 | 0 | 1 |
| Top-3 | 7 | 1 |
| Top-5 | 6 | 3 |
| Top-10 | 7 | 4 |

k=0 also has lower CE (0.247966 versus 0.249483) and a slightly better mean correct-product rank; neither difference is precise. For k=0 versus k=1, the CE difference has p=0.182 and the mean-rank interval includes zero. Exact top-1 therefore does not distinguish the readouts, while the consistent secondary direction favors k=0.

## Same-identity representation geometry

### Source geometry

| Epoch-2 checkpoint/readout | Source variance | Source effective rank | Source mean-direction energy |
|---|---:|---:|---:|
| Base ChemFM k=1 | 0.027503 | 20.99 | 0.569844 |
| Native k=1 | 0.009297 | 21.77 | 0.873288 |
| Original symmetric k=1 | 0.00004454 | 39.37 | 0.999481 |
| Target-stop-gradient k=1 | **0.00017298** | 41.60 | **0.997808** |
| Symmetric SIGReg k=0 | 0.00017264 | **46.40** | 0.997814 |
| Symmetric SIGReg k=1 | 0.00012744 | 35.38 | 0.998454 |

### Target and pair geometry

| Epoch-2 checkpoint/readout | Target variance | Target effective rank | Target mean-direction energy | Raw pair margin | Raw retrieval top-1 |
|---|---:|---:|---:|---:|---:|
| Base ChemFM k=1 | 0.022246 | 21.71 | 0.663507 | 0.030049 | 0.378906 |
| Native k=1 | 0.008797 | 3.44 | 0.831499 | 0.003340 | 0.281250 |
| Original symmetric k=1 | 0.00004941 | **37.10** | 0.999425 | 0.000145 | 0.644531 |
| Target-stop-gradient k=1 | **0.00034236** | 32.28 | **0.995663** | **0.000742** | 0.492188 |
| Symmetric SIGReg k=0 | 0.00014086 | 29.02 | 0.998218 | 0.000577 | **0.742188** |
| Symmetric SIGReg k=1 | 0.00030683 | 10.91 | 0.996280 | 0.000413 | 0.566406 |

SIGReg produces relative movement away from the original symmetric baseline, but not healthy absolute geometry:

- k=0 target variance is 2.85 times the original symmetric value, yet remains 62.5 times below native and mean-direction energy is still 99.82%;
- k=1 target variance is 6.21 times the original symmetric value, yet remains 28.7 times below native and mean-direction energy is still 99.63%;
- source variance remains 53.9 times below native for k=0 and 73.0 times below native for k=1;
- neither SIGReg condition is healthier than the exposure-matched target-stop-gradient checkpoint across both source and target scale.

Large fold changes from the tiny original baseline therefore overstate the rescue. SIGReg only partially relaxes concentration and does not enforce an approximately isotropic Gaussian representation in this setup.

Pair-specific information remains strong in the residual. After joint centering and removing two shared PCs, margin/retrieval are 0.3433/0.8516 for k=0 and 0.2368/0.7461 for k=1. The failure is again not classical representation collapse.

## Source sensitivity and decoder coupling

| Readout | Contributor-removal raw sensitivity | Replacement raw sensitivity | Unrelated-source raw sensitivity | Unrelated-source decoder CE change |
|---|---:|---:|---:|---:|
| Native k=1 | 0.069450 | 0.053758 | 0.124279 | 0.842276 |
| Original symmetric k=1 | 0.000446 | 0.000162 | 0.000428 | 0.841096 |
| Target-stop-gradient k=1 | 0.002366 | 0.000694 | 0.001818 | 0.833917 |
| Symmetric SIGReg k=0 | **0.012815** | 0.000708 | **0.001839** | 0.778729 |
| Symmetric SIGReg k=1 | 0.001327 | 0.000461 | 0.001332 | 0.788411 |

k=0's EOS state is more sensitive to deleting the selected contributor, but this does not generalize to the stronger replacement or unrelated-source controls. Both SIGReg readouts remain roughly two orders of magnitude less sensitive than native under replacement, while the normal decoder incurs large CE changes. Residual sensitivities remain large (unrelated-source: 0.904 for k=0 and 0.859 for k=1), again showing chemistry buried in the residual rather than absent.

Neither condition shows positive decoder coupling. For k=0, raw/residual pair margins have correlations from -0.079 to -0.003 with CE or rank improvement, and every interval includes zero. For k=1, CE correlations also include zero; the only precise association is adverse, with stronger raw pair signal associated with worse rank improvement versus native (rho=-0.146, 95% interval `[-0.261,-0.027]`). This isolated negative association is not evidence that the decoder uses the JEPA relationship beneficially.

## Answers

### 1. Did SIGReg materially restore healthy geometry without damaging native generation?

**No.** It increased variance and reduced mean-direction concentration relative to the extreme original symmetric baseline, but the absolute scale remains tens of times below native, more than 99.6% of target energy remains in the mean direction, and SIGReg loss did not decrease. Both k values substantially underperform exposure-matched native generation and have significantly worse target-token CE. This negative result is specific to direct regularization of the 2,048-dimensional ChemFM endpoint space with the frozen physical batch of two and two-epoch budget; it does not establish that SIGReg is ineffective in other batch/projector regimes.

### 2. Under SIGReg + symmetric JEPA, is k=0 or k=1 better?

**Exact top-1 is inconclusive.** k=1 wins by one reaction, with only one discordant top-1 pair and p=1.0. The broader evidence favors k=0: higher top-3/5/10, lower CE, better mean rank, stronger raw/residual pair retrieval, and greater contributor-removal sensitivity. Neither readout restores generation or convincing decoder coupling, so this experiment does not support retaining `[PRED]` as beneficial under SIGReg; it also does not isolate k as the original failure's cause.

The requested experiment stops here. No SIGReg tuning, additional seed, later epoch, objective change, or task expansion was run.

## Evidence artifacts

- Training results: `runs/gate4_sigreg_k_ablation/sigreg-k{0,1}-s533.json`
- Epoch-2 checkpoints: `runs/gate4_sigreg_k_ablation/sigreg-k{0,1}-s533-checkpoints/epoch_2`
- Direct paired k comparison: `runs/diagnostics/sigreg_k_ablation_256/sigreg_k0_vs_k1_summary_256.json`
- Native comparisons: `runs/diagnostics/sigreg_k_ablation_256/native_epoch2_vs_sigreg_k{0,1}_summary_256.json`
- Existing-reference comparisons: `runs/diagnostics/sigreg_k_ablation_256/native_epoch2_vs_{symmetric,target_sg}_epoch2_summary_256.json`
- Generation outputs: `runs/diagnostics/sigreg_k_ablation_256/*_generation_256.jsonl`
- Per-reaction CE/pair/intervention diagnostics: `runs/diagnostics/sigreg_k_ablation_256/*_diagnostics_256.json`
- Matched epoch-2 geometry: `runs/diagnostics/sigreg_k_ablation_256/geometry_matched_epoch2_256.json`
