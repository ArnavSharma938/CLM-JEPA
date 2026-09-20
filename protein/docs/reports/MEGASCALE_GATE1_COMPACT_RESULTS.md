# MegaScale corrected-sign compact Stage-1 results

## Reporting scope

This report records the corrected-sign compact MegaScale pilot. It is not the definitive full-data experiment. It reports observed measurements, comparisons, uncertainty, preregistered gate outputs, and limitations without adding a mechanistic interpretation.

## Stability convention and provenance

The source is the original Tsuboyama MegaScale dataset. The original paper defines sufficiently positive ΔΔG values as stabilizing ([Tsuboyama et al., Nature 2023](https://www.nature.com/articles/s41586-023-06328-6)). Its raw `ddG_ML` convention therefore treats positive values as stabilizing. `PLAN.md` registers the opposite canonical convention: canonical `ddg < 0` is stabilizing and fitness is `-ddg`. Preprocessing performs exactly one conversion, `ddg = -ddG_ML`, at ingestion. Training, evaluation, and reporting consume only canonical `ddg` and apply no further sign correction.

The ingestion invariant found 3,583 eligible source rows with `ddG_ML >= 1`; all mapped to canonical `ddg <= -1`. All nonzero raw/canonical pairs had opposite signs. The SFT predicate `ddg < 0` selected 90,849 variants with positive source `ddG_ML`.

- Source CSV SHA-256: `0c7b2bf1f162122d83d5bb39b901cc446e436b5643684ca268f934be145d8398`
- Corrected eligible manifest SHA-256: `79278f2f6f668c94fcbe9be2281593bc216dc2de4bed3a594d3658bc1b181a2f`
- Corrected compact manifest SHA-256: `2827221cffff221d36200e4234c4cd1b358047b1c658b037896c6eaea71d1e44`
- Ordered compact sequence-ID SHA-256: `c6149717b91bc88b1587ddadbb2350b019d018c11be2c732dd304305f6212eda`

The eligible manifest contained 499,678 train, 10,789 validation, and 16,737 test records across 412 domains and 371 structures. Eligibility preceded structural clustering, split construction, and held-out structural-distance calculation.

The compact benchmark contained 11,016 training variants from 36 domains and 30 structural clusters, 960 validation variants from 6 domains and 2 held-out clusters, and 1,440 test variants from 9 domains and 3 held-out clusters. No cluster crossed splits. Every training domain contributed 306 canonical stabilizing variants; these comprised 10,319 single and 697 double mutants.

## Experimental configuration

Evidence conditions were FT, DTFT, LoRA-r8, and LoRA-r64, each with seeds 11 and 23. Training retained the registered objective, effective batch size 32, paired ordering, 0.1 Å coordinate noise, bf16 execution, validation-based checkpoint selection, and minimum two epochs. Validation ran every 0.5 epoch; early stopping used four validation checks without a new optimum and a ten-epoch safety cap.

FT and DTFT used learning rate `1e-7`. LoRA learning rates were selected separately with HPO seed 7 from `1e-6`, `3e-6`, `1e-5`, and `3e-5`: all candidates ran one epoch, the best two per rank continued to three epochs, and selection used minimum validation autoregressive NLL.

| Condition | Selected learning rate |
|---|---:|
| FT | `1e-7` |
| DTFT | `1e-7` |
| LoRA-r8 | `3e-5` |
| LoRA-r64 | `1e-5` |

## LoRA LR-screen trajectories

Entries list validation NLL at epochs 0, 0.5, 1.0 and, for finalists, 1.5, 2.0, 2.5, and 3.0.

| Rank | LR | Continued | Validation-NLL trajectory | Minimum |
|---|---:|:---:|---|---:|
| r8 | `1e-6` | no | 1.066259, 1.066278, 1.066217 | 1.066217 |
| r8 | `3e-6` | no | 1.066233, 1.066141, 1.065596 | 1.065596 |
| r8 | `1e-5` | yes | 1.066200, 1.064046, 1.062495, 1.061950, 1.062247, 1.063530, 1.065907 | 1.061950 |
| r8 | `3e-5` | yes | 1.066276, 1.061837, 1.065595, 1.088577, 1.155711, 1.257748, 1.348393 | 1.061837 |
| r64 | `1e-6` | no | 1.066289, 1.065958, 1.065187 | 1.065187 |
| r64 | `3e-6` | yes | 1.066246, 1.063906, 1.062204, 1.061324, 1.061379, 1.061251, 1.061299 | 1.061251 |
| r64 | `1e-5` | yes | 1.066228, 1.061075, 1.062287, 1.078234, 1.130592, 1.219842, 1.314219 | 1.061075 |
| r64 | `3e-5` | no | 1.066283, 1.071243, 1.271186 | 1.066283 |

Under the registered minimum-NLL rule, r8 selected `3e-5` and r64 selected `1e-5`. Both selected HPO trajectories subsequently deteriorated; this observation is recorded separately from checkpoint selection.

## Held-out measurements

Trained-condition values are mean ± sample standard deviation across seeds 11 and 23. Base was evaluated once. NLL is lower-is-better; the other metrics are higher-is-better.

| Condition | Held-out NLL | Mean domain Spearman | Mean domain AUROC | Sequence recovery |
|---|---:|---:|---:|---:|
| Base | 1.456673 | 0.648275 | 0.782394 | 0.592254 |
| FT | 1.403119 ± 0.005902 | 0.663595 ± 0.001049 | 0.790407 ± 0.000451 | 0.586874 ± 0.000058 |
| DTFT | 1.411758 ± 0.000036 | 0.665845 ± 0.000312 | 0.791637 ± 0.000168 | 0.587600 ± 0.000340 |
| LoRA-r8 | 1.408387 ± 0.000107 | 0.666505 ± 0.000644 | 0.793562 ± 0.000246 | 0.591643 ± 0.001190 |
| LoRA-r64 | 1.400673 ± 0.012763 | 0.668259 ± 0.002482 | 0.794000 ± 0.001505 | 0.587485 ± 0.002309 |

The individual held-out NLL values were FT 1.407292/1.398946, DTFT 1.411733/1.411783, LoRA-r8 1.408463/1.408311, and LoRA-r64 1.409698/1.391648 for seeds 11/23 respectively.

## Validation trajectories and checkpoint selection

| Condition | Seed | Best step | Best epoch | Completed epochs | Best validation NLL | Recorded status |
|---|---:|---:|---:|---:|---:|---|
| FT | 11 | 1,208 | 3.5 | 5.5 | 1.053599 | mixed |
| FT | 23 | 1,553 | 4.5 | 6.5 | 1.054544 | deteriorating |
| DTFT | 11 | 3,450 | 10.0 | 10.0 | 1.059872 | improving |
| DTFT | 23 | 3,450 | 10.0 | 10.0 | 1.059973 | improving |
| LoRA-r8 | 11 | 173 | 0.5 | 2.5 | 1.062034 | deteriorating |
| LoRA-r8 | 23 | 173 | 0.5 | 2.5 | 1.063111 | deteriorating |
| LoRA-r64 | 11 | 173 | 0.5 | 2.5 | 1.061323 | deteriorating |
| LoRA-r64 | 23 | 345 | 1.0 | 3.0 | 1.062169 | deteriorating |

Neither LoRA rank was classified as plateaued. Both DTFT runs remained improving at the ten-epoch cap. Consequently, `lora_gap_interpretable` is false for both ranks; the measured checkpoint outcomes and convergence classifications are reported separately.

## Preregistered gap decomposition

The implementation converted NLL to negative NLL before applying the higher-is-better definitions in `PLAN.md`.

| LoRA condition | Outcome | `G_total` | `G_coverage` | `G_LoRA` | Coverage fraction |
|---|---|---:|---:|---:|---:|
| LoRA-r8 | Held-out likelihood | 0.005269 | 0.008639 | -0.003371 | 1.639804 |
| LoRA-r8 | Stability ranking | -0.002911 | -0.002250 | -0.000661 | 0.772956 |
| LoRA-r64 | Held-out likelihood | -0.002446 | 0.008639 | -0.011085 | -3.532747 |
| LoRA-r64 | Stability ranking | -0.004664 | -0.002250 | -0.002414 | 0.482376 |

`G_LoRA` was negative for both registered outcomes and both LoRA ranks: the selected LoRA checkpoints did not exhibit a DTFT advantage on either outcome. Therefore the Stage-1 rule to stop LoRA-specific mechanism diagnosis applies, the Stage-2 target-coverage stop statement is not emitted, and Stage 3/H1–H4 are not triggered by this compact pilot.

## Runtime and optimization measurements

The end-to-end ledger covered 16,930.45 seconds (4 h 42 min 10 s), including preprocessing/preflight, the complete HPO screen, base evaluation, eight evidence runs, evaluations, and summarization.

| Condition | Seed 11 training | Seed 23 training |
|---|---:|---:|
| FT | 1,509.47 s | 1,757.87 s |
| DTFT | 2,298.98 s | 2,272.98 s |
| LoRA-r8 | 797.40 s | 801.40 s |
| LoRA-r64 | 811.61 s | 958.21 s |

Accepted training-path changes retained the parity tests and included manifest path caching, pinned persistent workers, asynchronous CUDA prefetch, fused AdamW, removal of a target-token GPU synchronization, and a selectively enabled fixed-padding decoder path. On matched fixed-padding A6000 measurements, that decoder path changed FT from 0.6495 to 0.6331 s/step (2.5%), DTFT from 0.5599 to 0.5317 s/step (5.0%), and LoRA-r64 from 0.8721 to 0.8366 s/step (4.1%). It was disabled for LoRA-r8 after a measured 3.6% regression. CUDA graphs, compiled GVP execution, frozen-bf16 precasting, persistent autocast, and shared-backbone evaluation were not accepted because they failed, regressed, or did not retain required numerical behavior.

## Verification and retained artifacts

The complete local and A6000 protein test suites each passed 28 tests, with one NumPy empty-slice warning. The corrected full/compact manifest hashes matched across machines. The transfer archive matched SHA-256 `77d6db0365109d278bdb582c09add075bef47b33b765a8320795bc44962997b0` before extraction. The retained artifact set contains all result, history, evaluation, preflight, HPO-selection, ledger, and log files; the exact calibration/test source; corrected manifests and provenance; and all eight `best.pt` checkpoints (every condition and evidence seed). The redundant compressed transfer archive and redundant epoch/terminal checkpoints were not retained.

## Limitations

- This is a compact 11,016-example pilot, not the definitive full-data MegaScale experiment.
- Each trained condition has two evidence seeds; sample standard deviations have one degree of freedom, and no formal multi-seed significance test is reported.
- Base was evaluated once.
- Both selected LoRA trajectories deteriorated after early validation optima, and neither was plateaued.
- Both DTFT runs were still improving at the ten-epoch safety cap.
- The compact workflow did not run expensive generation, Rosetta, ESMFold, or later H1–H4 experiments.
- Structural partitioning used the recorded deterministic TM-align fallback rather than pinned FoldSeek 8.ef4e960 because the execution hosts lacked an authorized FoldSeek route.
- GPU scatter operations retain the underlying implementation's intrinsic atomic nondeterminism.
