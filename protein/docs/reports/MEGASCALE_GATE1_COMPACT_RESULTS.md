# MegaScale compact Gate 1 results report

## Reporting scope

This report records measurements from the fixed compact MegaScale Gate 1 surrogate. It reports observed results, comparisons, uncertainty, preregistered gate outputs, and limitations. It does not assign a mechanistic explanation or treat the compact surrogate as the definitive full-data Gate 1 experiment.

## Data and provenance

The benchmark contained 11,016 training variants from 36 domains and 36 structural clusters, 960 validation variants from 6 domains and 2 held-out clusters, and 1,440 test variants from 9 domains and 3 held-out clusters. No structural cluster occurred in more than one split. Every training domain contributed 306 stabilizing variants. The training set contained 8,835 single and 2,181 double mutants; validation contained 776 single and 184 double mutants; test contained 1,160 single and 280 double mutants.

The compact manifest SHA-256 was `944a0283100205b7e203d4b3cbb87d83d8c27ed3d5464a4f6b9a30ba437a3f2e`. The ordered sequence-ID record SHA-256 was `dad8df9d673659c76a43270c987b83b3b7e327d6a9f7233a68dbfeec5d96821b`. The eligible full manifest from which it was constructed had SHA-256 `7fed569afa6b6941ba6480994555840602e4960aa0e15614887120f13fd6409f`.

Final study eligibility was applied before structural clustering, split assignment, and held-out structural-distance calculation. Because native FoldSeek 8.ef4e960 was unavailable on the Windows host, the eligible-population structural partition used the recorded deterministic TM-align fallback. This differs from the planned FoldSeek implementation and is an experimental limitation.

## Experimental configuration

The four trained conditions were full fine-tuning (FT), dense-target fine-tuning (DTFT), LoRA rank 8, and LoRA rank 64. Evidence seeds were 11 and 23. FT and DTFT used learning rate `1e-7`; both LoRA conditions used `3e-5`. Training used the same objective, effective batch size 32, 0.1 Å coordinate noise, and paired ordering. Validation was evaluated every 0.5 epoch. Runs had a minimum of 2 epochs, a maximum of 10 epochs, and stopped after four validation checks without a new optimum. Checkpoints were selected only by validation NLL. The unadapted base model was evaluated once.

All eight training runs and all nine evaluations completed with exit code 0 on one NVIDIA A6000. End-to-end queue time, including benchmark reconstruction, preflight, base evaluation, training, evaluation, and summarization, was 7,638.18 seconds (2 h 7 min 18 s).

## Held-out measurements

Values for trained conditions are mean ± sample standard deviation across two independent training seeds. Base was evaluated once. NLL is lower-is-better; the remaining metrics are higher-is-better as registered by the evaluation implementation.

| Condition | Held-out NLL | Mean domain Spearman | Mean domain AUROC | Sequence recovery |
|---|---:|---:|---:|---:|
| Base | 1.454950 | -0.644060 | 0.225032 | 0.592254 |
| FT | 1.403049 ± 0.009813 | -0.651548 ± 0.000487 | 0.220484 ± 0.000613 | 0.587937 ± 0.005418 |
| DTFT | 1.437105 ± 0.000026 | -0.646738 ± 0.000270 | 0.223249 ± 0.000446 | 0.587571 ± 0.001190 |
| LoRA-r8 | 1.454946 ± 0.000031 | -0.644275 ± 0.000784 | 0.224183 ± 0.000345 | 0.591425 ± 0.001173 |
| LoRA-r64 | 1.454874 ± 0.000025 | -0.643719 ± 0.000278 | 0.224745 ± 0.000076 | 0.591425 ± 0.001173 |

Relative to base, mean held-out NLL changed by -0.051901 for FT, -0.017845 for DTFT, -0.000004 for LoRA-r8, and -0.000076 for LoRA-r64. FT minus DTFT on the higher-is-better negative-NLL scale was 0.034056.

## Validation trajectories and checkpoint selection

| Condition | Seed | Selected step | Completed epochs | Selected validation NLL | Early stop |
|---|---:|---:|---:|---:|---:|
| FT | 11 | 1,035 | 5.0 | 1.055411 | yes |
| FT | 23 | 1,725 | 7.0 | 1.057226 | yes |
| DTFT | 11 | 863 | 4.50 | 1.063468 | yes |
| DTFT | 23 | 863 | 4.50 | 1.063442 | yes |
| LoRA-r8 | 11 | 0 | 2.0 | 1.064827 | yes |
| LoRA-r8 | 23 | 0 | 2.0 | 1.064831 | yes |
| LoRA-r64 | 11 | 0 | 2.0 | 1.064835 | yes |
| LoRA-r64 | 23 | 0 | 2.0 | 1.064836 | yes |

For both LoRA ranks and both seeds, every post-update validation measurement was worse than step 0. At epoch 2, LoRA-r8 validation NLL was 1.157115 and 1.153775; LoRA-r64 validation NLL was 1.557304 and 1.574119. Consequently, validation selection restored the step-0 checkpoint for all four LoRA runs. None of the four LoRA trajectories was still improving at termination under the recorded three-delta diagnostic.

## Preregistered gap decomposition

The implementation converted NLL to negative NLL before applying the higher-is-better gap definitions.

| LoRA condition | Outcome | `G_total` | `G_coverage` | `G_LoRA` | Coverage fraction |
|---|---|---:|---:|---:|---:|
| LoRA-r8 | Held-out likelihood | 0.051897 | 0.034056 | 0.017841 | 0.656222 |
| LoRA-r8 | Stability ranking | -0.007272 | -0.004810 | -0.002462 | 0.661407 |
| LoRA-r64 | Held-out likelihood | 0.051825 | 0.034056 | 0.017769 | 0.657134 |
| LoRA-r64 | Stability ranking | -0.007829 | -0.004810 | -0.003019 | 0.614379 |

The preregistered stop rule requires positive `G_total`, positive `G_coverage`, and coverage fraction above 0.5 on both primary cheap outcomes. The stability-ranking gaps were negative for both LoRA comparisons. Therefore, the generated `stop_h1_h4` value was `false` for LoRA-r8 and LoRA-r64, and no registered stop statement was emitted.

## Runtime and memory measurements

Observed training durations were 1,287.73 and 1,693.60 seconds for FT; 914.94 and 968.50 seconds for DTFT; 613.25 and 592.14 seconds for LoRA-r8; and 613.29 and 600.12 seconds for LoRA-r64. The LoRA runs were shorter because all four stopped at the two-epoch minimum with step 0 selected.

Prequeue A6000 forward/backward/optimizer smoke measurements at batch size 32 were 0.610 s/step and 13.62 GiB peak for FT, 0.542 s/step and 3.62 GiB for DTFT, 0.824 s/step and 3.20 GiB for LoRA-r8, and 0.888 s/step and 3.31 GiB for LoRA-r64. Exact accepted training-path changes included manifest path caching, pinned persistent workers, asynchronous CUDA prefetch, fused AdamW, removal of a target-token GPU synchronization, and a fixed-padding decoder path enabled only where its measured throughput improved.

On matched fixed-padding benchmarks, the decoder fast path changed FT from 0.6495 to 0.6331 s/step (2.5%), DTFT from 0.5599 to 0.5317 s/step (5.0%), and LoRA-r64 from 0.8721 to 0.8366 s/step (4.1%). It was disabled for LoRA-r8 after a 3.6% regression. CUDA graphs, compiled GVP execution, frozen-BF16 precasting, persistent autocast, and shared-backbone evaluation were not accepted because they either failed, regressed, or did not retain the required numerical behavior.

## Verification and reproducibility artifacts

The local and A6000 test suites each completed with 23 passing tests and one NumPy empty-slice warning. The final summary SHA-256 was `9280886ba65caad59b31049dbb1f3d8fa56f7ea8bc4bab69e905127450893ba5`.

The transferred key-artifact archive contains the repository source and configuration, tests, exact data manifests and provenance, execution ledger, logs, profiles, all result/history/evaluation JSON, and the selected `best.pt` checkpoint for every condition and seed. Its SHA-256 is `e163f7892a1d1ace94a33b58ee8ca6c08a051d29561c8abcf1dd9588640ff230`.

## Experimental limitations

- The benchmark is a compact surrogate rather than the full eligible MegaScale training population.
- Each trained condition has two seeds; reported standard deviations therefore have one degree of freedom, and no confidence interval or formal multi-seed significance test is reported.
- Base was evaluated once and has no between-seed uncertainty estimate.
- Both LoRA learning rates produced validation deterioration from the initial checkpoint; the held-out LoRA measurements are consequently measurements of the selected step-0 checkpoints.
- The compact workflow did not run the full generative sampling and expensive diagnostic suite specified for a definitive full-data stage.
- Structural partitioning used the recorded TM-align fallback rather than pinned FoldSeek 8.ef4e960.
- The accepted optimizations were constrained to transformations that retained the repository's parity checks; GPU scatter operations retain the underlying implementation's intrinsic atomic nondeterminism.

