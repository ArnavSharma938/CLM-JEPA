# Propedia26 Gate 1: Vanilla LoRA versus DTFT

**Final report — 2026-09-28**  
**Result:** At both ranks, the preregistered overall peptide-NLL endpoint shows no materially meaningful LoRA deficit, while the receptor-conditioning endpoint shows a replicated meaningful deficit. The evidence supports a specific loss of receptor-conditioned behavior, not a generic failure to adapt. It does not identify the cause of that loss.

## Executive conclusion

Rank-8 and rank-64 LoRA both retained overall true-pair peptide NLL close to the DTFT reference: for each seed, the 95% cluster-bootstrap interval for `G_LoRA = NLL_LoRA(true) - NLL_DTFT(true)` lay entirely below the preregistered task-deficit threshold, `delta_task = 0.0197383`. The intervals include zero; the result is **no materially meaningful deficit at the specified margin**, not proof that the models are identical.

The receptor-conditioning comparison differed. For both ranks and both seeds, the 95% interval for `D_cond = C_DTFT - C_LoRA` lay above the fixed `delta_cond = 0.00650711` threshold. Under the preregistered replication rule, each rank therefore has a **replicated meaningful loss of acquired receptor dependence**. LoRA's estimated `C_LoRA` remained positive in all four runs: it retained some receptor sensitivity, but less than DTFT. The r8/seed-23 lower confidence bound clears `delta_cond` by only about `0.000119`, so that particular result is threshold-close even though it satisfies the stated rule.

The next justified branch is a later mechanistic diagnosis focused on why both ranks lose receptor-conditioned signal (subspace, structure, or optimization diagnostics). Gate 1 does not establish any H1–H4 mechanism; none was tested here.

| Rank | Overall task-NLL endpoint | Receptor-conditioning endpoint | Replication decision |
|---|---|---|---|
| r8 | No materially meaningful deficit | Meaningful deficit | Both seeds agree on each endpoint |
| r64 | No materially meaningful deficit | Meaningful deficit | Both seeds agree on each endpoint |

## Scope and locked experimental design

This report covers only vanilla LoRA ranks 8 and 64 against the locked Gate-0 DTFT reference. There was no full fine-tuning run, third seed, new baseline suite, or H1–H4 experiment. Both ranks used the same corrected test panel and fixed decoys.

The model was `facebook/esm2_t33_650M_UR50D`, revision `08e4846e537177426273712802403f7ba8261b6c`. For each example the model received `[CLS] receptor [MASK]... [MASK] [EOS]`, with the full peptide span masked simultaneously and cross-entropy applied only to the peptide positions.

LoRA targeted the same 198 matrices as DTFT: Q/K/V, attention output, FFN input, and FFN output in all 33 transformer layers. Base weights were frozen; only `lora_A` and `lora_B` were trainable. With `alpha = rank`, `alpha/r = 1`. The tested parameter counts were 6,082,560 at r8 and 48,660,480 at r64. No embeddings, LayerNorms, or MLM-head parameters were added to the target set. Focused synthetic tests verified the trainable set/counts, zero-update output parity with Base at initialization, rank bounds on merged updates, and unchanged full-span mask/loss and evaluation manifests. The local focused test suite recorded 29 passing tests.

### Leakage controls and evaluation panel

Gate 0 prepared the data from the official Propedia26 v17 CSV (SHA-256 `22ba847c52ca4f6e6a4aa111c45032d25cfbee376a6218237e19ca2516437d8d`). It retained 23,641 unique valid pairs after sequence, assignment, length, and exact-pair filters. DIAMOND 2.2.8 all-vs-all receptor search and connected components at at least 30% identity and 80% coverage on both sequences produced 2,979 receptor clusters. Complete clusters were kept within one split. To prevent exact peptides from crossing between training and held-out data, 2,342 training candidates with held-out peptide collisions were excluded before freezing the final manifests. The frozen manifests contain 16,573 training pairs / 1,757 clusters, 2,363 validation pairs / 369 clusters, and an original 2,363-pair test set / 527 clusters. The remaining 326 clusters were not assigned to these frozen splits; Gate 1 did not use the optional unused-cluster extension. The split seed was `20260926`.

An exact peptide-sequence audit found 100 sequences shared by validation and test, affecting 150 test pairs. Those 150 pairs were excluded from the **test evaluation panel only**; training, validation, and the frozen source manifests were not changed. The corrected fixed panel contains **2,213 pairs / 481 receptor clusters** and has no exact peptide overlap with validation. The Gate-0 report records no train–test exact-peptide overlap and a nearest qualifying train–test receptor match at 29.6% identity (96.6% query and 97.9% subject coverage).

Every retained pair used the same four approximately length-matched decoy receptors for all evaluated models: 8,852 decoy scores in total. The machine-readable reference audit verifies matching panel/decoy membership for Base, DTFT, SVD-r8, and SVD-r64. No additional clusters or expanded precision panel were used in Gate 1.

Key frozen hashes:

| Asset | SHA-256 |
|---|---|
| Training manifest | `5f2fe8e783470b1e90fe2a1b528ca18da92d6800ef2539f13d34e63c57cc2859` |
| Validation manifest | `dfd3fad9bc84001ba1967cfec031f4283515dcef715ab8b683d7b9c018c8faa5` |
| HPO training subset (canonical LF) | `e9592bcf4790c82bd4b6fc4efedfc8dd03636fd748b39e44c31083d06379fca4` |
| Corrected test panel | `c1100f378585d2c5f82f722c06c6f796e37db8d48fd922fe0ef3ef9af5103de1` |
| Fixed-decoy manifest | `e20cc91e2933977d8a043b7f79a4df04ef4a47122fe9912acb3006ada86cef81` |

### Locked Gate-0 reference and material thresholds

The primary estimand is the **equal-receptor-cluster-weighted mean**: first average pair-level quantities within each receptor cluster, then weight clusters equally. The receptor cluster is the independent statistical unit. The corrected Gate-0 reference values, recomputed with this same estimand, are:

- `A_DTFT = 0.1973830387773504` (corrected Gate-0 95% cluster-bootstrap CI `[0.166372, 0.231341]`).
- `C_DTFT = 0.06507106981284858` (corrected Gate-0 95% cluster-bootstrap CI `[0.030862, 0.101965]`).
- `delta_task = 0.10 * A_DTFT = 0.01973830387773504`.
- `delta_cond = 0.10 * C_DTFT = 0.0065071069812848575`.

These Gate-0 values and thresholds were treated as fixed for Gate 1, as preregistered. Full-precision values, not rounded display values, were used in the analysis.

## Learning-rate selection

Each rank was tuned independently with HPO seed 7 using the deterministic cluster-balanced 4,096-pair subset and the frozen 2,363-pair validation set. The initial peak-LR grid was `[1e-5, 3e-5, 1e-4, 3e-4]`. Candidates ran for four epochs with validation every 0.5 epoch, then extended only when the specified material-improvement rule applied; no candidate exceeded six epochs. If an initial-grid edge won, only the one allowed outer candidate (`1e-3` here) was tested. Candidates were stable, with no numerical failures.

The primary score below is the candidate's lowest mean over two consecutive validation checks. `Minimum → terminal` shows the best single observation and final observation after the candidate's completed screen/extension. Values are validation peptide NLL.

| Rank | Peak LR | Epochs | Best two-check mean | Minimum → terminal | Extension to 6? |
|---|---:|---:|---:|---:|---|
| r8 | `1e-5` | 6 | 3.014760 | 3.014418 → 3.014418 | Yes |
| r8 | `3e-5` | 6 | 2.980742 | 2.979116 → 2.979116 | Yes |
| r8 | `1e-4` | 6 | 2.915102 | 2.913698 → 2.913698 | Yes |
| r8 | `3e-4` | 6 | 2.884975 | 2.883649 → 2.883649 | Yes |
| **r8** | **`1e-3`** (single outer probe) | **6** | **2.867238** | **2.866718 → 2.921633** | **Yes** |
| r64 | `1e-5` | 6 | 2.972360 | 2.970940 → 2.970940 | Yes |
| r64 | `3e-5` | 6 | 2.919466 | 2.918200 → 2.918200 | Yes |
| r64 | `1e-4` | 6 | 2.886256 | 2.885233 → 2.885233 | Yes |
| **r64** | **`3e-4`** | **6** | **2.866859** | **2.866462 → 2.895368** | **Yes** |
| r64 | `1e-3` (single outer probe) | 4 | 2.870683 | 2.863578 → 2.945580 | No |

The r8 outer probe beat adjacent `3e-4` by 0.017737 NLL and was selected; the protocol capped expansion at that one outer probe, so no further LR search was performed. For r64, the `3e-4` score beat the `1e-3` boundary probe by 0.003824, greater than the 0.002 practical-tie band, so `3e-4` was selected. The r64 boundary probe's single best point was lower than its two-check score; selection followed the prespecified two-check score, not that transient point.

The JSON HPO reports contain the complete validation trajectories, optimizer-step records, gradient-norm maxima, and candidate configs ([r8](../../runs/protein/propedia26_gate1/lr_selection/rank8/hpo_report.json), [r64](../../runs/protein/propedia26_gate1/lr_selection/rank64/hpo_report.json)). One reporting caveat: after an extension, the field named `still_improving_at_four_epoch_screen` is recalculated from the final six-epoch trajectory rather than preserved as an epoch-4 snapshot. It should not be read as the historical epoch-4 decision; the full trajectory and `extended_to_six_epochs` flag are retained. This metadata caveat does not change the saved trajectories or selected scores.

## Evidence training and compute

Both ranks used the same training/validation manifests, full-span objective, 10-epoch schedule horizon, and the independently selected LR. Optimizer and execution settings were AdamW (`betas=(0.9,0.999)`, `eps=1e-8`, `weight_decay=0.01`, fused), global gradient clipping at 1.0, BF16 autocast, TF32 matmuls, uncompiled PyTorch SDPA, no gradient checkpointing, 8,192 padded input-token budget, 8,192 supervised-peptide-token accumulation budget, and two workers. The schedule warmed up for 5%, then cosine-decayed to 10% of peak by epoch 10. Validation ran every half epoch.

All Gate-1 GPU work used the single allowed Thunder instance: one NVIDIA L40 (48 GB), 6 vCPU, Base template, Prototyping instance, 100 GB storage. No second instance was launched.

The early-stopping rule required at least five epochs and then stopped after four consecutive half-epoch checks without a `>=0.001` NLL improvement. All four runs stopped by that rule at about 6.5 epochs; none reached the 12-, 16-, or 20-epoch ceilings. All were numerically stable. The best-validation checkpoint, not the terminal weights, was evaluated. Late validation deterioration occurred in each run; this documents checkpoint selection but does not by itself identify a cause such as overfitting.

| Rank / seed | Selected LR | Best epoch | Best validation NLL | Terminal validation NLL | Optimizer steps | Wall time | PyTorch peak allocated VRAM* |
|---|---:|---:|---:|---:|---:|---:|---:|
| r8 / 11 | `1e-3` | 1.5 | 2.871991 | 3.171473 | 253 | 44.18 min | 22.22 GiB |
| r8 / 23 | `1e-3` | 1.5 | 2.874352 | 3.288804 | 254 | 44.17 min | 22.22 GiB |
| r64 / 11 | `3e-4` | 1.5 | 2.878089 | 3.307630 | 253 | 44.63 min | 23.14 GiB |
| r64 / 23 | `3e-4` | 1.5 | 2.877974 | 3.249707 | 254 | 44.62 min | 23.15 GiB |

\*VRAM is the PyTorch allocator's recorded peak allocation, not a whole-device/NVML peak.

Seed 23 was triggered once per rank because seed 11's conditional-loss ratio `D_cond / C_DTFT` exceeded 0.08. Neither rank triggered on the task-gap criteria. No third seed was run, and clusters from different seeds were not pooled as independent observations.

## Corrected-panel results

For each seed, test metrics below use equal receptor-cluster weighting and a paired bootstrap over 481 clusters with 20,000 draws. `G_LoRA` is the LoRA-minus-DTFT true-pair NLL difference; `F_gap = G_LoRA / A_DTFT`. The primary material threshold is `delta_task = 0.0197383`.

| Rank / seed | LoRA true-pair NLL | `G_LoRA` (95% cluster CI) | `F_gap` (95% cluster CI) | Task endpoint |
|---|---:|---:|---:|---|
| r8 / 11 | 2.874031 | 0.001207 `[-0.007021, 0.009469]` | 0.006116 `[-0.035572, 0.047971]` | No material deficit |
| r8 / 23 | 2.870898 | -0.001926 `[-0.009773, 0.005689]` | -0.009758 `[-0.049511, 0.028825]` | No material deficit |
| r64 / 11 | 2.873371 | 0.000547 `[-0.007496, 0.008642]` | 0.002772 `[-0.037978, 0.043780]` | No material deficit |
| r64 / 23 | 2.873702 | 0.000878 `[-0.006681, 0.008344]` | 0.004450 `[-0.033849, 0.042273]` | No material deficit |

All four task-gap confidence intervals lie entirely below `delta_task`; all include zero. None of the point estimates falls in the originally tracked `0.08 <= F_gap < 0.10` borderline region. The preregistered replication rule classifies the task endpoint as **replicated no materially meaningful deficit** at both ranks. This is a threshold-based result, not an equivalence proof.

For the predeclared conditioning endpoint, `S_model = NLL(decoy) - NLL(true)`, `C_LoRA = S_LoRA - S_Base`, `D_cond = C_DTFT - C_LoRA`, and `Q_LoRA = C_LoRA / C_DTFT`. The material threshold is `delta_cond = 0.00650711`.

| Rank / seed | `C_LoRA` (95% cluster CI) | `D_cond` (95% cluster CI) | `D_cond / C_DTFT` | `Q_LoRA` (95% cluster CI) | Conditioning endpoint |
|---|---:|---:|---:|---:|---|
| r8 / 11 | 0.044884 `[0.010643, 0.081097]` | 0.020187 `[0.011721, 0.028681]` | 0.3102 | 0.6898 `[0.1636, 1.2463]` | Meaningful loss |
| r8 / 23 | 0.049613 `[0.015292, 0.085292]` | 0.015458 `[0.006626, 0.024301]` | 0.2376 | 0.7624 `[0.2350, 1.3108]` | Meaningful loss |
| r64 / 11 | 0.035656 `[0.003320, 0.070745]` | 0.029415 `[0.020337, 0.038560]` | 0.4520 | 0.5480 `[0.0510, 1.0872]` | Meaningful loss |
| r64 / 23 | 0.039665 `[0.006301, 0.074253]` | 0.025406 `[0.016665, 0.034212]` | 0.3904 | 0.6096 `[0.0968, 1.1411]` | Meaningful loss |

Each `D_cond` interval is wholly above `delta_cond`, so each rank meets the rule for a **replicated meaningful conditioning deficit**. The r8/seed-23 lower bound is only slightly above the threshold, as noted above. The positive `C_LoRA` intervals and sub-1 point estimates of `Q_LoRA` are consistent with LoRA retaining some receptor signal while retaining less of DTFT's acquired signal. Bootstrap intervals for the ratio `Q_LoRA` extend above 1 in some cases; they are reported without truncation and are not used for the preregistered classification.

## Same-rank SVD comparison (descriptive only)

On the same corrected panel, the post-hoc dense-update SVD results were `R8=0.911300`, `Q8=0.480876`, `R64=0.965485`, and `Q64=0.754678`. These are descriptive reconstructions of the DTFT update, not trained LoRA models and not causal controls. The trained LoRA Q point estimates were 0.6898/0.7624 at r8 and 0.5480/0.6096 at r64. Thus the relative LoRA–SVD ordering differs by rank; it does not support treating truncation as a proxy for LoRA training or attributing the observed conditioning gap to rank alone.

## Limitations and interpretation boundaries

- The inference is for this fixed Propedia26 panel, ESM-2 revision, objective, selected LRs, and training procedure; it is not a universal claim about LoRA.
- The independent uncertainty unit is the receptor cluster. Seeds are reported separately; there is no pooled seed-level “extra sample” CI.
- The validation-selected checkpoints occur early (epoch 1.5), with later validation deterioration. The reported metrics use those selected checkpoints only. This pattern does not identify whether the deterioration reflects overfitting, optimization dynamics, or another cause.
- The two ranks both show conditional loss, but no direct rank-vs-rank hypothesis test was run. Their point estimates do not establish that one rank is better or worse than the other.
- No precision extension was used because the preregistered per-seed intervals cleared the material thresholds; no unused-cluster panel was needed.
- No H1–H4 mechanism was tested or established.

## Reproducibility and retained artifacts

The final machine-readable result is [`gate1_summary.json`](../../runs/protein/propedia26_gate1/gate1_summary.json); the paired analysis is [`primary_panel_analysis.json`](../../runs/protein/propedia26_gate1/primary_panel_analysis.json). Per-rank HPO reports contain full trajectories. Per-seed `convergence_summary.json`, `history.json`, `execution.json`, `config.json`, selected `best.pt`, resumable `latest.pt`, and optimizer-step logs remain under `runs/protein/propedia26_gate1/evidence/`. Evaluation score vectors and summaries remain under `runs/protein/propedia26_gate1/evaluations/`.

The exact Gate-1 module copied from Thunder is preserved at [`provenance/gate1_lora_executed.py`](../../runs/protein/propedia26_gate1/provenance/gate1_lora_executed.py), SHA-256 `f5ee0ca528be55ec0343cc69c7b0befeff220064cbf53f7d197c469c3c60df6e`. The current local [`gate1_lora.py`](../../src/esm2_peptide_generation/gate1_lora.py) has SHA-256 `e4e69d719bae6a4966b05f12673338391ca4aa0ae3358c84501d6bb57be4ce6a`; inspection found the difference confined to `write_final_summary()` serialization of per-seed training metadata. The HPO, training, evaluation, and analysis paths did not differ. `training.py` matched between local and Thunder at SHA-256 `1524cc37f94768b7db66efec7023fb8fe34c1210db7ea09f1fe91c326696d39a`.

After transfer, all 138 files in the original Thunder checksum manifest were hash-verified locally. Ten stale remote PID marker files were then moved out of the run tree; a refreshed local checksum manifest covers the retained 129 run files (excluding the manifest itself). Checkpoints, HPO trajectories, score files, logs, manifests, and configs were retained. The one L40 Thunder instance was terminated after verification; no instance remains.

Related records: [corrected Gate-0 report](PROPEDIA26_GATE0_L40.md), [r8 HPO](../../runs/protein/propedia26_gate1/lr_selection/rank8/hpo_report.json), [r64 HPO](../../runs/protein/propedia26_gate1/lr_selection/rank64/hpo_report.json), [transfer checksum manifest](../../runs/protein/propedia26_gate1/transfer_SHA256SUMS.txt), and [Gate-1 tests](../../tests/test_propedia_gate1.py).
