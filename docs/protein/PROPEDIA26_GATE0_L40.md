# Propedia26 ESM-2 peptide-generation Gate 0 — L40

## Final result: PASS

The leakage-controlled task is valid; the selected DTFT reference is converged, has
reliable adaptation gain and acquired receptor dependence, and its rank-64 SVD does
not preserve 90% of both. Stage-6 power is only an informational planning proxy based
on Base-to-DTFT variability; the actual DTFT-versus-LoRA paired variance is unknown
until Gate 1 and the proxy does not stop Gate 0. Proceed to Gate 1 using the preserved
`1e-4` DTFT checkpoint.

## Stage 1 — Frozen task

- Source: official Propedia26 v17 CSV, SHA-256
  `22ba847c52ca4f6e6a4aa111c45032d25cfbee376a6218237e19ca2516437d8d`.
- After canonical-sequence, unambiguous-pair and ESM-2 length filters, then exact
  pair deduplication: 23,641 unique pairs, 16,178 unique receptors, 11,023 unique
  peptides; 30,069 noncanonical rows and 541 overlength rows were rejected.
- DIAMOND 2.2.8 all-vs-all receptor search; connected components at >=30% identity
  and >=80% coverage on both sequences produced 2,979 receptor clusters. Complete
  clusters stayed within a split; exact peptides do not cross train/test.
- Frozen counts: train 16,573 pairs / 1,757 clusters; validation 2,363 / 369;
  test 2,363 / 527. Train-test exact-peptide overlap is zero. Nearest qualifying
  train-test receptor match: 29.6% identity, 96.6% query and 97.9% subject coverage.
- Frozen manifest hashes and source provenance:
  [`frozen_split_summary.json`](../../data/protein/propedia26/manifests/frozen_split_summary.json),
  [`frozen_evaluation_manifests.json`](../../data/protein/propedia26/manifests/frozen_evaluation_manifests.json).

## Stage 2 — Objective and implementation

Pinned `facebook/esm2_t33_650M_UR50D` revision
`08e4846e537177426273712802403f7ba8261b6c`. Inputs are `[CLS] receptor [MASK]... [EOS]`,
with the entire peptide span masked simultaneously and loss only at peptide positions.
The 198 transformer matrices (Q/K/V, attention output, FFN input/output across 33
layers; 648,806,400 trainable weights) are the only DTFT targets. Focused masking,
loss-equivalence, batching, split, attention, cluster-weighting, and planning-proxy
tests passed 10/10 in the pinned remote environment.

## Stage 3 — Base, L40 profile, HPO and one evidence run

Base full-span peptide NLL: validation 3.017992; test true receptor 3.045782; fixed
test decoys 3.081477.

Exactly one Thunder instance was used: 1x NVIDIA L40, 6 vCPU, Base template,
Prototyping, 100 GB. The real Propedia DTFT loop selected uncompiled PyTorch SDPA:

- Mean/median total input length 276.50/245 tokens; supervised peptide length
  19.83/14 tokens per example.
- Padded microbatch budget 8,192; optimizer step budget 8,192 supervised peptide
  tokens; 2 workers.
- Measured throughput 8,943 input tokens/s; peak profile VRAM 29.38 GiB; estimated
  8.55 minutes per epoch. Runtime estimate: 43 minutes for the 5-epoch minimum,
  56–86 minutes for 6.5–10 epochs, 103 minutes at the initial 12-epoch ceiling,
  and 171 minutes at the hard 20-epoch ceiling.
- The 12,288-token setting approached 40 GiB and was rejected. `torch.compile`
  default was substantially slower; `max-autotune-no-cudagraphs` was 3.6% slower.

HPO used seed 7, a deterministic cluster-round-robin 4,096-pair subset, the frozen
validation set, AdamW and the same batching/precision/backend as the evidence run.
All candidates completed 4 epochs with eight post-start observations and were
numerically stable; neither finalist remained materially improving, so no extension
was required. The edge winner triggered exactly one `3e-4` expansion. The complete
machine-readable trajectories, gradients and selection record are in
[`hpo_report.json`](../../runs/protein/propedia26_gate0/hpo/hpo_report.json).

| Peak LR | Eight validation NLLs, at 0.5-epoch intervals | Best two-check mean | Min → terminal | Drift from min |
|---|---|---:|---:|---:|
| `1e-6` | 2.96266, 2.95937, 2.95767, 2.95619, 2.95546, 2.95521, 2.95505, 2.95489 | 2.954968 | 2.954889 → 2.954889 | 0.000000 |
| `3e-6` | 2.96151, 2.95548, 2.94829, 2.94494, 2.94245, 2.94021, 2.93870, 2.93805 | 2.938378 | 2.938052 → 2.938052 | 0.000000 |
| `1e-5` | 2.94909, 2.92973, 2.91766, 2.91033, 2.90710, 2.90618, 2.90566, 2.90537 | 2.905514 | 2.905371 → 2.905371 | 0.000000 |
| `3e-5` | 2.92042, 2.90150, 2.89504, 2.89308, 2.89034, 2.88955, 2.88868, 2.88844 | 2.888562 | 2.888444 → 2.888444 | 0.000000 |
| `1e-4` | 2.89913, 2.88480, 2.86813, 2.87172, 2.88399, 2.90073, 2.91675, 2.93332 | 2.869924 | 2.868128 → 2.933316 | 0.065189 |
| `3e-4` (boundary expansion) | 2.89983, 2.87218, 2.87352, 2.88530, 2.94891, 2.95660, 3.12975, 3.09029 | 2.872854 | 2.872184 → 3.090285 | 0.218101 |

Selected peak LR: `1e-4`. It beat adjacent `3e-5` by more than 0.002, so the one-step
boundary probe was run; `1e-4` then beat `3e-4` by 0.002930 (not a practical tie).
The winner did deteriorate late, but no lower-LR finalist was within the 0.002 tie
band, so the specified lower-LR tie fallback did not apply.

Exactly one full-split DTFT evidence seed (seed 11) ran at `1e-4`. It completed
6.499 epochs / 253 optimizer steps and stopped by the specified early-stopping rule
after the 5-epoch minimum; total wall time was 43.68 minutes and peak VRAM was
29.43 GiB. The selected best checkpoint is epoch 1.0, validation NLL 2.874981,
SHA-256 `031313c623a689fc10a77c6819a35c3659ec43d7ab30303906a2d3911e984088`.
Terminal validation NLL was 4.096650; the deteriorated terminal checkpoint was not
used. The run is classified converged by the prespecified early-stopping rule.

Last eight observations (the complete trajectory is in
[`convergence_report.json`](../../runs/protein/propedia26_gate0/evidence_seed11/convergence_report.json)):

| Epoch | Validation NLL | Current LR |
|---:|---:|---:|
| 3.000 | 3.067564 | `8.5860e-5` |
| 3.500 | 3.213030 | `7.9562e-5` |
| 4.000 | 3.235539 | `7.3536e-5` |
| 4.500 | 3.478052 | `6.6339e-5` |
| 5.000 | 3.533827 | `5.9197e-5` |
| 5.500 | 3.781612 | `5.1565e-5` |
| 6.000 | 3.824003 | `4.4402e-5` |
| 6.499 | 4.096650 | `3.7515e-5` |

### One requested full-split LR sensitivity run

Exactly one additional full-split seed-11 DTFT run was performed at peak LR `3e-5`.
No HPO or other training run was repeated. It used the same frozen train/validation
manifests (hashes match the frozen LF manifests), target matrices, AdamW settings,
8,192 padded-input/supervised-token budgets, schedule, SDPA/BF16 execution,
validation cadence, and stopping rule. It was numerically stable and early-stopped
at 6.499 epochs / 253 optimizer steps. Best validation NLL was 2.892907 at epoch 1.0;
terminal NLL was 3.395283. Wall time was 43.04 minutes and peak VRAM was 29.43 GiB.
The best 1e-4 checkpoint remains lower at 2.874981 (by 0.017925 NLL), so the lower-LR
sensitivity did not improve the dense reference. Full trajectory and provenance are
in [`sensitivity_lr3e-5/convergence_report.json`](../../runs/protein/propedia26_gate0/sensitivity_lr3e-5/convergence_report.json).

The final eight 3e-5 validation observations were:

| Epoch | Validation NLL | Current LR |
|---:|---:|---:|
| 3.000 | 2.948875 | `2.5758e-5` |
| 3.500 | 3.007844 | `2.3869e-5` |
| 4.000 | 3.023741 | `2.2061e-5` |
| 4.500 | 3.102738 | `1.9902e-5` |
| 5.000 | 3.153308 | `1.7759e-5` |
| 5.500 | 3.250603 | `1.5469e-5` |
| 6.000 | 3.284060 | `1.3321e-5` |
| 6.499 | 3.395283 | `1.1254e-5` |

The sensitivity checkpoint is preserved for audit, but the selected DTFT reference
remains [`evidence_seed11/best.pt`](../../runs/protein/propedia26_gate0/evidence_seed11/best.pt).

## Stage 4 — Adaptation and acquired receptor dependence

On the frozen 2,363-pair / 527-cluster test set, using the same four approximately
length-matched decoy receptors for every model, token-pooled Base→DTFT true-pair NLL
improved from 3.045782 to 2.886046. The primary estimand is the equal-receptor-
cluster-weighted mean, with receptor clusters as independent units. Cluster-bootstrap
results:

- `A_DTFT=0.192811`, 95% CI [0.163651, 0.224689].
- `C_DTFT=0.059566`, 95% CI [0.027284, 0.094036]; `C/A=0.309`.
- Stage 4 passes: reliable adaptation and positive acquired correct-receptor
  dependence, exceeding the 10%-of-adaptation conditional-signal threshold.
- Recomputed cluster-level score and bootstrap details:
  [`stage4_selected_1e-4.json`](../../runs/protein/propedia26_gate0/sensitivity_lr3e-5/analysis/stage4_selected_1e-4.json).

## Stage 5 — Rank-64 reconstruction

Independent best truncated SVDs were applied to each of the 198 actual DTFT updates;
neither reconstruction was trained. Using the same equal-cluster estimand in each
numerator and denominator, rank-64 retained `R64=0.9677` of the adaptation gain and
`Q64=0.7495` of the acquired conditional signal. Rank-8 retained `R8=0.9128`,
`Q8=0.4536`. Rank-64 therefore does not preserve >=90% of both signals;
the task passes Stage 5 and is not `LOW-RANK-EASY`. This only shows that this dense
update is not functionally preserved by its own rank-64 truncation; it does not
establish that vanilla LoRA would fail. Details:
[`rank_screen_report_equal_cluster.json`](../../runs/protein/propedia26_gate0/sensitivity_lr3e-5/analysis/rank_screen_report_equal_cluster.json).

## Stage 6 — Informational detectability planning proxy

On the original 527-cluster panel, the Base-to-DTFT variability proxy gave MDE
0.043859 and proxy power 0.1897 at the primary material effect `0.10*A=0.019281`.
All 326 receptor clusters unused by the
frozen splits were then added: 889 pairs, zero train-test peptide overlap. On the
expanded 3,252-pair / 853-cluster set:

- `A_DTFT=0.180367`, 95% CI [0.157984, 0.203993].
- `C_DTFT=0.053267`, 95% CI [0.030108, 0.077853]; `C/A=0.295` (Stage 4 still passes).
- The Gate-1 material threshold remains anchored to the Stage-4 primary estimand:
  `delta_material=0.10*A_DTFT=0.019281` (not 10% of the expanded-panel estimate).
- Proxy MDE at alpha 0.05 / 80% power is `0.032926`; proxy power at the material
  threshold is `0.3535` across 853 clusters.
- This proxy substitutes Base→DTFT adaptation-gain variability for a future paired
  DTFT-versus-LoRA variance. The actual paired variance is unknown until LoRA exists;
  the proxy is not validated as conservative. It flags a precision risk for planning
  and does not make Gate 0 inadequate or block Gate 1.

No further unused valid receptor clusters remain. Gate 1 must estimate the actual
paired cluster-level difference `G_cluster = mean-within-cluster(NLL_LoRA -
NLL_DTFT)` on the same held-out panel. Compare its cluster-bootstrap confidence
interval against `delta_material=0.10*A_DTFT`: entirely above means a meaningful
LoRA deficit; entirely below means no materially meaningful deficit; substantial
overlap means `UNRESOLVED` for insufficient precision. Do not substitute the Gate-0
proxy for that test.

Expanded-set report and pair scores:
[`stage6_proxy_selected_1e-4.json`](../../runs/protein/propedia26_gate0/sensitivity_lr3e-5/analysis/stage6_proxy_selected_1e-4.json).

## Artifact transfer

The prior Gate-0 run tree remains locally preserved under `runs/protein/propedia26_gate0/`;
its 100 files were SHA-256 verified. This sensitivity run's 2,595,291,238-byte
`best.pt`, convergence/history/configuration, optimizer-step log, run ledger, and
updated Stage-4/5/6 analysis JSONs are also local under
`runs/protein/propedia26_gate0/sensitivity_lr3e-5/`. The best checkpoint SHA-256 is
`ebba120ed0560de5f204e3298c8e0da042acc306a31936853b34c136f19efb08`; copied artifacts
and analysis reports matched their remote hashes. Since the run converged, only the
selected best model checkpoint—not a redundant resumable optimizer-state checkpoint—
was retained from this run. After verification, the sole Thunder instance (`keqktlpk`,
1x L40) was deleted; `tnr status` reported no remaining instances.
