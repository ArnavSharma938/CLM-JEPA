# USPTO-MIT exact SIGReg batch-16 experiment (A6000)

## Result

**Batch-16 SIGReg did not prevent the cLM-JEPA geometric shortcut and did not improve generation over the cadence-matched native control.** At epoch 2, SIGReg happened to score 7/256 exact top-1 versus native's 5/256, but the paired difference was uncertain (`+0.78 pp`, 95% bootstrap CI `[-0.78, +2.34] pp`, McNemar `p=0.625`), all larger beam cutoffs were worse, and target-token CE was 1.80% worse. At epoch 4, SIGReg fell to 3/256 versus native's 6/256 (`-1.17 pp`, 95% CI `[-2.73, 0.00] pp`, McNemar `p=0.25`) and CE remained 1.30% worse.

SIGReg also failed its intended mechanism test. Relative to matched native, source/target variance was 12.3x/9.8x lower at epoch 2 and 16.9x/14.8x lower at epoch 4, while mean-direction energy approached one. The residual remained multidimensional and chemically pair-specific after common-component removal, so this is still not evidence of classical representation collapse. It is further variance contraction/common-direction concentration with useful information buried in the residual.

The requested experiment is complete. No coefficient tuning, new seed, alternative regularizer, or follow-up training was run.

## Protocol and execution fidelity

- Model/task: ChemFM-1B, USPTO-MIT forward prediction, fixed 1,280-row pilot training manifest, seed 533.
- Matched conditions: native NTP and symmetric k=0 EOS cosine-JEPA plus exact batch-16 SIGReg.
- Objective on active auxiliary updates: `L_NTP + 2 * [L_cos + (0.01/0.99) * L_SIGReg]`; 50% auxiliary dropout; symmetric gradients through source and target.
- Both conditions used physical batch 4, gradient accumulation 4, effective optimizer batch 16, 80 updates/epoch, 320 updates over four epochs, identical data exposure, LR `1e-4`, cosine-to-`1e-5` schedule with 5% warmup, fused AdamW, BF16, no gradient checkpointing, and checkpointing after every epoch.
- Exact SIGReg used four recomputed chunks of four for one joint statistic over 16 source and 16 target representations. It is the previously verified sufficient-statistic/VJP implementation: no queue, stale embeddings, or averaging of batch-4 losses.
- A direct physical-batch-16 SIGReg attempt exceeded memory when an auxiliary-active batch exercised both branches. It was discarded before the controlled comparison. The final physical-4/accumulation-4 setup was used for both conditions and does not change the scientific objective or optimizer cadence.
- The fixed evaluation panel was selected before inspecting checkpoint outputs: seed-533 random selection of 32 identities from each of eight equal-frequency source-length strata in the frozen 1,024-identity validation panel. It contains 256 unique canonical reactions, one view each, source lengths 24–235 tokens (median 70.5), and target lengths 12–114 (median 42). Panel hashes are preserved in the manifest.
- Generation used the established independent batch-1 beam-10 computation. Batched generation at 2 and 4 changed candidates/cutoff outcomes and was rejected. Four independent batch-1 GPU workers reproduced all candidates, validity flags, and top-1/3/5/10 outcomes on the 32-row equivalence panel and reduced that panel from a projected 269 seconds to 117 seconds (2.3x), so this exact sharded path was retained.

The trainer's two-row epoch-4 validation was only a lightweight completion check and is not reported as scientific evidence. All results below come from the prespecified 256-reaction panel.

## A6000 runtime

| Condition | Wall time, 4 epochs | Optimizer steps | Examples/s | Native tokens/s | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| Native | 20.58 min | 320 | 4.15 | 506.8 | 6.16 GiB |
| k=0 JEPA + SIGReg-16 | 24.33 min | 320 | 3.51 | 428.7 | 14.92 GiB |

The SIGReg run's overhead was 18.2% in end-to-end training time. It performed 1,968 model calls versus native's 1,280 because active exact-SIGReg updates require recomputation. There was no CPU/shared-memory paging in the retained runs.

## Optimization trajectory

Epoch values are means over the 80 optimizer updates; auxiliary losses are means over active updates only.

| Epoch | Native-control NTP | SIGReg-run NTP | Cosine JEPA | SIGReg | Active updates |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.1603 | 0.9916 | 0.19853 | 7.0744 | 47/80 |
| 2 | 0.3031 | 0.2984 | 0.00541 | 7.6186 | 43/80 |
| 3 | 0.2057 | 0.1950 | 0.00237 | 7.4762 | 42/80 |
| 4 | 0.1700 | 0.1572 | 0.00200 | 7.4345 | 40/80 |

NTP optimized normally in both conditions, so the run was not numerically unstable or dominated into failure. The mechanistic failure is instead clear: cosine JEPA fell by about 99% from epoch 1 to epoch 4, while SIGReg ended 5.1% above its epoch-1 mean and showed no sustained improvement. With the fixed literature-grounded coefficient and N=16 statistic, SIGReg did not exert enough effective pressure to stop the cosine solution.

## Frozen generation and target CE

| Epoch | Condition | Top-1 | Top-3 | Top-5 | Top-10 | Valid candidates | Aggregate target CE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | Native | 5/256 (1.95%) | 17/256 (6.64%) | 35/256 (13.67%) | 43/256 (16.80%) | 73.20% | 0.251414 |
| 2 | SIGReg-16 | 7/256 (2.73%) | 15/256 (5.86%) | 27/256 (10.55%) | 38/256 (14.84%) | 86.68% | 0.255941 |
| 4 | Native | 6/256 (2.34%) | 26/256 (10.16%) | 40/256 (15.63%) | 52/256 (20.31%) | 78.20% | 0.240683 |
| 4 | SIGReg-16 | 3/256 (1.17%) | 20/256 (7.81%) | 32/256 (12.50%) | 46/256 (17.97%) | 86.76% | 0.243811 |

Paired epoch-2 top-1 outcomes were 4 both correct, 1 native-only, 3 SIGReg-only, and 248 neither. At epoch 4 they were 3 both, 3 native-only, 0 SIGReg-only, and 250 neither. SIGReg improved validity, but validity did not convert into correct-product rank: at epoch 2 rank improved/worsened/tied on 19/22/215 reactions; at epoch 4, 19/31/206. The mean rank differences and their bootstrap intervals also favored neither SIGReg checkpoint reliably.

Per-reaction CE was similarly null. At epoch 2 the mean `native CE - SIGReg CE` was -0.00105 (95% bootstrap CI `[-0.00673, 0.00472]`, Wilcoxon `p=0.745`); at epoch 4 it was +0.00040 (`[-0.00560, 0.00647]`, `p=0.851`). Exactly 52.7% of reactions improved at both checkpoints, despite the token-weighted aggregate CE being worse.

## Representation geometry

All conditions use the requested k=0 final-source-EOS readout. The analysis-only residual is joint source/target mean centering followed by removal of shared top PCs.

| Epoch | Condition | Source variance | Target variance | Source / target mean energy | Source / target effective rank | Raw margin | Raw retrieval | Residual PC2 margin / retrieval |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | Native | 2.028e-3 | 3.203e-3 | 0.9742 / 0.9577 | 36.58 / 19.61 | 0.007600 | 44.9% | 0.2512 / 75.0% |
| 2 | SIGReg-16 | 1.655e-4 | 3.277e-4 | 0.9980 / 0.9959 | 41.45 / 11.27 | 0.000486 | 58.6% | 0.2559 / 72.7% |
| 4 | Native | 1.431e-3 | 2.320e-3 | 0.9818 / 0.9695 | 41.00 / 22.61 | 0.005479 | 43.4% | 0.2588 / 76.6% |
| 4 | SIGReg-16 | 8.492e-5 | 1.568e-4 | 0.9988 / 0.9978 | 45.28 / 15.88 | 0.000301 | 57.8% | 0.2490 / 73.0% |

Large raw retrieval ratios are not sufficient evidence of restored scale: their absolute margins are small and four-way retrieval chance is 25%. The scale measurements provide the relevant comparison. By epoch 4, SIGReg's raw pair margin was 18.2x smaller than native's. Yet after removing the common component, its margin/retrieval was close to native; after removing four PCs, SIGReg was `0.2844 / 78.5%` and native `0.2829 / 79.3%`. Pair-specific information survived in a multidimensional residual even as the raw space contracted.

## Source sensitivity and decoder coupling

At epoch 4, the SIGReg checkpoint's raw EOS representation was much less responsive than native to source interventions even though its residual and normal generative pathway remained responsive:

| Intervention | Native raw sensitivity | SIGReg raw sensitivity | Native / SIGReg residual-PC2 sensitivity | Native / SIGReg target-CE change |
|---|---:|---:|---:|---:|
| Contributor removal | 0.02546 | 0.01175 | 0.4437 / 0.4510 | 0.5633 / 0.5889 |
| Contributor replacement | 0.00560 | 0.000390 | 0.3390 / 0.3796 | 0.7431 / 0.7695 |
| Unrelated source | 0.01587 | 0.000996 | 0.8864 / 0.8962 | 0.9071 / 0.9300 |

Thus the SIGReg raw representation was 14.4x less sensitive to contributor replacement and 15.9x less sensitive to an unrelated source, while teacher-forced target CE changed at least as much as native. The signal is again present after common-component removal, but the raw auxiliary readout is poorly scaled relative to the decoder-visible response.

Across reactions, neither raw nor residual pair signal predicted decoder improvement. At epoch 2, Spearman correlations with CE improvement were -0.005 (95% bootstrap CI `[-0.143, 0.125]`) raw and -0.036 (`[-0.153, 0.089]`) residual; correlations with rank improvement were 0.011 and 0.061. At epoch 4, CE correlations were 0.044 and 0.047, and rank correlations were 0.021 and -0.050. All intervals included zero. This supports weak/absent decoder coupling, not a hidden generative benefit missed by exact top-1.

## Interpretation and stopping decision

This controlled run answers the central question negatively. At the prescribed N=16/two-view weighting, SIGReg did **not** repair ChemFM's unusually extreme JEPA contraction; the cosine shortcut strengthened while SIGReg remained flat-to-worse. Consequently this experiment cannot test whether successfully repairing global geometry would causally improve generation. It does show that merely adding plain SIGReg at this qualified batch and coefficient is insufficient, and the worse CE/rank/top-k trajectory gives no reason to continue it.

The evidence does not support post-hoc tuning of the SIGReg coefficient on this panel. Consistent with the stopping rule, the next research decision should be made from the existing gradient-response evidence and decoder-coupling diagnosis rather than launching another anti-contraction run automatically.

## Evidence and provenance

- Fixed panel: [`data/sigreg_batch16_pilot/manifest.json`](../../data/sigreg_batch16_pilot/manifest.json)
- Training outputs and all epoch checkpoints: [`runs/sigreg_batch16_pilot/matched_b4/`](../../runs/sigreg_batch16_pilot/matched_b4/)
- Epoch-2 aggregate/paired/coupling results: [`runs/sigreg_batch16_pilot/evaluation/summary_epoch2.json`](../../runs/sigreg_batch16_pilot/evaluation/summary_epoch2.json)
- Epoch-4 aggregate/paired/coupling results: [`runs/sigreg_batch16_pilot/evaluation/summary_epoch4.json`](../../runs/sigreg_batch16_pilot/evaluation/summary_epoch4.json)
- Geometry and caches: [`runs/sigreg_batch16_pilot/evaluation/geometry.json`](../../runs/sigreg_batch16_pilot/evaluation/geometry.json)
- Exact generation rows and intervention diagnostics: [`runs/sigreg_batch16_pilot/evaluation/`](../../runs/sigreg_batch16_pilot/evaluation/)
- Evaluation equivalence benchmarks: [`runs/sigreg_batch16_pilot/evaluation/parallel_benchmark/summary.json`](../../runs/sigreg_batch16_pilot/evaluation/parallel_benchmark/summary.json) and [`runs/sigreg_batch16_pilot/evaluation/beam_benchmark/summary.json`](../../runs/sigreg_batch16_pilot/evaluation/beam_benchmark/summary.json)
- Checksum-verified Thunder archive: [`runs/thunder_sigreg_batch16_transfer/results/sigreg_batch16_pilot_artifacts.tar.gz`](../../runs/thunder_sigreg_batch16_transfer/results/sigreg_batch16_pilot_artifacts.tar.gz), SHA-256 `acc2aa39eb9a51acc01336b9b80e0dda196b57b920afa4559f8c6039370d6ce1`
- A6000 instance `i1b7v30z` was terminated after local checksum and row-count verification.
