# USPTO-MIT MSE cLM-JEPA ablation

## Result

**Raw MSE alone did not avoid ChemFM's contraction shortcut. MSE plus exact batch-16 SIGReg did materially repair the endpoint geometry, but the repair did not improve reaction generation or decoder coupling.**

At the prespecified epoch-2 gate, plain MSE retained only 25.4%/25.3% of the exposure-matched native source/target variance and concentrated 99.35%/98.99% of source/target energy in the mean direction. MSE+SIGReg was substantially healthier: it retained 79.2%/47.2% of native variance, reduced mean-direction energy to 96.65%/96.79%, and achieved 83.98% raw four-way pair retrieval. It was therefore the only condition continued to epoch 4.

The geometric repair persisted at epoch 4, but the behavioral outcome was negative. MSE+SIGReg tied native exact top-1 at 6/256, scored worse at top-3/5/10, and had 3.36% worse aggregate target-token CE. Pair strength was unrelated to CE or rank improvement. The experiment therefore supports **global endpoint geometry being insufficient for useful ChemFM generation**; it shifts the bottleneck toward the JEPA objective's coupling to the autoregressive pathway rather than toward adding more anti-contraction pressure.

No projector, stop-gradient, EMA, alternative k, new seed, coefficient sweep, larger dataset, or follow-up objective was run.

## Primary-source verification and exact objectives

The objective was checked independently before training against the [LLM-JEPA paper](https://arxiv.org/abs/2509.14252), the [official LLM-JEPA implementation](https://github.com/galilai-group/llm-jepa/blob/main/finetune.py), the [LeJEPA paper](https://arxiv.org/abs/2511.08544), and the [official LeJEPA repository](https://github.com/galilai-group/lejepa). The upstream LLM-JEPA MSE branch is exactly:

`mean((z_source - z_target) ** 2)`

It does not normalize endpoints. LeJEPA is not cosine plus SIGReg: it combines a squared-Euclidean view-center prediction loss with per-view SIGReg. For two views `a,b`, its prediction term is

`mean(((a+b)/2-a)^2 + ((a+b)/2-b)^2) / 2 = mean((a-b)^2) / 4`.

Consequently, rescaling LeJEPA's mixture so raw pairwise MSE retains coefficient one gives

`L_aux = L_MSE + [4 * 0.01 / 0.99] * L_SIGReg`

`      = L_MSE + 0.040404... * L_SIGReg`.

The two trained conditions were therefore:

- MSE: `L = L_NTP + active * 2 * L_MSE`.
- MSE+SIGReg-16: `L = L_NTP + active * 2 * [L_MSE + 0.040404... * L_SIGReg]`.

`active` is the established 50% auxiliary-dropout policy, giving expected coefficient one on raw MSE. Gradients were symmetric through source and target. SIGReg separately regularized the 16 source and 16 target EOS states and averaged the two distribution statistics.

Focused tests verified the exact upstream MSE value, the two-view factor-of-four algebra, and materialized-versus-streamed MSE+SIGReg values and parameter gradients. The final local focused suite passed: `23 passed`.

## Controlled protocol

- ChemFM-1B, USPTO-MIT forward prediction, fixed 1,280-row training manifest, seed 533.
- k=0 final source EOS and target EOS states; no `[PRED]` token.
- Symmetric gradients, unchanged native NTP, LoRA configuration, serialization, data order, LR `1e-4`, AdamW parameters, cosine-to-`1e-5` scheduler, 5% warmup, and BF16.
- Physical batch 4, gradient accumulation 4, effective optimizer batch 16, 80 updates/epoch.
- Exact batch-16 SIGReg used the verified two-pass sufficient-statistic/VJP computation. There was no queue, stale embedding bank, or average of smaller SIGReg losses.
- Both stage-1 jobs were planned with a four-epoch scheduler and stopped after epoch 2. This allowed exact resume of only the selected condition without restarting or changing the schedule.
- Evaluation used the same seed-533, length-stratified 256-identity panel prespecified for the cadence-matched batch-16 pilot: unique canonical reactions, one view per identity, and exact beam 10.

One unavoidable dropout-semantics detail is explicit: plain MSE samples auxiliary activity at the established physical-microbatch granularity, whereas exact SIGReg requires one activity decision for the joint 16-example statistic. Expected auxiliary exposure is unchanged, but realized activity granularity differs. The epoch-4 behavioral comparison is against native NTP, for which this distinction is irrelevant; the stage-1 MSE-versus-MSE+SIGReg contrast should not be interpreted as isolating SIGReg activity variance perfectly.

## A6000 execution optimization

The retained A6000 path used PyTorch SDPA, BF16, fused AdamW, dynamic padding, pinned host memory, `zero_grad(set_to_none=True)`, no activation checkpointing, and physical batch 4. It avoided all CPU/shared-memory offload and paging. A diagnostic hot-loop change replaced one CPU synchronization per parameter norm with one stacked GPU reduction and final synchronization while preserving the same reported value.

Short real-data benchmarks rejected changing scientific batch/accumulation, unsafe precision, token limits, or objective semantics. `torch.compile`, attention reimplementation, and activation checkpointing were not retained because they did not offer a stable measured end-to-end benefit for this variable-length LoRA workload. The selected path used about 15 GiB peak VRAM, leaving ample A6000 headroom without paying checkpointing or paging overhead.

| Condition | Exposure | Wall time | Examples/s | Peak VRAM |
|---|---:|---:|---:|---:|
| MSE | 2 epochs / 2,560 examples | 13.87 min | 3.08 | 14.88 GiB |
| MSE+SIGReg | 2 epochs / 2,560 examples | 13.45 min | 3.17 | 14.92 GiB |
| MSE+SIGReg | 4 epochs / 5,120 examples | 27.09 min | 3.15 | 14.80 GiB |
| Existing matched native | 4 epochs / 5,120 examples | 20.58 min | 4.15 | 6.16 GiB |

SIGReg's four-chunk no-graph statistics pass plus exact recomputation explains its remaining overhead over native. No performance result was obtained by weakening or approximating that computation.

## Stage 1: epoch-2 mechanism gate

Losses are epoch means; auxiliary columns include only active updates.

| Condition | Epoch | NTP | MSE | SIGReg | Combined auxiliary | Active updates |
|---|---:|---:|---:|---:|---:|---:|
| MSE | 1 | 1.06584 | 0.032813 | — | 0.032813 | 76/80 |
| MSE | 2 | 0.30591 | 0.003569 | — | 0.003569 | 72/80 |
| MSE+SIGReg | 1 | 1.17629 | 0.035154 | 6.6591 | 0.30421 | 47/80 |
| MSE+SIGReg | 2 | 0.30665 | 0.004796 | 6.9490 | 0.28556 | 43/80 |

Both native losses descended normally and all gradients remained finite. Plain MSE fell by 89.1% from its epoch-1 mean; MSE+SIGReg's MSE fell by 86.4%. SIGReg itself did not decrease—it rose 4.35%—so the run did not directly minimize its empirical characteristic-function statistic over this short pilot. Nevertheless, the endpoint geometry changed materially.

| Epoch-2 condition | Source / target variance | Mean-direction energy | Effective rank | Raw margin / retrieval | Residual PC2 margin / retrieval |
|---|---:|---:|---:|---:|---:|
| Native | 2.028e-3 / 3.203e-3 | 0.9742 / 0.9577 | 36.58 / 19.61 | 0.007600 / 44.9% | 0.2512 / 75.0% |
| Prior cosine+SIGReg-16 k=0 | 1.655e-4 / 3.277e-4 | 0.9980 / 0.9959 | 41.45 / 11.27 | 0.000486 / 58.6% | 0.2559 / 72.7% |
| MSE | 5.147e-4 / 8.114e-4 | 0.9935 / 0.9899 | 43.43 / 20.27 | 0.001650 / 64.5% | 0.2890 / 83.6% |
| **MSE+SIGReg-16** | **1.607e-3 / 1.513e-3** | **0.9665 / 0.9679** | **37.61 / 33.07** | **0.011916 / 84.0%** | **0.3937 / 85.2%** |

Plain MSE is 3.11x/2.48x higher-variance than the prior cosine+SIGReg trajectory, so changing the metric clearly weakens the cosine shortcut. Absolute geometry is still unhealthy relative to native: source and target variance remain 3.94x/3.95x lower and mean-direction energy remains near one. Raw MSE therefore does **not** solve the contraction by itself.

SIGReg adds a clear absolute benefit: relative to MSE alone it raises source variance 3.12x and target variance 1.86x, reduces common-direction concentration on both branches, balances target effective rank, and raises raw margin/retrieval. Embedding norms also move from about 12.78/12.80 under MSE to 9.92/9.82, while mean-vector norms fall to 9.75/9.66. This is not a large ratio manufactured from a tiny baseline; several absolute geometry measures move toward or beyond native scale. MSE+SIGReg passed the gate and was continued. Plain MSE was stopped as specified.

Stage-1 target CE did not favor the geometrically healthier checkpoint: native was `0.251414`, MSE `0.257012`, prior cosine+SIGReg `0.255941`, and MSE+SIGReg `0.261326`. The gate was therefore a representation-mechanism decision, not early performance selection.

## Stage 2: epoch-4 behavior

MSE+SIGReg resumed exactly from epoch 2. Its active MSE decreased to `0.002470` at epoch 3 and `0.002154` at epoch 4; SIGReg remained essentially flat at `6.9976` and `6.9992`. NTP continued descending to epoch means `0.19851` and `0.15981`.

### Generation and target CE

| Epoch-4 condition | Top-1 | Top-3 | Top-5 | Top-10 | Valid candidates | Aggregate target CE |
|---|---:|---:|---:|---:|---:|---:|
| Native | 6/256 (2.34%) | 26/256 (10.16%) | 40/256 (15.63%) | 52/256 (20.31%) | 78.20% | **0.240683** |
| MSE+SIGReg | 6/256 (2.34%) | 24/256 (9.38%) | 34/256 (13.28%) | 49/256 (19.14%) | **86.60%** | 0.248779 |

Exact top-1 was a true tie: four reactions were correct under both, two native-only, two MSE+SIGReg-only, and 248 neither. The paired top-1 difference was `0.00 pp` with bootstrap 95% CI `[-1.56,+1.56] pp`; exact McNemar `p=1.0`.

The secondary direction was adverse. Correct-product rank improved/worsened/tied on 18/35/203 reactions, and mean rank improvement was `-0.137` with 95% CI `[-0.383,0.105]`. MSE+SIGReg's aggregate CE was 3.36% worse. Mean per-reaction `native CE - MSE+SIGReg CE` was `-0.00773`, 95% CI `[-0.01386,-0.00188]`; 44.5% improved and 55.5% worsened; Wilcoxon `p=0.0221`. Improved validity did not translate into more correct chemistry.

### Epoch-4 geometry

| Condition | Source / target variance | Mean-direction energy | Effective rank | Embedding norm / mean-vector norm | Raw margin / retrieval | Residual PC2 margin / retrieval |
|---|---:|---:|---:|---:|---:|---:|
| Native | 1.431e-3 / 2.320e-3 | 0.9818 / 0.9695 | 41.00 / 22.61 | 12.68/12.57; 12.48/12.29 | 0.005479 / 43.4% | 0.2588 / 76.6% |
| **MSE+SIGReg** | **1.293e-3 / 1.281e-3** | **0.9709 / 0.9698** | **38.36 / 34.06** | **9.53/9.39; 9.32/9.18** | **0.011634 / 85.9%** | **0.4360 / 93.8%** |

The geometric effect is durable and absolute: source variance is 90.3% of native and target variance 55.2%; common-direction energy is close to native rather than approximately 0.998–0.999; raw pair margin is 2.12x native and retrieval is far above the 25% matched-set chance level. Strong multidimensional residual information also remains. This is neither the original extreme concentration nor classical representation collapse.

### Source sensitivity and coupling

| Intervention | Native raw sensitivity | MSE+SIGReg raw sensitivity | Native / MSE+SIGReg residual-PC2 sensitivity | Native / MSE+SIGReg target-CE change |
|---|---:|---:|---:|---:|
| Contributor removal | 0.02546 | 0.03181 | 0.4437 / 0.4330 | 0.5633 / 0.5846 |
| Contributor replacement | 0.00560 | 0.01036 | 0.3390 / 0.4224 | 0.7431 / 0.7781 |
| Unrelated source | 0.01587 | 0.02468 | 0.8864 / 0.9484 | 0.9071 / 0.9469 |

Unlike the earlier contracted cosine readout, the repaired k=0 state is not unusually insensitive: all three raw source sensitivities exceed native, and its generative pathway also responds strongly. The remaining failure is **useful coupling**, not simple source invariance. Across reactions, raw pair margin had Spearman rho `0.028` with CE improvement and `-0.086` with rank improvement; residual-PC2 margin had rho `0.090` and `-0.110`. Every bootstrap interval included zero. Stronger JEPA pair structure therefore did not predict where generation improved.

## Answers and stopping decision

### 1. Does raw MSE avoid cosine's common-direction shortcut?

**Partially, but not sufficiently.** It gives several-fold healthier variance than the directly comparable cosine+SIGReg-16 trajectory, yet still contracts source and target variance about fourfold below native and remains heavily common-direction dominated. Plain MSE failed the epoch-2 continuation gate.

### 2. Does SIGReg add meaningful protection against MSE contraction?

**Yes.** MSE+SIGReg materially restores both branches' absolute variance, mean-direction energy, target effective rank, raw pair discrimination, and source-intervention sensitivity. The repair persists through epoch 4 despite the scalar SIGReg statistic not decreasing, showing why scalar loss alone is not an adequate mechanism diagnostic.

### 3. Does repaired geometry become decoder-visible and improve generation?

**No.** Exact top-1 ties native, all wider beam cutoffs are worse, CE is significantly worse, rank changes favor native, and reaction-level JEPA pair strength does not predict decoder improvement. Healthy global endpoint geometry and strong source-target pairing are therefore insufficient in this formulation.

The most supported interpretation is that ChemFM's original extreme geometric shortcut was real and avoidable, but it was not the sole cause of failed reaction prediction. Further anti-contraction regularization is not justified by this result. Any later experiment should address how the auxiliary relationship is coupled to the autoregressive decoder, but none is launched here.

## Evidence and provenance

- Stage-1 training and epoch checkpoints: [`runs/mse_ablation/stage1/`](../runs/mse_ablation/stage1/)
- Selected resumed training result: [`runs/mse_ablation/stage2/mse_sigreg.json`](../runs/mse_ablation/stage2/mse_sigreg.json)
- Epoch-2 and epoch-4 geometry/CE/intervention artifacts: [`runs/mse_ablation/evaluation/`](../runs/mse_ablation/evaluation/)
- Exact 256-row beam output: [`runs/mse_ablation/evaluation/mse_sigreg_epoch4_generation.jsonl`](../runs/mse_ablation/evaluation/mse_sigreg_epoch4_generation.jsonl)
- Paired summary with bootstrap/correlation analyses: [`runs/mse_ablation/evaluation/summary_epoch4.json`](../runs/mse_ablation/evaluation/summary_epoch4.json)
- Checksum-verified Thunder archive: [`runs/thunder_sigreg_batch16_transfer/results/mse_ablation_artifacts.tar.gz`](../runs/thunder_sigreg_batch16_transfer/results/mse_ablation_artifacts.tar.gz), SHA-256 `4f04e4fb7979ef861a5dbc9cb37da7e3e1691271a936ee53537156c0298b9b0f`.
- Thunder instance `drjr0grt` was deleted after local checksum, checkpoint-count, and generation-row-count verification.
