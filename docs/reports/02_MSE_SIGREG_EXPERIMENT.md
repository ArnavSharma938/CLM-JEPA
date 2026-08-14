# MSE and MSE+SIGReg experiment

## Result

Raw MSE reduced but did not eliminate the common-direction shortcut. MSE+exact-SIGReg-16 materially restored source/target geometry and source sensitivity. It did not improve generation: at epoch 4 it tied native at 6/256 exact top-1, was worse at top-3/5/10, and had 3.36% worse aggregate target-token CE. Pair strength did not predict CE or rank improvement.

This is the key mechanism result: the extreme cosine geometry was avoidable, but healthy global endpoint geometry was not sufficient for better autoregressive reaction prediction.

## Objectives and protocol

The objective definitions were verified against the official LLM-JEPA and LeJEPA papers/repositories. Upstream LLM-JEPA MSE is unnormalized:

`L_MSE = mean((z_source - z_target)^2)`

For two views, LeJEPA's view-center prediction term is one quarter of raw pairwise MSE. Preserving coefficient one on raw MSE maps the two-view `lambda_sig=0.01` mixture to:

`L_aux = L_MSE + [4 * 0.01 / 0.99] L_SIGReg`

`L_total = L_NTP + active * 2 * L_aux`

The MSE-only condition omitted the SIGReg term. `active` followed the established 50% auxiliary-dropout policy. Both used k=0 source EOS and target EOS, symmetric gradients, ChemFM-1B, seed 533, the fixed 1,280-row USPTO-MIT pilot, LoRA, LR `1e-4`, BF16, physical batch 4, accumulation 4, effective batch 16, 80 updates/epoch, and the same scheduler/data order. SIGReg was computed exactly over 16 source and 16 target representations with the verified two-pass statistic/VJP implementation.

Both conditions trained to epoch 2. Only MSE+SIGReg passed the prespecified geometry gate and continued to epoch 4. No projector, stop-gradient, EMA, alternative k, new seed, or coefficient search was added.

Plain MSE sampled auxiliary activity at physical-microbatch granularity; exact SIGReg required one activity decision for the joint 16-example statistic. Expected auxiliary exposure was unchanged, but realized activity granularity differed, so the epoch-2 MSE-versus-MSE+SIGReg comparison does not isolate activity variance perfectly.

## Epoch-2 mechanism gate

| Condition | Source / target variance | Mean energy source / target | Effective rank source / target | Raw margin / retrieval | Target CE |
|---|---:|---:|---:|---:|---:|
| Native | `2.028e-3 / 3.203e-3` | `0.9742 / 0.9577` | `36.58 / 19.61` | `0.007600 / 44.9%` | 0.251414 |
| Cosine+SIGReg-16 | `1.655e-4 / 3.277e-4` | `0.9980 / 0.9959` | `41.45 / 11.27` | `0.000486 / 58.6%` | 0.255941 |
| MSE | `5.147e-4 / 8.114e-4` | `0.9935 / 0.9899` | `43.43 / 20.27` | `0.001650 / 64.5%` | 0.257012 |
| **MSE+SIGReg-16** | **`1.607e-3 / 1.513e-3`** | **`0.9665 / 0.9679`** | **`37.61 / 33.07`** | **`0.011916 / 84.0%`** | 0.261326 |

Plain MSE retained only 25.4%/25.3% of native source/target variance and remained strongly common-direction dominated. MSE+SIGReg retained 79.2%/47.2%, increased target effective rank, and raised raw four-way retrieval to 83.98%. It was continued; MSE-only was stopped.

Training remained finite and NTP descended normally:

| Condition | Epoch | NTP | MSE | SIGReg | Active updates |
|---|---:|---:|---:|---:|---:|
| MSE | 1 | 1.06584 | 0.032813 | - | 76/80 |
| MSE | 2 | 0.30591 | 0.003569 | - | 72/80 |
| MSE+SIGReg | 1 | 1.17629 | 0.035154 | 6.6591 | 47/80 |
| MSE+SIGReg | 2 | 0.30665 | 0.004796 | 6.9490 | 43/80 |
| MSE+SIGReg | 3 | 0.19851 | 0.002470 | 6.9976 | - |
| MSE+SIGReg | 4 | 0.15981 | 0.002154 | 6.9992 | - |

SIGReg did not decrease. The geometry decision therefore came from the measured representation distribution, not from assuming scalar SIGReg loss had improved.

## Epoch-4 comparison

### Generation and CE

| Endpoint | Native | MSE+SIGReg | Difference/direction |
|---|---:|---:|---|
| Exact top-1 | 6/256 (2.34%) | 6/256 (2.34%) | 0.00 pp; CI `[-1.56,+1.56]`; McNemar `p=1.0` |
| Top-3 | 26/256 (10.16%) | 24/256 (9.38%) | native higher |
| Top-5 | 40/256 (15.63%) | 34/256 (13.28%) | native higher |
| Top-10 | 52/256 (20.31%) | 49/256 (19.14%) | native higher |
| Valid candidates | 78.20% | 86.60% | MSE+SIGReg higher |
| Aggregate target CE | 0.240683 | 0.248779 | MSE+SIGReg 3.36% worse |

Top-1 paired outcomes were 4 both correct, 2 native-only, 2 MSE+SIGReg-only, and 248 neither. Correct-product rank improved/worsened/tied on 18/35/203 reactions. Mean rank improvement was `-0.137` with 95% CI `[-0.383,+0.105]`. Mean per-reaction `native CE - MSE+SIGReg CE` was `-0.00773`, 95% CI `[-0.01386,-0.00188]`; 44.5% improved, 55.5% worsened, Wilcoxon `p=0.0221`.

### Geometry and interventions

| Condition | Source / target variance | Mean energy source / target | Effective rank source / target | Raw margin / retrieval | Residual PC2 margin / retrieval |
|---|---:|---:|---:|---:|---:|
| Native | `1.431e-3 / 2.320e-3` | `0.9818 / 0.9695` | `41.00 / 22.61` | `0.005479 / 43.4%` | `0.2588 / 76.6%` |
| **MSE+SIGReg** | **`1.293e-3 / 1.281e-3`** | **`0.9709 / 0.9698`** | **`38.36 / 34.06`** | **`0.011634 / 85.9%`** | **`0.4360 / 93.8%`** |

MSE+SIGReg retained 90.3%/55.2% of native source/target variance and produced higher raw and residual pair discrimination. Source embedding/mean-vector norms were `9.53/9.39` and target norms were `9.32/9.18`; native values were `12.68/12.57` and `12.48/12.29`.

| Intervention | Native raw | MSE+SIGReg raw | Native / MSE+SIGReg residual PC2 | Native / MSE+SIGReg target-CE change |
|---|---:|---:|---:|---:|
| Contributor removal | 0.02546 | 0.03181 | `0.4437 / 0.4330` | `0.5633 / 0.5846` |
| Contributor replacement | 0.00560 | 0.01036 | `0.3390 / 0.4224` | `0.7431 / 0.7781` |
| Unrelated source | 0.01587 | 0.02468 | `0.8864 / 0.9484` | `0.9071 / 0.9469` |

The MSE+SIGReg representation and normal decoder both responded to the source interventions.

The missing link was reaction-level utility. Raw pair margin correlations with CE and rank improvement were `rho=0.028/-0.086`; residual-PC2 correlations were `0.090/-0.110`. Every bootstrap interval included zero.

## Conclusion

1. Raw MSE was healthier than cosine+SIGReg but still contracted about fourfold below native at epoch 2.
2. Adding exact batch-16 SIGReg to MSE materially and durably repaired global representation geometry.
3. That repair did not improve exact generation, wider beam ranks, CE, or measured decoder coupling.

The result does not support another anti-contraction-only intervention. A later experiment would need to change how the auxiliary relationship affects the autoregressive pathway, while exact generated top-1 remains the primary endpoint.

## Runtime and evidence

| Run | Wall time | Throughput | Peak VRAM |
|---|---:|---:|---:|
| MSE, 2 epochs | 13.87 min | 3.08 examples/s | 14.88 GiB |
| MSE+SIGReg, 2 epochs | 13.45 min | 3.17 examples/s | 14.92 GiB |
| MSE+SIGReg, 4 epochs | 27.09 min | 3.15 examples/s | 14.80 GiB |
| Matched native, 4 epochs | 20.58 min | 4.15 examples/s | 6.16 GiB |

- Checkpoints/training: `runs/mse_ablation/stage1/`, `runs/mse_ablation/stage2/`
- Evaluation and geometry: `runs/mse_ablation/evaluation/`
- Complete 256-reaction candidates: `runs/mse_ablation/evaluation/mse_sigreg_epoch4_generation.jsonl`
- Paired results: `runs/mse_ablation/evaluation/summary_epoch4.json`
- Verified archive: `runs/thunder_sigreg_batch16_transfer/results/mse_ablation_artifacts.tar.gz`
- Archive SHA-256: `4f04e4fb7979ef861a5dbc9cb37da7e3e1691271a936ee53537156c0298b9b0f`
