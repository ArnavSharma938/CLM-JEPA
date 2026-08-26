# Endpoint cLM-JEPA objective experiments

## Measured summary

Raw MSE retained 25.4%/25.3% of native source/target variance at epoch 2.
MSE+exact-SIGReg-16 retained 90.3%/55.2% at epoch 4. Its epoch-4
exact top-1 tied native at 6/256; top-3/5/10 were 24/34/49 versus native
26/40/52, and historical aggregate target-token CE was 0.248779 versus
0.240683 native. Pair-margin correlations with CE and rank change had
bootstrap intervals containing zero.

This report also contains the precursor cosine, stop-gradient, and
cosine+SIGReg measurements formerly stored in report 01.

## Precursor cosine and SIGReg measurements

### Symmetric cosine endpoint

The seed-533 native and k=1 symmetric cosine checkpoints were selected at
epoch 3 and tied at 2/32 exact top-1 on the Gate-5 selector.

| 512-reaction metric | Native | Symmetric cosine cLM-JEPA |
|---|---:|---:|
| Exact top-1 | 24/512 (4.688%) | 17/512 (3.320%) |
| Top-3 | 14.063% | 13.672% |
| Top-5 | 20.508% | 20.313% |
| Top-10 | 26.563% | 26.563% |
| Target-token CE | 0.239942 | 0.238685 |

Top-1 had 12 native-only and 5 cLM-JEPA-only successes, a -1.367 pp
difference (95% bootstrap CI `[-2.93,+0.20]`, exact McNemar `p=0.143`).
The aggregate CE difference was 0.524%; its per-reaction interval crossed zero
and Wilcoxon `p=0.367`.

On 1,024 fixed identities:

| Checkpoint | Target variance | Effective rank | Mean-direction energy | Raw pair margin | Raw retrieval |
|---|---:|---:|---:|---:|---:|
| Base ChemFM | 0.022171 | 26.16 | 0.674849 | 0.025028 | 34.38% |
| Native epoch 3 | 0.015871 | 3.05 | 0.779364 | 0.016019 | 23.63% |
| Symmetric cosine epoch 3 | 0.00003206 | 56.50 | 0.999617 | 0.00009372 | 72.27% |

The symmetric endpoint's target variance was 692x below base and 495x below
native. Common-component removal produced:

| Analysis-only representation | Pair margin | Retrieval |
|---|---:|---:|
| Raw | 0.0000937 | 72.27% |
| Mean centered | 0.213060 | 76.27% |
| Centered, remove PC1 | 0.251651 | 81.64% |
| Centered, remove top 2 PCs | 0.285808 | 84.77% |
| Centered, remove top 4 PCs | 0.273486 | 85.94% |

Reaction-level pair metrics and decoder changes were:

| Signal | Outcome | Spearman rho | 95% bootstrap CI |
|---|---|---:|---:|
| Raw margin | CE improvement | -0.056 | `[-0.140,+0.033]` |
| Residual margin | CE improvement | -0.054 | `[-0.142,+0.037]` |
| Raw margin | Rank improvement | 0.006 | `[-0.076,+0.095]` |
| Residual margin | Rank improvement | 0.003 | `[-0.080,+0.085]` |

Raw representation sensitivity under contributor removal, contributor
replacement, and unrelated-source substitution was
`0.000337/0.000111/0.000271`; decoder CE changed by
`0.5541/0.7595/0.9221`. Residual sensitivities were
`0.4622/0.3991/0.9106`.

### Controls preceding raw MSE

| Experiment | Primary generation measurement | Geometry measurement |
|---|---|---|
| Target stop-gradient k=1 | 26/512 vs native 24/512; +0.39 pp, CI `[-1.37,+2.15]`, `p=0.8318`; CE 7.69% higher | Target variance 12.03x symmetric and 46.3x below native |
| SIGReg batch 2, k=0/k=1 | 2/256 and 3/256 vs native 7/256 | Both remained concentrated; the statistic used two samples/view |
| SIGReg batch 128, k=0 | 0/256; CE 1.080978 | Variance near native scale; optimizer updates were 20 versus 320 and auxiliary activity was 15/20 |
| SIGReg batch-16 preflight | No performance evaluation | Streamed/direct gradients matched; applied SIGReg gradient was 1.20% of NTP |
| Frozen gradient response | No optimizer steps | SIGReg endpoint norm was 0.0376--0.0427 across the measured checkpoints |
| Cadence-matched SIGReg-16 k=0 | Epoch-4 3/256 vs native 6/256; CE 1.30% higher | Source/target variance 16.9x/14.8x below native |

The batch-16 objective was
`L_NTP + active * 2 * [L_cos + (0.01/0.99)L_SIGReg]`, with effective batch
16, 80 updates/epoch, and four epochs. Cosine loss changed from 0.19853 at
epoch 1 to 0.00200 at epoch 4; SIGReg changed from 7.0744 to 7.4345.
Residual-PC2 retrieval was 73.0%, and raw source/target mean-direction energy
was 0.9988/0.9978.

### Reduced upstream LLM-JEPA reference

A two-epoch, one-seed, rank-16 LoRA DeepSeek-1.5B/GSM8K run used the official
LLM-JEPA repository. NTP/JEPA accuracy was 36/300 and 28/300 (`p=0.229`).
JEPA epoch-2 target variance was 1.45x below matched NTP, target effective rank
was 73.35, and raw 300-way/four-way retrieval was 67.67%/94.00%.

Precursor evidence paths:

- Original geometry/coupling: `runs/diagnostics/decoder_coupling/`,
  `runs/diagnostics/uspto_mit_geometry_diagnosis.json`,
  `runs/diagnostics/geometry_cache/`
- Target stop-gradient: `runs/diagnostics/target_sg_rescue_512/`
- Batch-2 k ablation: `runs/diagnostics/sigreg_k_ablation_256/`
- Batch-128: `runs/diagnostics/sigreg_k0_batch128_256/`
- Batch-16 preflight/gradient assay:
  `runs/diagnostics/sigreg_batch16_preflight.json`,
  `runs/diagnostics/sigreg_gradient_response.json`
- Cadence-matched batch-16: `runs/sigreg_batch16_pilot/`
- GSM8K reference: `runs/diagnostics/llm_jepa_reference/`

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

SIGReg changed from `6.6591` at epoch 1 to `6.9992` at epoch 4. The frozen continuation gate used representation-distribution measurements rather than the scalar SIGReg loss.

## Epoch-4 comparison

### Generation and CE

| Endpoint | Native | MSE+SIGReg | Difference/direction |
|---|---:|---:|---|
| Exact top-1 | 6/256 (2.34%) | 6/256 (2.34%) | 0.00 pp; CI `[-1.56,+1.56]`; McNemar `p=1.0` |
| Top-3 | 26/256 (10.16%) | 24/256 (9.38%) | native higher |
| Top-5 | 40/256 (15.63%) | 34/256 (13.28%) | native higher |
| Top-10 | 52/256 (20.31%) | 49/256 (19.14%) | native higher |
| Valid candidates | 78.20% | 86.60% | MSE+SIGReg higher |
| Historical aggregate CE | 0.240683 | 0.248779 | MSE+SIGReg 3.36% higher |

Top-1 paired outcomes were 4 both correct, 2 native-only, 2 MSE+SIGReg-only, and 248 neither. Correct-product rank improved/worsened/tied on 18/35/203 reactions. Mean rank improvement was `-0.137` with 95% CI `[-0.383,+0.105]`. Mean per-reaction `native CE - MSE+SIGReg CE` was `-0.00773`, 95% CI `[-0.01386,-0.00188]`; 44.5% improved, 55.5% worsened, Wilcoxon `p=0.0221`.

The later mechanistic audit found that this historical aggregate denominator excludes the supervised `<prostart>` and `<eos>` labels: 10,642 raw-product tokens versus 11,154 model labels. Per-reaction CE and the native-vs-cLM direction were unaffected. The correctly label-normalized local reproduction was `0.229702` native and `0.237275` MSE+SIGReg (3.30% higher); see [report 04](04_MECHANISTIC_GRADIENT_AND_BLOCK_SWAP_AUDIT.md).

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

Raw pair margin correlations with CE and rank change were `rho=0.028/-0.086`; residual-PC2 correlations were `0.090/-0.110`. Every bootstrap interval included zero.

## Recorded comparisons

- Raw MSE source/target variance at epoch 2 was 25.4%/25.3% of native.
- MSE+SIGReg source/target variance at epoch 4 was 90.3%/55.2% of native.
- MSE+SIGReg and native each produced 6/256 exact top-1; native had higher
  top-3, top-5, and top-10 counts and lower target-token CE.

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
