# Why the project moved from cosine JEPA to MSE+SIGReg

## Result

Symmetric cosine cLM-JEPA retained multidimensional source-target structure (85.94% residual four-way retrieval after removing four PCs) but compressed it into a small residual around a dominant common direction. This was not classical rank collapse: effective rank remained 56.50. The residual signal did not predict improvement in ChemFM generation or target-token probability. Stop-gradient and several cosine+SIGReg controls either only partially relaxed the geometry or introduced a cadence confound. A cadence-matched batch-16 cosine+SIGReg run confirmed that the prescribed regularization did not prevent contraction. These results motivated the controlled MSE and MSE+SIGReg ablation in [report 02](02_MSE_SIGREG_EXPERIMENT.md).

## Original symmetric cosine result

The seed-533 native and k=1 symmetric cosine checkpoints were selected at epoch 3. They tied at 2/32 exact top-1 on the original Gate 5 selector; the larger frozen one-view panel did not reveal a hidden advantage.

| 512-reaction metric | Native | Symmetric cosine cLM-JEPA |
|---|---:|---:|
| Exact top-1 | 24/512 (4.688%) | 17/512 (3.320%) |
| Top-3 | 14.063% | 13.672% |
| Top-5 | 20.508% | 20.313% |
| Top-10 | 26.563% | 26.563% |
| Target-token CE | 0.239942 | 0.238685 |

Top-1 had 12 native-only and 5 cLM-JEPA-only successes, a -1.367 pp difference (95% bootstrap CI `[-2.93,+0.20]`, exact McNemar `p=0.143`). The 0.524% aggregate CE advantage was not broad: its per-reaction interval crossed zero and Wilcoxon `p=0.367`.

### Geometry and pair signal

On 1,024 fixed identities:

| Checkpoint | Target variance | Effective rank | Mean-direction energy | Raw pair margin | Raw retrieval |
|---|---:|---:|---:|---:|---:|
| Base ChemFM | 0.022171 | 26.16 | 0.674849 | 0.025028 | 34.38% |
| Native epoch 3 | 0.015871 | 3.05 | 0.779364 | 0.016019 | 23.63% |
| Symmetric cosine epoch 3 | 0.00003206 | 56.50 | 0.999617 | 0.00009372 | 72.27% |

The cLM-JEPA target variance was 692x below base and 495x below native. High effective rank rules out a one-dimensional/constant representation. Common-component removal exposed the pair structure:

| Analysis-only representation | Pair margin | Retrieval |
|---|---:|---:|
| Raw | 0.0000937 | 72.27% |
| Mean centered | 0.213060 | 76.27% |
| Centered, remove PC1 | 0.251651 | 81.64% |
| Centered, remove top 2 PCs | 0.285808 | 84.77% |
| Centered, remove top 4 PCs | 0.273486 | 85.94% |

The measured failure is therefore extreme variance contraction/common-direction concentration with informative residual structure, not classical representation collapse.

### Decoder coupling

Reaction-level pair strength did not identify reactions helped by cLM-JEPA:

| Signal | Outcome | Spearman rho | 95% bootstrap CI |
|---|---|---:|---:|
| Raw margin | CE improvement | -0.056 | `[-0.140,+0.033]` |
| Residual margin | CE improvement | -0.054 | `[-0.142,+0.037]` |
| Raw margin | rank improvement | 0.006 | `[-0.076,+0.095]` |
| Residual margin | rank improvement | 0.003 | `[-0.080,+0.085]` |

Raw cLM-JEPA representation sensitivity was tiny under contributor removal/replacement/unrelated-source interventions (`0.000337/0.000111/0.000271`), while its decoder CE changed by `0.5541/0.7595/0.9221`. Residual sensitivities were `0.4622/0.3991/0.9106`. Chemistry was present in the residual but weakly expressed by raw cosine and not measurably coupled to decoder improvement.

## Controlled attempts before MSE

| Experiment | Primary generation result | Geometry result | Interpretation |
|---|---|---|---|
| Target stop-gradient k=1 | 26/512 vs native 24/512; +0.39 pp, CI `[-1.37,+2.15]`, `p=0.8318`; CE 7.69% worse | Target variance 12.03x symmetric but still 46.3x below native | Partial geometric change; no established generation gain |
| SIGReg batch 2, k=0/k=1 | 2/256 and 3/256 vs native 7/256 | SIGReg did not decrease; both remained highly concentrated | Statistical batch too small; no k conclusion from top-1 |
| SIGReg batch 128, k=0 | 0/256; CE 1.080978 | Variance moved near native scale | Uninterpretable as a controlled performance test: optimizer updates fell 320 to 20 and auxiliary activity was 15/20 |
| SIGReg batch-16 preflight | No performance evaluation | Exact streamed/direct gradients matched; applied SIGReg gradient was 1.20% of NTP | Required cadence-matched native control |
| Frozen gradient response | No optimizer steps | SIGReg endpoint norm stayed 0.0376-0.0427 across contraction | SIGReg force did not vanish in the observed common-direction regime |
| Cadence-matched SIGReg-16 k=0 | Epoch-4 3/256 vs native 6/256; CE 1.30% worse | Source/target variance 16.9x/14.8x below native | Plain cosine+SIGReg-16 did not prevent the shortcut |

The batch-16 run used the exact objective `L_NTP + active * 2 * [L_cos + (0.01/0.99)L_SIGReg]`, effective batch 16, 80 updates/epoch, four epochs, and a matched native control. Cosine loss fell from 0.19853 at epoch 1 to 0.00200 at epoch 4 while SIGReg moved from 7.0744 to 7.4345. Residual PC2 retrieval remained 73.0%, despite raw source/target mean-direction energy of 0.9988/0.9978.

## Reduced upstream LLM-JEPA reference

A separate two-epoch, one-seed, rank-16 LoRA DeepSeek-1.5B/GSM8K run used the official LLM-JEPA repository. It was not a successful behavioral control: NTP/JEPA scored 36/300 and 28/300 (`p=0.229`). Its geometry still provides a bounded comparison. LLM-JEPA epoch-2 target variance was only 1.45x below matched NTP rather than ChemFM's 495x, target effective rank was 73.35, and raw 300-way/four-way retrieval was 67.67%/94.00%. This does not support LoRA alone as a sufficient explanation for ChemFM's extreme residualization, but the unsuccessful reduced behavior prevents using it as a positive reference.

## Decision leading to MSE+SIGReg

The controlled evidence supported changing the prediction metric, not adding more cosine-specific tuning:

1. Cosine could approach zero by amplifying a common direction while retaining chemistry in a small residual.
2. SIGReg at batch 16 retained gradient but did not stop that trajectory under the fixed two-view coefficient.
3. Repairing geometry in the cadence-confounded batch-128 run did not establish decoder benefit.
4. The official LLM-JEPA repository includes unnormalized squared-Euclidean prediction, and LeJEPA combines squared-Euclidean prediction with SIGReg.

The next controlled experiment therefore compared raw MSE with MSE+exact-SIGReg-16 at k=0, without projector, stop-gradient, EMA, or coefficient search.

## Evidence paths

- Original geometry/coupling: `runs/diagnostics/decoder_coupling/`, `runs/diagnostics/uspto_mit_geometry_diagnosis.json`, `runs/diagnostics/geometry_cache/`
- Target stop-gradient: `runs/diagnostics/target_sg_rescue_512/`
- Batch-2 k ablation: `runs/diagnostics/sigreg_k_ablation_256/`
- Batch-128: `runs/diagnostics/sigreg_k0_batch128_256/`
- Batch-16 preflight/gradient assay: `runs/diagnostics/sigreg_batch16_preflight.json`, `runs/diagnostics/sigreg_gradient_response.json`
- Cadence-matched batch-16: `runs/sigreg_batch16_pilot/`
- GSM8K reference: `runs/diagnostics/llm_jepa_reference/`
