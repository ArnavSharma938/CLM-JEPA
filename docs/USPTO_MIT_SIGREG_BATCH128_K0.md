# USPTO-MIT k=0 symmetric JEPA + batch-128 SIGReg

## Question and fixed protocol

This experiment asks whether the prior negative SIGReg result was caused by estimating the representation distribution from only two source and two target states. Exactly one new condition was trained: seed 533, k=0 source `<eos>` readout, symmetric cosine JEPA gradients, native ChemFM next-token prediction, and standard SIGReg over **128 source and 128 target representations per estimate**. It stopped after epoch 2. No projector, stop-gradient, EMA, whitening, centering/covariance penalty, extra seed, later epoch, or outcome-based tuning was used.

All data, serialization, LoRA modules/rank/dropout, physical microbatch size 2, LR `1e-4`, AdamW parameters, cosine coefficient, 50% JEPA-loss dropout policy, scheduler family, and seed match the prior reliable setup. The official [LeJEPA implementation](https://github.com/galilai-group/lejepa) at commit `c293d291ca87cd4fddee9d3fffe4e914c7272052` remains the SIGReg authority: Epps-Pulley empirical characteristic function, 17 trapezoidal knots on `[0,3]`, 1,024 normalized Gaussian slices, and the standard 0.05 trade-off. As before, the loss is outer-rescaled to retain this project's frozen cosine coefficient:

`L = L_native + lambda_actual * active * [L_cos + (0.05 / 0.95) * L_SIGReg]`

with `lambda_actual=2` and logical-batch JEPA activity sampled from the unchanged 0.5 Bernoulli policy.

## Exact memory-efficient batch computation

Materializing 128 full ChemFM graphs is unnecessary. Each active logical batch uses an exact two-pass algorithm at one fixed parameter state:

1. A no-graph pass over 64 physical chunks accumulates the sums of the cosine and sine terms of the projected empirical characteristic function for all 128 distinct representations.
2. Those global sufficient statistics determine the exact materialized-batch SIGReg value and its derivative with respect to every representation.
3. A second pass restores each chunk's saved CPU/CUDA RNG state, recomputes the identical representations with gradients, and injects their exact vector-Jacobian products.
4. Parameters are updated only after all 128 examples contribute. No detached queue, stale bank, independent batch-2 loss average, or gradient approximation is used.

Tests compare the streamed statistic against direct materialization, k=0 source selection, forced activity/RNG behavior, and complete tiny-causal-LM loss and parameter gradients on identical token IDs and masks. On eight real 2,048-dimensional ChemFM source/target endpoints, direct and streamed values were both `3.313000679`; maximum absolute representation-gradient error was `5.82e-10`. The full active 128-example smoke step was finite and used 2.43 GB allocated / 2.62 GB reserved.

### Necessarily changed optimization semantics

Exact batch-128 gradients require all representations to be evaluated at one parameter snapshot. Updating every eight examples would make later states stale and would no longer equal a joint batch statistic. Consequently:

- optimizer batch changes from 8 to 128 examples;
- updates change from 160 to 10 per epoch, or 320 to 20 total;
- cosine/native gradients are averaged across the same 128-example logical batch;
- the cosine/SIGReg dropout decision is made once per logical batch rather than per physical chunk;
- the cosine scheduler spans 20 updates with one warmup update rather than 320 updates with 16 warmup updates;
- LR remains `1e-4`; it was not linearly scaled or otherwise compensated.

The seed produced 7 active groups in epoch 1 and 8 in epoch 2. Thus 15/20 logical batches were active, despite the unchanged expected activity of 0.5. This high realized rate is sampling variation from only 20 Bernoulli decisions and materially increases realized auxiliary exposure. These forced changes mean the experiment tests the complete faithful batch-128 training regime; it cannot isolate estimator batch size from optimizer-batch/update-count effects.

## General pipeline profiling

The baseline profile used the existing batch-2/accumulation-4 trainer on a short representative ChemFM workload.

| Profile | Step time | Examples/s | Tokens/s | Mean sampled GPU utilization | Peak allocated / reserved |
|---|---:|---:|---:|---:|---:|
| Current baseline | 2.176 s | 3.68 | 436 | 76.2% | 2.43 / 2.62 GB |
| Exact active SIGReg-128 logical step | 103.5 s | 1.24 | 144 | observed 78-100% under sustained load | 2.43 / 2.62 GB |

Baseline component timing over eight microbatches was 1.43 s forward, 3.02 s backward, 0.17 s optimizer/clip/zeroing, 0.017 s data loading/tokenization, and approximately 0.021 s host-to-device/Python overhead. The active SIGReg-128 smoke step spent 13.68 s in the sufficient-statistic pass, 89.34 s in recomputed forward/backward, 0.30 s in optimizer work, and 0.17 s loading data.

The stack already uses BF16, TF32 matmuls, `LlamaSdpaAttention`, `zero_grad(set_to_none=True)`, dynamic per-microbatch padding, and non-reentrant activation checkpointing. PyTorch 2.3's memory-efficient SDPA kernel works on this GPU; the installed Windows build reports that native Flash Attention was not compiled. Disabling checkpointing was rejected: it was slower on the representative workload and raised allocation/reservation to approximately 5.76/6.31 GB, leaving no safe margin on the 6 GB GPU. Static preprocessing and transfers were below 1% and did not justify caching/workers/pinned memory changes. Fused AdamW plus deferred logging initially looked faster, but a controlled identical-activity repeat was 45% slower at identical VRAM; it was rejected as non-reproducible. `torch.compile`, attention-package changes, sequence bucketing, and larger physical batches were not retained because they were unsupported, scientifically non-equivalent, or lacked a measured benefit.

The scientific run took 1,518 seconds (25.3 minutes), including two beam-validation passes, checkpoint I/O, and final diagnostics. Across the 20 training updates, recorded time was 189.0 s for SIGReg statistics, 803.2 s for gradient forward/backward, 2.38 s optimizer work, and 1.24 s data loading. Full-run peak allocation was 5.52 GB during beam validation; pure training retained the 2.43 GB active-step profile.

## Training behavior

| Epoch | Active 128-sample groups | Mean native loss | Mean cosine loss | Mean SIGReg loss | Mean combined JEPA objective | Built-in validation CE |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 7/10 | 1.7990 | 0.4788 | 50.3115 | 3.1268 | 1.1028 |
| 2 | 8/10 | 1.3370 | 0.3342 | 52.2991 | 3.0868 | 1.0579 |

SIGReg did **not** decrease: the first active value was 50.461, the epoch-1 last value was 52.021, and the epoch-2 last value was 52.304. Epps-Pulley multiplies the statistic by sample count, so its absolute scale changed from approximately 0.94 at `N=2` to approximately 50-52 at `N=128`. Keeping the verified 0.05 weighting therefore made SIGReg a large component of the gradient; changing that coefficient after seeing the result would have violated the fixed experiment.

Both built-in epoch validations had zero exact top-1/3/5/10. The legacy selector retained epoch 1 because top-1 tied at zero, but all analyses below explicitly use the requested epoch-2 checkpoint.

## Fixed 256-reaction generation and CE

All rows are the same 256 unique reaction identities used in `USPTO_MIT_SIGREG_K_ABLATION.md`, with one official enumeration and beam 10.

| Epoch-2 checkpoint | Top-1 | Top-3 | Top-5 | Top-10 | Valid-candidate rate | Target-token CE |
|---|---:|---:|---:|---:|---:|---:|
| Native | 0.027344 (7/256) | 0.117188 | 0.175781 | 0.226562 | 0.809375 | 0.236778 |
| Original symmetric k=1 | 0.015625 (4/256) | 0.109375 | 0.160156 | 0.246094 | 0.852344 | 0.238418 |
| Target-stop-gradient k=1 | **0.035156 (9/256)** | **0.152344** | **0.222656** | **0.277344** | 0.780078 | **0.232881** |
| Previous symmetric SIGReg k=0, batch 2 | 0.007812 (2/256) | 0.093750 | 0.136719 | 0.210938 | **0.871094** | 0.247966 |
| **Symmetric SIGReg k=0, batch 128** | **0.000000 (0/256)** | **0.000000** | **0.000000** | **0.003906** | **0.507031** | **1.080978** |

Against native, the new run has zero shared/new-only top-1 successes and seven native-only successes; the difference is -2.73 points (paired bootstrap 95% interval `[-4.69,-0.78]`, McNemar p=0.0156). Correct-product rank improves on zero reactions, worsens on 58, and ties on 198. Native CE is better on every reaction; mean per-reaction `native CE - new CE` is -0.9023 with 95% interval `[-0.9460,-0.8593]` and Wilcoxon p=`9.64e-44`.

The new checkpoint also has zero top-1-only wins against every relevant reference:

| Baseline | Baseline-only top-1 | New-only top-1 | McNemar p |
|---|---:|---:|---:|
| Native | 7 | 0 | 0.0156 |
| Original symmetric k=1 | 4 | 0 | 0.1250 |
| Target-stop-gradient k=1 | 9 | 0 | 0.00391 |
| Previous batch-2 SIGReg k=0 | 2 | 0 | 0.5000 |

CE is worse on all 256 reactions against all four baselines. The one top-10 hit is shared with each comparator rather than a new success.

## Representation geometry

### Source geometry

| Checkpoint/readout | Source variance | Effective rank | Mean-direction energy |
|---|---:|---:|---:|
| Base ChemFM k=1 | 0.027503 | 20.99 | 0.569844 |
| Native k=1 | 0.009297 | 21.77 | 0.873288 |
| Original symmetric k=1 | 0.00004454 | 39.37 | 0.999481 |
| Target-stop-gradient k=1 | 0.00017298 | 41.60 | 0.997808 |
| Previous SIGReg k=0, batch 2 | 0.00017264 | 46.40 | 0.997814 |
| **SIGReg k=0, batch 128** | **0.016730** | **25.15** | **0.771896** |

### Target and pair geometry

| Checkpoint/readout | Target variance | Effective rank | Mean-direction energy | Raw pair margin | Raw retrieval top-1 |
|---|---:|---:|---:|---:|---:|
| Base ChemFM k=1 | 0.022246 | 21.71 | 0.663507 | 0.030049 | 0.378906 |
| Native k=1 | 0.008797 | 3.44 | 0.831499 | 0.003340 | 0.281250 |
| Original symmetric k=1 | 0.00004941 | 37.10 | 0.999425 | 0.000145 | 0.644531 |
| Target-stop-gradient k=1 | 0.00034236 | 32.28 | 0.995663 | 0.000742 | 0.492188 |
| Previous SIGReg k=0, batch 2 | 0.00014086 | 29.02 | 0.998218 | 0.000577 | **0.742188** |
| **SIGReg k=0, batch 128** | **0.015702** | **17.65** | **0.797893** | **0.046768** | 0.500000 |

This is a material geometric change, not a fold-change illusion. Relative to batch-2 k=0, source variance rises 96.9-fold and target variance 111.5-fold. Target variance is now 1.79 times native and 70.6% of base ChemFM; mean-direction energy falls from 99.82% to 79.79%, close to native's 83.15%. Raw margin rises 81-fold and exceeds base ChemFM. Batch-128 SIGReg therefore prevents the diagnosed variance-contraction/common-direction shortcut.

The older checkpoint retains stronger analysis-only residual discrimination: after joint centering/removing two PCs, margin/retrieval are 0.3433/0.8516 for batch 2 versus 0.1346/0.4961 for batch 128. In the new run, pair signal is expressed at healthy raw scale rather than hidden in a tiny residual, but retrieval is not uniformly better.

## Source intervention and decoder coupling

| Readout | Removal raw sensitivity | Replacement raw sensitivity | Unrelated-source raw sensitivity | Unrelated-source decoder CE change |
|---|---:|---:|---:|---:|
| Native k=1 | 0.069450 | 0.053758 | 0.124279 | 0.842276 |
| Previous batch-2 SIGReg k=0 | 0.012815 | 0.000708 | 0.001839 | 0.778729 |
| **Batch-128 SIGReg k=0** | **0.117102** | **0.078266** | **0.221037** | 0.476477 |

Raw k=0 sensitivity is now comparable to or larger than native and rises 9-fold for contributor removal and over 100-fold for replacement/unrelated controls versus batch 2. Thus the source readout is no longer unusually invariant to source chemistry. The normal decoder also responds to interventions, although its CE changes are smaller than native's against a globally degraded model.

This restored sensitivity does not yield beneficial reaction-level coupling. Versus native, raw pair margin has rho=0.031 with CE improvement and rho=0.024 with rank improvement; residual margin has rho=-0.003 and -0.064, respectively. Every bootstrap interval includes zero. There are no new top-1/3/5 successes to associate with pair strength.

## Answer

**Yes, the >=128-sample regime changes the representation behavior fundamentally, but not in the hoped-for complete sense.** It prevents extreme variance contraction and common-direction concentration, restores raw-scale pair margins, and makes the k=0 source state chemically intervention-sensitive. This establishes that the prior physical-batch-two SIGReg estimate was a major limitation for geometry.

**No, SIGReg loss does not meaningfully decrease and the geometric repair does not benefit generation.** The new checkpoint has zero exact top-1/3/5, one shared top-10 hit, 50.7% validity, and target CE 4.57 times native. Pair strength remains unrelated to CE or rank changes. This is a stronger form of the prior mechanism conclusion: healthy-looking JEPA endpoint geometry and source sensitivity are not sufficient for useful autoregressive generation.

The causal interpretation must remain narrow. Batch-128 exactness necessarily reduced AdamW updates 16-fold, changed dropout granularity, produced 15/20 active groups under seed 533, and exposed the sample-count-scaled SIGReg statistic at much larger absolute magnitude. The catastrophic NTP result can therefore arise from some combination of fewer optimizer updates, high realized auxiliary exposure, and a now-dominant fixed SIGReg term; it cannot be attributed uniquely to healthy geometry or estimator batch size. No follow-up is run automatically.

## Evidence

- Training result: `runs/gate4_sigreg_batch128/sigreg-k0-b128-s533.json`
- Requested checkpoint: `runs/gate4_sigreg_batch128/sigreg-k0-b128-s533-checkpoints/epoch_2`
- Generation: `runs/diagnostics/sigreg_k0_batch128_256/sigreg_k0_b128_generation_256.jsonl`
- Per-reaction CE, pair, and interventions: `runs/diagnostics/sigreg_k0_batch128_256/sigreg_k0_b128_diagnostics_256.json`
- Paired comparisons: `runs/diagnostics/sigreg_k0_batch128_256/*_vs_sigreg_k0_b128_summary_256.json`
- Matched geometry: `runs/diagnostics/sigreg_k0_batch128_256/geometry_matched_epoch2_256.json`
