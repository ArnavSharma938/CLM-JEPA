# DeepSeek GSM8K LLM-JEPA A6000 reference

## Result in one paragraph

This compute-reduced reference is **not a successful behavioral reproduction** of LLM-JEPA. On the fixed first 300 examples of the official GSM8K test file, the matched two-epoch seed-82 NTP model scored 36/300 (12.00%) and LLM-JEPA scored 28/300 (9.33%), a paired difference of -2.67 percentage points (bootstrap 95% interval [-6.33, +1.00] points; exact McNemar p = 0.229). It therefore cannot serve as the requested successful-control setting in which JEPA behavior improved. Its frozen representations are nevertheless informative: LLM-JEPA does produce common-direction concentration and variance contraction, but target variance is only 1.45-fold below matched NTP, versus the approximately 495-fold symmetric-cLM-JEPA contraction previously measured against native ChemFM. LLM-JEPA's pair structure is also directly usable at raw scale: epoch-2 full-panel retrieval is 67.67% among 300 targets and four-way retrieval is 94.00%. These observations make LoRA alone an implausible explanation for ChemFM's much more extreme geometric shortcut, while leaving task/model interactions and the behaviorally unsuccessful reduced protocol as important limitations.

## Protocol and provenance

The experiment used the official [LLM-JEPA repository](https://github.com/galilai-group/llm-jepa) at commit `ea0017c654ad917066ff32afc88276bea8ca5f7e`. The upstream code supplied GSM8K preprocessing, chat serialization, LoRA targets, predictor-token construction, last-token readout, native causal loss, symmetric cosine JEPA loss, and GSM8K final-answer scoring. The scientific configuration was:

- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`;
- upstream GSM8K train/test JSONL;
- k=1 appended `<|predictor_1|>` source readout and target EOS readout;
- `L = L_NTP + 0.5 L_JEPA`;
- learning rate `8e-5`;
- official LoRA targets, rank 16, alpha 32, dropout 0.1;
- two epochs, fixed published seed 82;
- matched effective batch 128 and identical data order/exposure for NTP and JEPA.

This is not a reproduction of the paper's four-epoch, five-seed result. Prespecified compute reductions were LoRA rank 16, two epochs, one seed, and one A6000. At the user's later request, behavioral evaluation was restricted to the first 300 official test examples. No result was used for checkpoint selection or tuning.

The Thunder instance was one RTX A6000 (49,140 MiB), six vCPUs, 200 GB storage, and prototyping mode. Thunder CLI v2.0.71 exposed no literal no-template option, so its minimal `base` image was the unavoidable provisioning deviation. Dependencies were installed with `uv`. The environment used Python 3.12, PyTorch 2.13.0+cu130, Transformers 4.55.2, PEFT 0.17.0, Datasets 4.0.0, and Accelerate 1.10.1.

## Execution validation and A6000 optimization

The retained execution path did not change tokenization, truncation, maximum length, example order, objective weights, LoRA, optimizer, scheduler, or effective batch.

| Condition | Physical batch | Accumulation | Effective batch | Checkpointing | Train time | Examples/s | Peak training VRAM | Mean active GPU util. |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| NTP | 8 | 16 | 128 | Off | 25.65 min Trainer / 26.13 min process | 9.713 | 27,850 MiB | 95.8% |
| LLM-JEPA | 4 | 32 | 128 | Off | 73.08 min Trainer / 73.60 min process | 3.408 | 40,754 MiB | 96.4% |

NTP processed 4,973 fixed padded tokens/s (approximately 1,888 mean non-padding tokens/s). The three-stream JEPA path processed 5,235 fixed padded stream-tokens/s (approximately 1,308 mean non-padding stream-tokens/s). Swap was absent, CPU offload was disabled, and no CPU/shared-memory paging was observed.

Qwen2 used Transformers' native BF16 SDPA implementation. CUDA reported flash and memory-efficient SDPA backends enabled; no custom `flash-attn` package was installed. Disabling activation checkpointing was measurably faster and fit at the retained batches. Batch 12 NTP without checkpointing used 39.7 GB without improving end-to-end throughput over batch 8; batch 16 checkpointed was slower. JEPA batch 4 without checkpointing was faster than checkpointed batch 8.

The most important speed fix was an exact vectorization of upstream `_last_token_index`. The original Python loop read CUDA mask scalars one at a time and forced hundreds of host synchronizations. At JEPA batch 4, steady microstep time fell from approximately 6.0 seconds to approximately 1.15 seconds. Every extracted index and forward loss matched exactly.

Two apparent optimizations were rejected:

- Dynamic removal of batch-global padding columns changed deterministic NTP loss by 0.04788 and LoRA gradients by 21.97%, even with original absolute position IDs supplied. Fixed upstream 512-token padding was retained.
- The official additive-mask path asserts right padding, whereas the DeepSeek tokenizer uses left padding and upstream explicitly warns about this combination. It was not used.

Multi-example generation was also rejected after exact comparison with upstream greedy decoding: batch 32 differed on 3/4 verification examples and batch 2 differed on 4/4. Batch 1 matched 4/4 exactly and was used. The wrapper delegates serialization and scoring to upstream and persists every generated response. The GSM8K input was deliberately named `gsm8k_test.jsonl`, because upstream selects its final-answer parser from that filename prefix.

### Correctness fixes required for a matched single-GPU run

Two upstream/stack interactions had to be corrected before accepting the full jobs:

1. Upstream passed `finetune_seed` into model setup but did not apply it before pretrained-model LoRA and new-token initialization. Torch was seeded before both operations. Two independent constructions then had the identical parameter fingerprint `f464b281...f9887a`.
2. With Transformers 4.55, the custom `RepresentationTrainer` did not preserve native loss normalization across gradient accumulation. An initial JEPA endpoint consequently summed roughly 32 microbatch gradients; its losses were approximately 25-60 and gradient norms 6-18. It was quarantined and never evaluated. The corrected path passes `num_items_in_batch` into the native Qwen loss and divides the per-example JEPA mean by the accumulation count. A directly materialized batch-of-four loss was 2.8702 versus 2.8660 from four accumulated microbatches; BF16 LoRA gradients differed by 2.94% in relative L2 across the different kernel shapes. At the actual accumulation of 32, a two-update smoke test produced losses 2.562/2.530 and gradient norms 1.49/1.44 rather than the invalid 32-fold scale.

The measured pre-run projections were 25-30 minutes for NTP and 70-80 minutes for JEPA; actual times were 26.1 and 73.6 minutes. A full upstream sequential evaluation projected to roughly 2.5-2.9 hours per 1,319-example checkpoint. The final 300-example panel was a user-directed compute reduction, not a result-dependent selection.

## Behavioral results

| Metric | NTP epoch 2 | LLM-JEPA epoch 2 | JEPA minus NTP |
|---|---:|---:|---:|
| GSM8K exact final-answer accuracy | **0.1200 (36/300)** | 0.0933 (28/300) | -0.0267 |
| Frozen native target-token CE | **1.17240** | 1.17622 | +0.00383 |

Paired generation outcomes were 15 both correct, 21 NTP-only correct, 13 JEPA-only correct, and 251 neither correct. The paired bootstrap 95% interval for `JEPA - NTP` was [-0.0633, 0.0100], and exact two-sided McNemar p was 0.229. The small panel therefore does not establish statistically precise harm, but it provides no evidence of a JEPA gain. Native CE is also 0.33% worse under JEPA.

Overall Trainer loss was 1.03218 for NTP. Corrected LLM-JEPA combined Trainer loss was 1.08436. On the frozen panel, the LLM-JEPA cosine loss was 0.005438. These values use different objectives and should not be compared as if they were the same scalar.

The paper's published multi-seed, four-epoch improvement must not be compared numerically as though this were a reproduction. This two-epoch, one-seed LoRA run is behaviorally unsuccessful and cannot validate a “successful LLM-JEPA control.”

## Representation geometry

The same first 300 GSM8K identities were used at every checkpoint. Source/question representations follow upstream k=1 serialization and use the final `<|predictor_1|>` hidden state. Target/answer representations follow upstream answer-only serialization and use final EOS. Base-model k=1 uses the deterministically initialized new predictor token.

### Raw trajectory

| Checkpoint | Source variance | Target variance | Source mean energy | Target mean energy | Source eff. rank | Target eff. rank | Raw pair margin | Full retrieval | Four-way retrieval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 0.50565 | 0.58627 | 0.90959 | 0.91823 | 58.69 | 33.60 | -0.00043 | 0.0033 | 0.2000 |
| NTP epoch 1 | 0.93015 | 0.15732 | 0.90282 | 0.98249 | 101.65 | 31.73 | 0.00504 | 0.0033 | 0.4733 |
| NTP epoch 2 | 0.95795 | 0.12497 | 0.90698 | 0.98607 | 102.06 | 30.64 | 0.00438 | 0.0033 | 0.5267 |
| LLM-JEPA epoch 1 | 0.21169 | 0.16877 | 0.98353 | 0.98692 | 91.75 | 84.86 | **0.00879** | **0.7867** | **0.9667** |
| LLM-JEPA epoch 2 | 0.10347 | 0.08640 | 0.99239 | 0.99361 | 79.92 | 73.35 | 0.00392 | 0.6767 | 0.9400 |

Mean-direction energy is `||mean(z)||^2 / mean(||z||^2)`. Full retrieval chooses the paired target among all 300 candidates (chance 0.0033); four-way retrieval uses the correct target plus three fixed seed-82 negatives (chance 0.25).

LLM-JEPA clearly contracts from epoch 1 to epoch 2. Against exposure-matched NTP at epoch 2, source variance is 9.26 times lower and target variance is 1.45 times lower. Source energy outside the mean direction falls from 9.30% to 0.761%; target energy outside the mean falls from 1.393% to 0.639%. This is substantial anisotropy, especially in the k=1 source readout, but it is not constant-vector or dimensional collapse: source/target effective ranks remain 79.9/73.4, and paired retrieval is exceptionally strong.

Embedding scale does not vanish. At epoch 2, mean source/target embedding norms are 144.48/144.11 and mean-vector norms are 143.93/143.66, compared with base embedding norms 92.69/104.92. The contraction is across-example variance relative to a growing common vector, not shrinking vectors toward zero.

### Centered and common-component-removed pair geometry

| Checkpoint and transformation | Pair margin | Full retrieval | Four-way retrieval |
|---|---:|---:|---:|
| Base raw | -0.00043 | 0.0033 | 0.2000 |
| Base centered | -0.00111 | 0.0033 | 0.2733 |
| Base centered, remove PC1 | -0.00298 | 0.0133 | 0.1967 |
| Base centered, remove top 2 PCs | 0.01596 | 0.0133 | 0.3133 |
| Base centered, remove top 4 PCs | 0.01185 | 0.0067 | 0.3067 |
| NTP epoch 2 raw | 0.00438 | 0.0033 | 0.5267 |
| NTP epoch 2 centered | 0.02431 | 0.0333 | 0.6433 |
| NTP epoch 2 centered, remove PC1 | 0.13264 | 0.2667 | 0.6233 |
| NTP epoch 2 centered, remove top 2 PCs | 0.13767 | 0.3200 | 0.6233 |
| NTP epoch 2 centered, remove top 4 PCs | 0.13765 | 0.3400 | 0.6400 |
| LLM-JEPA epoch 2 raw | 0.00392 | 0.6767 | 0.9400 |
| LLM-JEPA epoch 2 centered | 0.44796 | 0.6500 | 0.9433 |
| LLM-JEPA epoch 2 centered, remove PC1 | **0.53164** | 0.7700 | 0.9333 |
| LLM-JEPA epoch 2 centered, remove top 2 PCs | 0.50118 | **0.8067** | **0.9700** |
| LLM-JEPA epoch 2 centered, remove top 4 PCs | 0.46894 | 0.7800 | 0.9600 |

Common-component removal reveals an even larger LLM-JEPA margin, but unlike cLM-JEPA it is not needed to recover useful retrieval: raw LLM-JEPA retrieval is already 67.7% over all 300 targets. The drop from epoch-1 to epoch-2 raw margin/retrieval while cosine loss continues toward zero is evidence that further common-direction growth can make the raw cosine objective less discriminative even here. However, the residual remains large and directly reflected in raw ranks.

## Comparison with ChemFM

| Endpoint comparison | Target variance contraction vs matched native | Target mean-direction energy | Target effective rank | Raw pair margin | Raw four-way retrieval |
|---|---:|---:|---:|---:|---:|
| LLM-JEPA epoch 2 vs NTP epoch 2 | 1.45x | 0.993611 | 73.35 | 0.003916 | 0.9400 |
| Symmetric cLM-JEPA epoch 3 vs native epoch 3 | approximately 495x | 0.999617 | 56.50 | 0.0000937 | 0.7227 |

Both systems become more common-direction dominated under symmetric cosine JEPA, so contraction by itself is not unique to ChemFM. The scale is nevertheless qualitatively different. ChemFM's target variance contraction is about 342 times larger in fold terms, its energy outside the mean direction is about 16.7 times smaller, and its raw pair margin is about 42 times smaller. ChemFM's useful chemistry was primarily exposed by centering/PCA; LLM-JEPA's question-answer pairing is already highly retrievable before any transformation.

This supports, but does not prove, the view that ChemFM found an unusually severe geometric shortcut. It also shifts part of the comparison toward coupling/usability: LLM-JEPA has a large, raw-scale, easily ranked pair signal, whereas cLM-JEPA carries much of its pair signal in a tiny residual and prior decoder-coupling tests found little relationship to generation.

### Is LoRA a plausible cause?

LoRA is not supported as a sufficient cause of the ChemFM pathology. The same rank-16, all-projection LoRA approach here allows high effective rank, raw 300-way retrieval of 67.7%, and only modest target contraction relative to NTP. NTP LoRA alone actually increases source variance relative to base. LoRA could still interact with ChemFM, its readout, or reaction serialization; a cross-model single run cannot rule that out. The evidence does rule against the simple explanation “LoRA necessarily creates the extreme common-direction shortcut.”

## Interpretation and limitation

Among the prespecified possibilities, the closest result is: **the reduced LoRA LLM-JEPA run itself fails behaviorally**. Consequently:

1. It does not establish what geometry looks like in a successful DeepSeek/GSM8K LLM-JEPA run.
2. It cannot show that contraction is compatible with improved GSM8K behavior in this configuration.
3. It does show that symmetric LLM-JEPA can produce moderate variance contraction and strong common-direction growth without losing multidimensional pair information.
4. Compared with ChemFM, its pair information is much larger at raw scale and much more retrievable, making ChemFM's extreme residualization/decoder decoupling a more specific concern than anisotropy alone.

No further seed, epoch, model, ChemFM, SIGReg, VICReg, or VISReg experiment was launched.

## Evidence paths

- Compact run summary: `runs/diagnostics/llm_jepa_reference/run_summary.json`
- Frozen geometry summary: `runs/diagnostics/llm_jepa_reference/diagnostics/geometry_300/geometry_summary.json`
- Per-checkpoint metrics and cached BF16 embeddings: `runs/diagnostics/llm_jepa_reference/diagnostics/geometry_300/`
- Fixed panel identities: `runs/diagnostics/llm_jepa_reference/diagnostics/geometry_300/panel.json`
- Paired behavior summary: `runs/diagnostics/llm_jepa_reference/evaluation/behavioral_summary_300.json`
- Exact per-example outputs: `runs/diagnostics/llm_jepa_reference/evaluation/ntp/results_300.jsonl` and `runs/diagnostics/llm_jepa_reference/evaluation/jepa_600/results_300.jsonl`
- Correct NTP/JEPA logs and GPU telemetry: `runs/diagnostics/llm_jepa_reference/{ntp,jepa}/`
- Quarantined accumulation-bug log: `runs/diagnostics/llm_jepa_reference/jepa_invalid_unscaled_ga/train.log`
- Upstream patch, environment lock, hashes, and GPU provenance: `runs/diagnostics/llm_jepa_reference/`
- Important endpoint and epoch-1/epoch-2 checkpoints: `runs/checkpoints/llm_jepa_reference/important_checkpoints.tar`
- Checkpoint SHA-256 manifest: `runs/checkpoints/llm_jepa_reference/important_checkpoints.tar.sha256`

Both local tar bundles were verified against the remote SHA-256 manifests before the Thunder instance was deleted. `tnr status` was empty after deletion.
