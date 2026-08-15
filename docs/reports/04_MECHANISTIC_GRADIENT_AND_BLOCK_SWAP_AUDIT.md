# Mechanistic gradient and LoRA block-swap audit

## Result

The MSE+SIGReg objective does not create a strong model-wide gradient opposition to native product-token prediction. At the selected epoch-4 cLM-JEPA checkpoint, the active weighted auxiliary gradient was `0.212x` the LoRA NTP-gradient norm, cosine `-0.042`, and `99.82%` orthogonal by the one-dimensional projection decomposition. The adverse signal was localized: SIGReg dominated the weighted auxiliary gradient in layers 17-21, where the auxiliary-versus-early-token NTP cosine was `-0.107` (`-0.272` and `-0.213` in layers 20 and 21).

Frozen adapter swaps provide causal support for this localization. Substituting cLM-JEPA layers 17-21 into the native adapter worsened aggregate CE by `+0.015607`; the reaction-level bootstrap 95% CI was `[+0.014308,+0.017021]`. Restoring native layers 17-21 in the cLM-JEPA adapter removed `55.1%` of the full native-to-cLM aggregate CE gap. In contrast, cLM-JEPA layers 12-16 improved the native background by `-0.003447`, CI `[-0.005278,-0.001631]`. The learned auxiliary update is therefore not uniformly harmful across depth.

The most supported mechanism is **localized, mostly SIGReg-driven interference with decoder-facing upper layers, combined with weak alignment between the pair-specific auxiliary residual and NTP**. Broad gradient conflict, target-branch interference, and a simple late-token alignment account are not supported as the primary explanations.

## Inputs and implementation

| Item | Frozen input |
|---|---|
| Base | `models/ChemFM-1B`, deterministic seed-533 LoRA wrapper |
| Native | `runs/sigreg_batch16_pilot/matched_b4/native_checkpoints/epoch_4` |
| cLM-JEPA | `runs/mse_ablation/stage1/mse_sigreg_checkpoints/epoch_4` |
| Gradient panel | The same 16 seed-533 USPTO-MIT training examples recorded in `runs/diagnostics/sigreg_gradient_response.json` |
| Swap panel | The same 256 identities and parent-panel R-SMILES views used by the MSE+SIGReg decoder-coupling evaluation |
| Code | `scripts/audit_chemfm_mechanism.py`; maintained `src/chemfm.py`, `src/jepa.py`, and `src/train.py` paths |
| Machine results | `runs/diagnostics/mse_sigreg_mechanistic_audit/gradient_audit.json` and `block_swap_audit.json` |

The audited active auxiliary gradient was the training objective

`g_aux = 2 g_MSE + 0.08080808 g_SIGReg`,

where `0.08080808 = 2 * (4 * 0.01 / 0.99)`. The expected gradient over the 50% auxiliary dropout is one half of this vector and has the same direction. SIGReg used the validated exact N=16 Epps-Pulley implementation with fixed seed-533 projection directions so checkpoint comparisons were not contaminated by slice resampling.

NTP target labels were split separately for each reaction after the causal shift, including EOS, by `floor(3 * token_rank / target_length)`. MSE branch isolation set the target endpoint VJP to zero for source-only pressure and the source endpoint VJP to zero for target-only pressure. The shuffled control used one deterministic unequal-target, token-length-matched derangement of the same 16 targets.

## Validation

- No optimizer was constructed and no optimizer step was possible.
- Model mode was `train` only for non-reentrant activation checkpointing; all 154 dropout modules and all 22 attention-dropout fields were set to zero.
- Every objective used the same examples, serialization, checkpoint state, and fixed SIGReg directions.
- Source-only plus target-only MSE gradients reproduced the symmetric MSE gradient in focused tests.
- All three trainable-state SHA-256 hashes were identical before and after the gradient audit.
- Every hybrid changed only the listed donor tensors; exact tensor equality was checked for both swapped and background keys.
- Full local endpoints reproduced the prior aggregate values within `8.9e-5`: native `0.240753` versus `0.240683`, cLM-JEPA `0.248691` versus `0.248779`.
- The gradient assay took 160.8 seconds total and peaked at 2.40 GiB allocated VRAM on the local RTX 4050. Each 256-reaction hybrid CE pass took 9.9-12.7 seconds.

### Aggregate-CE denominator correction

The prior decoder-coupling aggregate divides total NLL by the tokenized raw product length, excluding the supervised `<prostart>` and `<eos>` tokens. Its denominator is 10,642, while the model loss contains 11,154 supervised labels: exactly two additional labels for each of 256 reactions. Per-reaction CE was already normalized by all supervised labels, so paired reaction conclusions were unaffected. For continuity, the swap table below reproduces the historical aggregate; the correctly normalized local model-label CEs are native `0.229702` and cLM-JEPA `0.237275`. Both conventions show the same `3.30%` relative degradation locally.

## Global gradient relationships

Values use LoRA A/B parameters only. Norm ratios are raw objective gradient norm divided by NTP-gradient norm; the active-auxiliary row includes its actual training coefficients.

| State | MSE norm / cosine | SIGReg norm / cosine | Active auxiliary norm / cosine | Auxiliary orthogonal energy |
|---|---:|---:|---:|---:|
| Base ChemFM | `0.195 / +0.360` | `2.952 / -0.341` | `0.276 / +0.215` | 95.37% |
| Native epoch 4 | `0.184 / -0.048` | `5.270 / +0.022` | `0.359 / -0.023` | 99.95% |
| MSE+SIGReg epoch 4 | `0.029 / -0.045` | `2.807 / -0.028` | `0.212 / -0.042` | 99.82% |

At the cLM-JEPA endpoint, coefficient-weighted MSE and SIGReg norm ratios were approximately `0.058` and `0.227`; the combined ratio was `0.212`. SIGReg therefore supplied most of the remaining auxiliary force. The slightly negative global dot product exists, but only `0.18%` of auxiliary energy lies in the opposed one-dimensional NTP direction. This rules out strong, uniform gradient opposition as the main account.

## Source and target MSE branches

| State | Source-only norm / NTP; cosine | Target-only norm / NTP; cosine |
|---|---:|---:|
| Base | `0.128; +0.412` | `0.144; +0.123` |
| Native epoch 4 | `0.089; -0.010` | `0.213; -0.038` |
| MSE+SIGReg epoch 4 | `0.028; -0.026` | `0.034; -0.017` |

Target-side MSE was larger at the native checkpoint, but its conflict was weak; at the trained cLM checkpoint both branches were small and nearly orthogonal. The audit therefore weakens target-branch interference as the primary mechanism. This does not claim that symmetric target adaptation is always harmless; it shows that it does not explain this endpoint's dominant adverse signal.

## Pair specificity

| State | True-vs-shuffled MSE cosine | `||true-shuffle|| / ||true||` | True-vs-shuffled active-aux cosine | Active-aux difference ratio |
|---|---:|---:|---:|---:|
| Base | 0.724 | 0.803 | 0.444 | 1.137 |
| Native epoch 4 | 0.977 | 0.257 | 0.967 | 0.263 |
| MSE+SIGReg epoch 4 | 0.939 | 1.649 | 0.902 | 0.454 |

At the cLM endpoint the shuffled MSE gradient was much larger than the already-small true-pair MSE gradient, but pointed in nearly the same direction. Adding the pair-invariant SIGReg component made the true and shuffled active gradients still more similar in scale: cosine `0.902`, `||g_true|| / ||g_shuffle|| = 0.958`. The pair-specific residual was `0.096x` the NTP norm and cosine `+0.032` with NTP (`99.90%` orthogonal energy). Correct pairing therefore changes a nontrivial residual, but most auxiliary direction is preserved without the correct pair and the residual is not aligned with product-token optimization.

## Autoregressive positions and localization

| State | Active auxiliary vs early NTP | Middle | Late |
|---|---:|---:|---:|
| Base | +0.345 | -0.014 | -0.165 |
| Native epoch 4 | -0.030 | +0.007 | -0.004 |
| MSE+SIGReg epoch 4 | -0.041 | -0.010 | -0.012 |

There is no evidence that the endpoint objective is usefully aligned with late-token NTP but not early-token NTP. At the selected cLM checkpoint all three global relationships are weakly negative. The depth-localized result is stronger:

| cLM depth | Active aux / NTP norm | Active aux vs total NTP | Active aux vs early NTP | Raw SIGReg / NTP norm; cosine |
|---|---:|---:|---:|---:|
| 0-5 | 0.172 | +0.011 | +0.060 | `3.315; +0.069` |
| 6-11 | 0.115 | -0.031 | -0.014 | `1.521; -0.005` |
| 12-16 | 0.149 | -0.043 | -0.042 | `1.918; -0.045` |
| **17-21** | **0.445** | **-0.085** | **-0.107** | **`5.566; -0.084`** |

Within layers 17-21, layers 20 and 21 had active-auxiliary/early-NTP cosines `-0.272` and `-0.213`. Across module families at the cLM endpoint, the strongest adverse relationships were `down_proj` (cosine `-0.115`, norm ratio `0.796`) and `q_proj` (cosine `-0.110`, norm ratio `0.051`). Other attention and MLP families were weakly negative. Token I/O had cosine `+0.025`; its swap result below indicates co-adaptation, not a matching local gradient-conflict signature.

## Frozen block-swap causal localization

Native and cLM full aggregate CEs were `0.240753` and `0.248691`; the full gap was `0.007938`. `Gap fraction` is `(hybrid - native) / full_gap`. Because hybrids are off-trajectory combinations, fractions are normalized effects rather than additive variance attribution. `Better/worse` counts compare paired per-reaction CE with full native.

| Hybrid | Aggregate CE | Delta vs native | Delta vs cLM | Gap fraction | Mean paired delta vs native | Better/worse |
|---|---:|---:|---:|---:|---:|---:|
| Native + cLM layers 0-5 | 0.240394 | -0.000359 | -0.008296 | -4.5% | -0.000780 | 137/119 |
| cLM + native layers 0-5 | 0.246895 | +0.006142 | -0.001796 | 77.4% | +0.005755 | 115/141 |
| Native + cLM layers 6-11 | 0.240293 | -0.000460 | -0.008397 | -5.8% | -0.001050 | 133/123 |
| cLM + native layers 6-11 | 0.253085 | +0.012332 | +0.004394 | 155.4% | +0.012561 | 103/153 |
| **Native + cLM layers 12-16** | **0.237306** | **-0.003447** | **-0.011385** | **-43.4%** | **-0.003457** | **148/108** |
| cLM + native layers 12-16 | 0.252334 | +0.011581 | +0.003643 | 145.9% | +0.011293 | 94/162 |
| **Native + cLM layers 17-21** | **0.256360** | **+0.015607** | **+0.007669** | **196.6%** | **+0.017361** | **9/247** |
| **cLM + native layers 17-21** | **0.244320** | **+0.003567** | **-0.004371** | **44.9%** | **+0.002794** | **126/130** |
| Native + cLM token I/O | 0.255083 | +0.014331 | +0.006393 | 180.5% | +0.014777 | 88/168 |
| cLM + native token I/O | 0.267329 | +0.026577 | +0.018639 | 334.8% | +0.028449 | 40/216 |

The layers 17-21 effects are directionally consistent: adding the cLM block is harmful, and removing it from cLM improves CE. The native-background effect is broader (247/256 reactions worsened) and has a narrow bootstrap interval. Layers 12-16 show the opposite effect in the native background. The reverse 12-16 hybrid worsens cLM, demonstrating substantial block co-adaptation and preventing additive interpretation.

Token I/O swaps worsen both backgrounds. This means the two token-I/O states are incompatible with the opposite model context; it does not identify the cLM token-I/O state alone as the cause. The result is retained as evidence of co-adaptation rather than used to select the intervention.

## Unified diagnosis

| Hypothesis | Audit conclusion |
|---|---|
| Broad direct gradient interference | **Weakened.** The trained active auxiliary gradient is mostly orthogonal globally, with only 0.18% opposed energy. |
| SIGReg-specific interference | **Strengthened and localized.** At the cLM endpoint SIGReg dominates weighted auxiliary norm and its strongest adverse relationship is in layers 17-21. |
| Target-branch interference | **Weakened.** Source and target MSE branches are both small and nearly orthogonal at the selected endpoint; target conflict is not dominant. |
| Weak pair specificity | **Strengthened.** True and shuffled active gradients have cosine 0.902; the pair-specific residual is nearly orthogonal to NTP. |
| Endpoint/trajectory mismatch | **Partially strengthened.** There is no late-token alignment; conflict is most visible for early NTP in upper layers. |
| Localized harmful parameter changes | **Strongly strengthened.** Bidirectional layers 17-21 swaps track the adverse gradient localization, while layers 12-16 can improve native CE. |
| No meaningful conflict anywhere | **Ruled out.** Global conflict is weak, but upper-layer gradient and causal swap evidence agree. |

The audit does not reconstruct every gradient encountered during four training epochs. It measures exact local gradients at three frozen states. Block swaps are causal parameter interventions but are off the training trajectory and expose nonlinear co-adaptation. These limitations prevent assigning the full endpoint gap additively to individual blocks; they do not explain away the convergent layers 17-21 gradient and bidirectional-swap result.

## One next experiment

Run exactly one cadence-matched seed-533 MSE+SIGReg replication in which **only SIGReg gradients to LoRA parameters in layers 17-21 are masked**, while NTP and MSE gradients continue through all existing trainable parameters and every other setting remains unchanged. Evaluate the fixed epoch-4 checkpoint against the matched native endpoint using the existing CE panel and exact generation protocol. This intervention isolates the component and depth jointly identified by the audit; it does not alter pairing, MSE, target symmetry, token I/O, data exposure, or regularizer strength.
