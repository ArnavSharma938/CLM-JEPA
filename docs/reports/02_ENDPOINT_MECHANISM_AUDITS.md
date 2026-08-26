# Endpoint cLM-JEPA frozen mechanism audits

This record consolidates four frozen diagnostic phases that examined the selected direct endpoint MSE+SIGReg method. No phase below is pooled statistically with another; each retains its own checkpoint set, reaction panel, gradient scope, and limitations.

The original one-off audit scripts and bulky replay inputs were removed after
their outputs were consolidated. The compact machine-readable audit JSONs and
the selected native/direct epoch-4 checkpoints remain locally. Historical code
paths named below are provenance references and can be recovered from commit
`61fbc74`; they are not maintained entrypoints.

## Chronology and experimental context

| Part | Repository record date | Relative time | Model/checkpoint context | Data and parameter scope | Training status |
|---|---|---|---|---|---|
| I. Gradient and block-swap audit | 2026-08-14, commit `e85dd86` | First mechanistic audit after selecting the epoch-4 endpoint MSE+SIGReg checkpoint | ChemFM-1B base, matched native epoch 4, and direct endpoint MSE+SIGReg epoch 4 | One fixed 16-reaction gradient panel; 256-reaction block-swap panel; LoRA depth groups plus separate token-I/O swaps | Frozen; no optimizer |
| II. SIGReg pair-specificity audit | 2026-08-16, commit `fd04c0d` | Conducted two repository days after Part I to add temporal checkpoints and fresh SIGReg slice draws | Direct endpoint MSE+SIGReg epochs 1, 2, and 4; native epoch 4 reference | Four disjoint batches of 16, four fresh slice draws per batch; 308 LoRA A/B tensors | Frozen; no optimizer |
| III. Contraction and held-out-NTP directional audit | 2026-08-16, commit `90f2e8c` | Recorded later the same day as Part II, after the PCSF training experiment had produced its own checkpoints | Native epochs 1/2/4, MSE epochs 1/2, PCSF epochs 1/2/4, and direct MSE+SIGReg epochs 1/2/4 | Three train/held-out batch pairs of 16; LoRA A/B only; PCSF appears as a historical comparison trajectory | Frozen; no optimizer |
| IV. Generation-pathway audit | 2026-08-24, commit `a63e251` | Recorded eight days after Part III and after the projection and gradient-interaction experiments | Returned to the matched native epoch-4 and direct endpoint MSE+SIGReg epoch-4 checkpoints; alternative trained conditions were not substituted | 256-reaction pathway/retrieval panels, 64-reaction patching panel, 16-reaction AdamW counterfactual, and existing 1,280-reaction predictions | Frozen; exact virtual optimizer steps only; no persistent update |

The dates above are the first repository commits containing the original reports. All four parts use ChemFM-1B for the reaction model. They differ in checkpoint time, parameter scope, and sample panel. PCSF in Part III is a separately trained historical method, while Parts I, II, and IV principally analyze the direct endpoint MSE+SIGReg checkpoint family.

### Relation to the trained-intervention record

This report contains frozen measurements; [report 03](03_ENDPOINT_INTERVENTION_EXPERIMENTS.md) contains persistent training runs. Part III here was recorded after report 03's PCSF run and reads its checkpoints. Part IV was recorded after the projector and gradient-interaction runs, but deliberately returns to the native and direct endpoint MSE+SIGReg checkpoints rather than evaluating those alternative states. Shared condition names therefore identify references, not pooled samples or a continuous training trajectory.

## Part I — Gradient and LoRA block-swap audit

## Measured summary

At the selected epoch-4 cLM-JEPA checkpoint, the active weighted auxiliary
gradient was `0.212x` the LoRA NTP-gradient norm, cosine `-0.042`, and `99.82%`
orthogonal by the one-dimensional projection decomposition. In layers 17--21,
the auxiliary-versus-early-token NTP cosine was `-0.107`; layers 20 and 21 were
`-0.272` and `-0.213`. SIGReg supplied most of the coefficient-weighted
auxiliary norm at this checkpoint.

Substituting cLM-JEPA layers 17--21 into the native adapter changed aggregate
CE by `+0.015607`; the reaction-level bootstrap 95% CI was
`[+0.014308,+0.017021]`. Substituting native layers 17--21 into the cLM-JEPA
adapter changed CE by `-0.004371`, equal to 55.1% of the full native-to-cLM
aggregate CE gap. Substituting cLM-JEPA layers 12--16 into native changed CE by
`-0.003447`, CI `[-0.005278,-0.001631]`.


## Inputs and implementation

| Item | Frozen input |
|---|---|
| Base | `models/ChemFM-1B`, deterministic seed-533 LoRA wrapper |
| Native | `runs/sigreg_batch16_pilot/matched_b4/native_checkpoints/epoch_4` |
| cLM-JEPA | `runs/mse_ablation/stage1/mse_sigreg_checkpoints/epoch_4` |
| Gradient panel | The same 16 seed-533 USPTO-MIT training examples recorded in `runs/diagnostics/sigreg_gradient_response.json` |
| Swap panel | The same 256 identities and parent-panel R-SMILES views used by the MSE+SIGReg decoder-coupling evaluation |
| Code | Historical `scripts/audit_chemfm_mechanism.py` (recoverable at commit `61fbc74`); maintained `src/chemfm.py`, `src/jepa.py`, and `src/train.py` paths |
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

At the cLM-JEPA endpoint, coefficient-weighted MSE and SIGReg norm ratios were
approximately `0.058` and `0.227`; the combined ratio was `0.212`. The opposed
one-dimensional NTP direction contained `0.18%` of auxiliary energy.

## Source and target MSE branches

| State | Source-only norm / NTP; cosine | Target-only norm / NTP; cosine |
|---|---:|---:|
| Base | `0.128; +0.412` | `0.144; +0.123` |
| Native epoch 4 | `0.089; -0.010` | `0.213; -0.038` |
| MSE+SIGReg epoch 4 | `0.028; -0.026` | `0.034; -0.017` |

Target-side MSE was larger at the native checkpoint. At the trained cLM
checkpoint, source-only and target-only ratios/cosines were `0.028/-0.026` and
`0.034/-0.017`.

## Pair specificity

| State | True-vs-shuffled MSE cosine | `||true-shuffle|| / ||true||` | True-vs-shuffled active-aux cosine | Active-aux difference ratio |
|---|---:|---:|---:|---:|
| Base | 0.724 | 0.803 | 0.444 | 1.137 |
| Native epoch 4 | 0.977 | 0.257 | 0.967 | 0.263 |
| MSE+SIGReg epoch 4 | 0.939 | 1.649 | 0.902 | 0.454 |

At the cLM endpoint, true and shuffled active gradients had cosine `0.902` and
`||g_true|| / ||g_shuffle|| = 0.958`. The pair-specific residual was `0.096x`
the NTP norm, cosine `+0.032` with NTP, and `99.90%` orthogonal energy.

## Autoregressive positions and localization

| State | Active auxiliary vs early NTP | Middle | Late |
|---|---:|---:|---:|
| Base | +0.345 | -0.014 | -0.165 |
| Native epoch 4 | -0.030 | +0.007 | -0.004 |
| MSE+SIGReg epoch 4 | -0.041 | -0.010 | -0.012 |

At the selected cLM checkpoint, active-auxiliary cosine with early, middle, and
late NTP was `-0.041`, `-0.010`, and `-0.012`:

| cLM depth | Active aux / NTP norm | Active aux vs total NTP | Active aux vs early NTP | Raw SIGReg / NTP norm; cosine |
|---|---:|---:|---:|---:|
| 0-5 | 0.172 | +0.011 | +0.060 | `3.315; +0.069` |
| 6-11 | 0.115 | -0.031 | -0.014 | `1.521; -0.005` |
| 12-16 | 0.149 | -0.043 | -0.042 | `1.918; -0.045` |
| **17-21** | **0.445** | **-0.085** | **-0.107** | **`5.566; -0.084`** |

Within layers 17--21, layers 20 and 21 had active-auxiliary/early-NTP cosines
`-0.272` and `-0.213`. Across module families, `down_proj` had cosine `-0.115`
and norm ratio `0.796`; `q_proj` had cosine `-0.110` and norm ratio `0.051`.
Token I/O had cosine `+0.025`.

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

The native-plus-cLM-17--21 hybrid increased per-reaction CE for 247/256
reactions. The native-plus-cLM-12--16 hybrid changed aggregate CE by
`-0.003447`; the reverse cLM-plus-native-12--16 hybrid changed it by
`+0.003643`. Hybrid effects are not additive because each combines blocks from
different trained states.

Token I/O swaps increased CE in both backgrounds: `+0.014331` for native plus cLM token I/O and `+0.018639` relative to cLM for cLM plus native token I/O. Each swap changes token-I/O tensors while retaining the recipient's transformer blocks.

The audit does not reconstruct every gradient encountered during four training
epochs. It measures exact local gradients at three frozen states. Block swaps
are off-trajectory parameter interventions and combine co-adapted blocks from
different checkpoints, so the normalized gap fractions are not additive
attributions.

## Part II — SIGReg pair-specificity audit

## Measured summary

Across all 48 cLM checkpoint/batch/draw measurements (epochs 1, 2, and 4; four batches; four fresh slice draws), an infinitesimal SIGReg descent step increased squared-distance discrimination between the correct product and its length-matched derangement. It also increased cosine pair margin, reaction-center separation, and representation variance in every draw.

The applied SIGReg-to-MSE norm ratio rose from `1.51x` at epoch 1 to `3.43x` at epoch 2 and `4.06x` at epoch 4. The fraction of the full active auxiliary gradient that changed when the correct pairing was replaced by a derangement fell from `0.468` to `0.418` to `0.398`. SIGReg/NTP cosine was negative at all cLM checkpoints (`-0.056`, `-0.047`, `-0.047`).

## Frozen design and validation

| Item | Frozen specification |
|---|---|
| Checkpoints | `runs/mse_ablation/stage1/mse_sigreg_checkpoints/epoch_{1,2,4}`; native reference `runs/sigreg_batch16_pilot/matched_b4/native_checkpoints/epoch_4` |
| Data | 64 rows from the fixed 1,280-row manifest, split into four disjoint batches of 16 |
| Selection | First 64 indices of the seed-533 permutation; manifest SHA-256 `b5900bc7...27c8dba` |
| Pair control | One seed-fixed length-matched derangement per batch; all 64 mappings were non-identity and no source retained its true/equal product |
| Readout | k=0: final source EOS and final target EOS, using the training serialization |
| Objectives | Raw symmetric endpoint MSE; exact N=16 SIGReg; NTP on the same reactions and checkpoint |
| Applied active auxiliary | `2*g_MSE + 0.0808080808*g_SIGReg`; the audit conditions on an active 50%-dropout group |
| SIGReg draws | Four fresh 1,024-slice draws per batch; 16 unique direction hashes, common across checkpoints for paired comparison |
| Gradient scope | 6,307,840 parameters in 308 LoRA A/B tensors |
| Stochasticity | Model remained differentiable with all dropout and attention dropout disabled |
| Updates | No optimizer was constructed; before/after parameter fingerprints matched at every checkpoint |

The pinned official LeJEPA implementation (`c293d291ca87cd4fddee9d3fffe4e914c7272052`) samples and normalizes a new Gaussian direction matrix inside every `forward` and increments its global step. The local `src/jepa.py` uses the same fresh-call behavior through `seed + global_step`. This audit used independent seeds `104729 + 1009*batch + draw`, rather than reusing the fixed projection realization from the earlier mechanistic audit.

### Local execution optimization

The maintained cLM-JEPA call computes native logits plus source/target endpoints. For endpoint VJPs, the audit uses the same serialized source and target rows through the PEFT-injected LlamaModel while omitting the independent native row and unused vocabulary projection. On a deterministic physical-batch-2 comparison with the maintained path:

| Parity check | Result |
|---|---:|
| Endpoint mean absolute difference | `0.000861` |
| Endpoint maximum absolute difference | `0.019531` |
| MSE absolute difference | `1.53e-5` |
| LoRA MSE-gradient cosine | `0.999239` |
| LoRA MSE-gradient relative L2 difference | `0.03947` |

The difference is BF16 batch-shape arithmetic, not a change in serialization or readout. Physical batch 4 allocated 7.83 GB and paged on the 6 GB RTX 4050; physical batch 2 used at most 4.87 GB and reduced a one-block benchmark from 147.7 s to 35.2 s. The complete audit took 956.9 s.

## Geometry trajectory

Values are mean ± sample SD across four reaction batches. Effective rank is bounded by batch size here and is only a within-assay trajectory measure.

| State | Source variance | Target variance | Source mean energy | Target mean energy | Source rank | Target rank | Squared-distance margin | Cosine margin | Euclidean top-1 retrieval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MSE+SIGReg e1 | `.002738±.000202` | `.005814±.003572` | `.9341±.0049` | `.8661±.0820` | `11.38±.29` | `7.64±3.71` | `.002212±.000410` | `.02545±.00540` | `42.2%` |
| MSE+SIGReg e2 | `.001492±.000135` | `.001547±.000197` | `.9690±.0027` | `.9673±.0042` | `10.64±.25` | `9.45±.49` | `.001501±.000210` | `.01570±.00228` | `68.8%` |
| MSE+SIGReg e4 | `.001219±.000116` | `.001326±.000179` | `.9726±.0025` | `.9689±.0042` | `10.65±.43` | `9.48±.61` | `.001374±.000219` | `.01576±.00261` | `71.9%` |
| Native e4 reference | `.001268±.000119` | `.002480±.000516` | `.9839±.0015` | `.9673±.0071` | `10.80±.21` | `7.41±1.30` | `.001078±.000293` | `.00691±.00186` | `28.1%` |

Variance and raw margins decreased from cLM epoch 1 to epoch 4, while retrieval improved. Absolute global scale and pair usability are not interchangeable. SIGReg's positive instantaneous variance gradient also does not imply that SIGReg wins the total optimization trajectory.

## Pair specificity and gradient balance

### True versus shuffled force

| cLM state | MSE true/shuffle cosine | `||g_true-g_shuffle||/||g_true||` | Raw pair residual/NTP | Applied SIGReg/NTP | Applied SIGReg/MSE | Pair-specific fraction of full auxiliary |
|---|---:|---:|---:|---:|---:|---:|
| Epoch 1 | `.918±.023` | `.671±.307` | `.040±.018` | `.184±.073` | `1.51±.59` | `.468±.069` |
| Epoch 2 | `.938±.015` | `1.305±.231` | `.058±.018` | `.302±.062` | `3.43±.53` | `.418±.061` |
| Epoch 4 | `.937±.012` | `1.513±.265` | `.042±.010` | `.223±.032` | `4.06±.53` | `.398±.060` |

The MSE true/shuffle cosine alone would overstate pair blindness. The absolute raw pair-specific residual remained `4.0–5.8%` of NTP, and the active outer coefficient doubles it in the full objective. At the same time, SIGReg became several times larger than applied MSE, so the correct-pair-dependent fraction of the full update declined to about 40%.

### Relation to NTP and pair discrimination

Define:

`M = mean(||s-t_shuffle||²) - mean(||s-t_true||²)`.

For an objective gradient `g`, the reported first-order effect under gradient descent is `ΔM / η = -∇M·g`. Positive values mean an infinitesimal step increases correct-versus-wrong discrimination.

| cLM state | MSE/NTP cosine | SIGReg/NTP cosine | Full/NTP cosine | MSE/`g_M` cosine | SIGReg/`g_M` cosine | Full/`g_M` cosine |
|---|---:|---:|---:|---:|---:|---:|
| Epoch 1 | `.001±.013` | `-.056±.021` | `-.056±.019` | `.390±.239` | `-.306±.084` | `-.071±.062` |
| Epoch 2 | `-.004±.022` | `-.047±.016` | `-.053±.013` | `.809±.045` | `-.367±.093` | `-.139±.108` |
| Epoch 4 | `.001±.031` | `-.047±.020` | `-.050±.016` | `.829±.033` | `-.317±.083` | `-.117±.086` |

MSE is approximately orthogonal to NTP, while SIGReg is mildly opposed to it in every cLM draw. Against `g_M`, the signs reverse: MSE is aligned with `g_M`, so MSE descent reduces discrimination; SIGReg is opposed to `g_M`, so SIGReg descent increases discrimination.

## First-order causal response

Effects are normalized by objective-gradient norm. Values are mean ± SD of four batch-level draw means. Positive means the metric increases under descent. For true-pair distance, positive is worse alignment; for other metrics, positive is the requested direction.

| State/objective | Pair discrimination | True-pair distance | Center separation | Cosine margin | Joint variance | Positive discrimination draws |
|---|---:|---:|---:|---:|---:|---:|
| e1 MSE | `-.00794±.00517` | `-.03467±.01447` | `-.02184±.00332` | `-.08987±.06545` | `-.01651±.00450` | `0/4` |
| e1 SIGReg | `+.00616±.00172` | `+.01721±.00882` | `+.01333±.00351` | `+.09430±.02935` | `+.01002±.00388` | `16/16` |
| e1 full | `+.00144±.00084` | `-.01234±.01381` | `-.00374±.00597` | `+.04447±.01134` | `-.00339±.00500` | `14/16` |
| e2 MSE | `-.01522±.00326` | `-.01442±.00147` | `-.02105±.00304` | `-.15884±.03465` | `-.01342±.00187` | `0/4` |
| e2 SIGReg | `+.00700±.00242` | `+.00607±.00109` | `+.00940±.00261` | `+.08487±.02651` | `+.00604±.00170` | `16/16` |
| e2 full | `+.00275±.00219` | `+.00192±.00123` | `+.00346±.00257` | `+.04161±.02406` | `+.00226±.00169` | `15/16` |
| e4 MSE | `-.01548±.00328` | `-.01235±.00137` | `-.02056±.00317` | `-.17631±.03787` | `-.01253±.00196` | `0/4` |
| e4 SIGReg | `+.00606±.00234` | `+.00437±.00097` | `+.00782±.00249` | `+.08228±.02823` | `+.00480±.00148` | `16/16` |
| e4 full | `+.00235±.00190` | `+.00134±.00096` | `+.00285±.00215` | `+.04086±.02334` | `+.00176±.00128` | `15/16` |

At epochs 2 and 4 the full auxiliary increased both true-pair and wrong-pair distance in most measurements, with a larger increase for wrong-pair distance. NTP had near-zero, mixed-sign effects on `M`.

## Reference-method comparison

- [LeJEPA](https://arxiv.org/abs/2511.08544) combines squared prediction with marginal isotropic-Gaussian regularization, and its [official implementation](https://github.com/galilai-group/lejepa) resamples random slices on every call. In this assay, the SIGReg descent direction increased variance, center separation, and mismatched-pair discrimination.
- [Alignment and uniformity](https://arxiv.org/abs/2005.10242) defines positive-pair alignment and distributional spreading as separate properties. Here, the MSE descent direction reduced absolute true-pair distance and finite-batch correct-versus-wrong margin, while the SIGReg descent direction increased both absolute true-pair distance and the margin.
- [Temporally Centered SIGReg](https://arxiv.org/abs/2607.26924) reports marginal-Gaussianization measurements in a multi-task temporal world model. In the ChemFM assay, SIGReg increased reaction-center separation in `48/48` cLM measurements.
- [Sub-JEPA](https://arxiv.org/abs/2605.09241), [VICReg](https://arxiv.org/abs/2105.04906), [Barlow Twins](https://proceedings.mlr.press/v139/zbontar21a.html), and [Whitening-MSE](https://proceedings.mlr.press/v139/ermolov21a.html) use different marginal or redundancy-reduction objectives; none was evaluated in this audit.

## Summary of measured signs

| Quantity | Measurement |
|---|---|
| Pair discrimination, cosine margin, center separation under SIGReg descent | Positive in `48/48` cLM measurements |
| Pair dependence of the statistic | SIGReg is invariant to target permutation |
| Applied SIGReg/MSE norm ratio | `1.51x → 3.43x → 4.06x` at epochs `1 → 2 → 4` |
| Pair-dependent fraction of full auxiliary | `.468 → .418 → .398` |
| SIGReg/NTP cosine | `-0.056`, `-0.047`, `-0.047` |
| Center separation under SIGReg descent | Increased in every cLM draw |
| MSE descent | Reduced true-pair distance, `M`, and cosine margin in every batch |

## Artifacts

- Machine result: `runs/diagnostics/sigreg_pair_specificity_audit/audit.json`
- Historical audit implementation: `scripts/audit_sigreg_pair_specificity.py` (recoverable at commit `61fbc74`)
- Focused tests: `tests/test_sigreg_pair_specificity_audit.py`

## Part III — Contraction and held-out NTP directional audit

## Measured summary

- On the same fixed batches, native pair-center spread fell from `0.05229` at epoch 1 to `0.03609` at epoch 4 (`-31.0%`). MSE-only was already `15.3%` below time-matched native at epoch 1 and `47.4%` below it at epoch 2.
- The active MSE step contracted pair-center spread in every checkpoint/batch measurement (`33/33`). In contrast, the correctly paired residual `2(g_true-g_shuffle)` increased spread in `33/33` measurements.
- When active, PCSF increased spread in every measurement. At MSE+SIGReg epochs 2 and 4, applied SIGReg more than reversed MSE contraction in all six measurements.
- At MSE+SIGReg epochs 2 and 4, the pair-specific MSE residual reduced held-out NTP loss in `6/6` measurements, while total MSE increased it in `6/6`.
- The applied SIGReg step increased held-out loss in `5/6` epoch-2/4 measurements, including `3/3` at epoch 4. The PCSF step reduced held-out loss in `9/9` low-spread measurements.

## Method

The assay used the existing fixed USPTO-MIT serialization and k=0 source/target EOS readout. Three seed-533 training batches of 16 reactions supplied auxiliary gradients. Each was paired with a different, disjoint batch of 16 reactions from the frozen validation panel for the NTP evaluation gradient. The same batches were used at every state.

| Condition | Frozen epochs |
|---|---|
| Matched native | 1, 2, 4 |
| MSE-only | 1, 2 |
| PCSF | 1, 2, 4 |
| MSE+SIGReg-16 | 1, 2, 4 |

All reported parameter-space directions use every trainable LoRA A/B tensor, matching the established mechanistic audit. Large `modules_to_save` token-I/O tensors were excluded from this focused audit because materializing all objectives for them caused host-memory paging; prior work treated token-I/O separately. The model used non-reentrant checkpointing, exact native+source+target padded-row endpoint computation, dropout disabled, and two independently seeded SIGReg projection draws averaged per measurement. The source/target-only BF16 candidate had `11.9%` LoRA-gradient relative error and was not used.

No optimizer was constructed. Trainable-state fingerprints were unchanged before and after all 11 checkpoints. The run completed 33 checkpoint-batch measurements in `2,020.9 s` on the local RTX 4050.

For an objective gradient `g`, the spread velocity is

```text
v(g) = - grad(sigma_PC) dot g
```

Positive values increase pair-center spread under an infinitesimal descent step. For held-out NTP,

```text
Delta L_eval(g) = - grad(L_NTP,heldout) dot g
```

Negative values improve held-out NTP; positive values worsen it. These are local first-order changes per unit learning rate, not finite-step loss predictions.

Actual active-step coefficients were retained:

```text
MSE-only:       2 MSE
PCSF:           2 MSE + 8.4 PCSF
MSE+SIGReg:     2 MSE + 0.08080808 SIGReg
```

The 50% auxiliary dropout halves their expectation without changing direction. Raw and applied values are retained in the machine artifact.

## Time-matched spread trajectory

`r(t)` is the mean batchwise current spread divided by native spread at the same epoch.

| State | Pair-center sigma | `r(t)` |
|---|---:|---:|
| Native e1 | 0.05229 | 1.000 |
| Native e2 | 0.04213 | 1.000 |
| Native e4 | 0.03609 | 1.000 |
| MSE e1 | 0.04471 | 0.847 |
| MSE e2 | 0.02219 | 0.526 |
| PCSF e1 | 0.04604 | 0.873 |
| PCSF e2 | 0.02583 | 0.611 |
| PCSF e4 | 0.01963 | 0.543 |
| MSE+SIGReg e1 | 0.05651 | 1.078 |
| MSE+SIGReg e2 | 0.03584 | 0.855 |
| MSE+SIGReg e4 | 0.03329 | 0.925 |

The instantaneous native NTP spread direction was mixed at epoch 1 and positive in `3/3` batches at epochs 2 and 4. The MSE direction was negative in every batch.

## Spread-direction measurements

The table reports the mean actual active-step spread velocity. Values are multiplied by `1e3`; signs in parentheses give batches with positive/negative velocity. `g_pair=2(g_true-g_shuffle)` is the applied pair-specific residual. `g_reg` and `g_full` use the regularizer and full objective appropriate to that condition.

| State | `v_NTP` | `v_MSE` | `v_pair` | `v_reg` | `v_full` |
|---|---:|---:|---:|---:|---:|
| Native e1 | -0.641 (1/2) | -22.45 (0/3) | +6.566 (3/0) | - | -22.45 |
| Native e2 | +2.490 (3/0) | -37.48 (0/3) | +5.143 (3/0) | - | -37.48 |
| Native e4 | +3.716 (3/0) | -30.51 (0/3) | +3.995 (3/0) | - | -30.51 |
| MSE e1 | -1.094 (0/3) | -14.94 (0/3) | +0.964 (3/0) | - | -14.94 |
| MSE e2 | +0.041 (2/1) | -1.863 (0/3) | +0.374 (3/0) | - | -1.863 |
| PCSF e1 | +0.150 (1/2) | -5.332 (0/3) | +1.065 (3/0) | 0; floor inactive | -5.332 |
| PCSF e2 | +0.002 (2/1) | -4.385 (0/3) | +0.516 (3/0) | +0.411 (3/0) | -3.973 (0/3) |
| PCSF e4 | +0.512 (3/0) | -1.028 (0/3) | +0.312 (3/0) | +0.732 (3/0) | -0.296 (1/2) |
| MSE+SIGReg e1 | -6.664 (0/3) | -7.730 (0/3) | +4.294 (3/0) | +6.566 (3/0) | -1.164 (1/2) |
| MSE+SIGReg e2 | +0.595 (2/1) | -5.078 (0/3) | +7.282 (3/0) | +8.495 (3/0) | +3.417 (3/0) |
| MSE+SIGReg e4 | +0.123 (1/2) | -4.498 (0/3) | +7.484 (3/0) | +7.482 (3/0) | +2.984 (3/0) |

The raw MSE gradient cosine with `grad(sigma_PC)` ranged from `+0.728` to `+0.907` across trained states. The pair-specific residual cosine ranged from `-0.417` to `-0.970`, with positive spread velocity in all 33 measurements.

PCSF is anti-aligned with the spread gradient while its hinge is active. It offset about `9%` of MSE's negative spread velocity at PCSF epoch 2 and `71%` at epoch 4. At the MSE-only epoch-2 state, the calibrated PCSF term offset about `42%`.

At MSE+SIGReg epochs 2 and 4, the applied SIGReg positive spread velocity was `1.67x` and `1.66x` the magnitude of the MSE negative spread velocity. The full auxiliary spread velocity was positive in `6/6` measurements.

## Held-out NTP directional measurements

The table reports mean held-out `Delta L_NTP` per unit learning rate, multiplied by `1e3`, for the most informative trained states. Negative is favorable. Parentheses again give positive/negative batch counts, so `(3/0)` is consistently harmful and `(0/3)` consistently helpful.

| State | MSE | Pair-specific residual | Regularizer | Full auxiliary |
|---|---:|---:|---:|---:|
| MSE e2 | +0.144 (2/1) | -0.115 (0/3) | - | +0.144 (2/1) |
| PCSF e2 | +0.108 (2/1) | -0.084 (0/3) | -0.038 (0/3) | +0.071 (2/1) |
| PCSF e4 | -0.019 (2/1) | -0.016 (2/1) | -0.046 (0/3) | -0.065 (1/2) |
| MSE+SIGReg e2 | +0.469 (3/0) | -0.527 (0/3) | +0.409 (2/1) | +0.877 (2/1) |
| MSE+SIGReg e4 | +0.209 (3/0) | -0.474 (0/3) | +1.068 (3/0) | +1.277 (3/0) |

The measured signs across all states were:

1. At MSE-only epochs 1/2, the pair-specific residual improved held-out NTP in `6/6` measurements. It also improved held-out NTP in `6/6` MSE+SIGReg epoch-2/4 measurements.
2. Total MSE increased held-out loss in `6/6` MSE+SIGReg epoch-2/4 measurements and had mixed signs elsewhere.
3. PCSF's applied step reduced held-out loss in every low-spread measurement where its hinge was active (`9/9`: MSE e2 plus PCSF e2/e4).
4. The MSE+SIGReg full objective increased held-out NTP loss in `5/6` epoch-2/4 measurements and in `3/3` at epoch 4.

## Recorded directional comparisons

- Native spread decreased `31.0%` from epoch 1 to epoch 4. MSE spread was `15.3%` and `47.4%` below time-matched native at epochs 1 and 2.
- MSE spread velocity was negative in `33/33` measurements; pair-residual velocity was positive in `33/33`.
- PCSF and SIGReg spread velocities were positive wherever reported active. The combined PCSF objective retained negative mean velocity at epochs 2 and 4; the combined SIGReg objective had positive velocity in `6/6` epoch-2/4 measurements.
- At MSE+SIGReg epochs 2 and 4, pair-residual held-out NTP change was negative in `6/6`, total MSE change was positive in `6/6`, and full-objective change was positive in `5/6`.

## Limits and artifacts

- This is an exact local first-order assay at frozen states, not a reconstruction of the full historical optimizer trajectory.
- Three independent train/held-out batch pairs establish sign consistency but do not estimate a population effect precisely.
- The result concerns LoRA A/B update space. It does not supersede the prior token-I/O swap evidence.

Artifacts:

- Machine results: `runs/diagnostics/contraction_ntp_directional_audit/audit.json`
- Compact extraction: `runs/diagnostics/contraction_ntp_directional_audit/primary_tables.txt`
- Execution logs: `runs/diagnostics/contraction_ntp_directional_audit/run.stdout.log`, `run.stderr.log`
- Historical implementation: `scripts/audit_contraction_ntp_direction.py` (recoverable at commit `61fbc74`)
- Focused tests: `tests/test_contraction_ntp_direction_audit.py`

## Part IV — Generation-pathway audit

## Measured summary

1. At the final layer, source-to-target-only CKA was `0.128` native and `0.547` cLM-JEPA. Source-to-teacher-forced-product CKA was `0.182` and `0.278`. Full-panel source-to-target retrieval was `47.27%` for cLM-JEPA; retrieval against its token-predicting representation was `3.52%`.
2. Patching cLM activations into native downstream layers changed aggregate target-token CE by `+0.00834`, `+0.00690`, and `+0.03069` after layers 11, 16, and 21. The layer-21 reaction-level bootstrap 95% CI was `[+0.02149,+0.04750]`. Reverse-patch changes were `-0.00722`, `-0.00330`, and `-0.00613`, with all 64-reaction confidence intervals crossing zero.
3. Raw NTP/JEPA gradient cosine was `-0.0223`; their saved-state AdamW update cosine was `+0.8143`. Observed held-out CE changes after NTP-only, JEPA-only, and combined virtual steps were `+0.000982`, `+0.000244`, and `+0.000541`.
4. Hard-four-way retrieval was `76.95%` cLM-JEPA and `42.19%` native. After independent canonicalization and source-component sorting, it was `72.27%` and `45.31%`. Full-256 canonicalized retrieval was `42.58%` and `10.94%`.
5. Across 1,280 existing five-view predictions, the cLM-minus-native changes in top-1 and best-top-3/5/10 Morgan Tanimoto were `-0.00066`, `-0.00911`, `-0.01823`, and `-0.01285`. The top-1 confidence interval crossed zero; the top-3/5/10 intervals did not. Scaffold-match changes are reported below.

## Scope and controls

This was a frozen, local diagnostic audit. It used the existing epoch-4 native checkpoint at `runs/sigreg_batch16_pilot/matched_b4/native_checkpoints/epoch_4`, the direct MSE+SIGReg checkpoint at `runs/mse_ablation/stage1/mse_sigreg_checkpoints/epoch_4`, their existing AdamW state, frozen manifests, and existing official prediction artifacts. No model was trained and no cloud service was used. All model diagnostics ran on the local RTX 4050.

The historical implementation is `scripts/audit_generation_mechanism.py`, with focused semantic tests formerly in `tests/test_generation_mechanism_audit.py`; both are recoverable at commit `61fbc74`. Compact machine-readable outputs remain under `runs/diagnostics/generation_mechanism/`.

## 1. Pathway representation comparison

### Method

For every transformer state—embedding output `0` and block outputs `1` through `22`—the audit collected:

- the source-only JEPA EOS state;
- the target-only JEPA EOS state;
- the same source EOS from the corresponding native teacher-forced row;
- the actual autoregressive prompt state at `<prostart>`;
- the mean hidden state at positions that predict target tokens or target EOS under the causal label shift.

The source-only state and the same teacher-forced source-prefix state matched exactly at every non-degenerate layer (`CKA=1`, mean L2 `0`). This is an internal check that the causal representation extraction did not accidentally compare different source computations. Layer 0 is a shared EOS-token embedding and is not interpreted.

The comparison used only the requested tools: centered linear CKA; ridge source-to-target prediction with a deterministic 204/52 identity split and `alpha=1`; and cosine pair retrieval over all 256 targets. The autoregressive product representation is deliberately based on token-prediction positions, not the final teacher-forced EOS, so it measures the states actually consumed by the LM head while scoring the product.

### Results

| Layer / relationship | Native | cLM-JEPA |
|---|---:|---:|
| 16 source vs target-only CKA | 0.231 | 0.413 |
| 16 source vs AR-product CKA | 0.595 | 0.627 |
| 16 target-only vs AR-product CKA | 0.397 | 0.627 |
| 16 source-to-target retrieval | 25.78% | 65.23% |
| 16 source-to-AR-product retrieval | 4.30% | 1.95% |
| 22 source vs target-only CKA | 0.128 | 0.547 |
| 22 source vs AR-product CKA | 0.182 | 0.278 |
| 22 target-only vs AR-product CKA | 0.142 | 0.335 |
| 22 held-out source-to-target explained variance | 0.049 | 0.342 |
| 22 held-out source-to-AR-product explained variance | 0.426 | 0.444 |
| 22 source-to-target retrieval | 10.55% | 47.27% |
| 22 source-to-AR-product retrieval | 0.39% | 3.52% |

At layer 16, the cLM-minus-native retrieval difference was `+39.45 pp` for source-to-target and `-2.35 pp` for source-to-AR-product. At layer 22, the differences were `+36.72 pp` and `+3.13 pp`.

## 2. Frozen activation patching

### Method

The first 64 reactions of the frozen 256 panel were teacher-forced under both checkpoints. At boundaries after layers 11, 16, and 21, the complete residual stream from one checkpoint was substituted into the other checkpoint, whose remaining downstream layers, final norm, and LM head then produced the logits. Positive CE change is harmful relative to the recipient checkpoint. Dropout was disabled and checkpoint parameter fingerprints were verified unchanged.

This tests checkpoint states, not individual JEPA directions. It can establish whether the complete learned intermediate state is useful to the other checkpoint's downstream computation; it cannot prove that no useful subspace is embedded inside a harmful state.

### Results

| Intervention | Aggregate target CE change | Mean reaction change, 95% bootstrap CI |
|---|---:|---:|
| native + cLM state after 11 | +0.00834 | +0.00743 `[-0.00586,+0.02025]` |
| native + cLM state after 16 | +0.00690 | +0.00643 `[-0.00643,+0.01888]` |
| native + cLM state after 21 | **+0.03069** | **+0.03431 `[+0.02149,+0.04750]`** |
| cLM + native state after 11 | -0.00722 | -0.00477 `[-0.01775,+0.00907]` |
| cLM + native state after 16 | -0.00330 | -0.00211 `[-0.01474,+0.01117]` |
| cLM + native state after 21 | -0.00613 | -0.00540 `[-0.01807,+0.00793]` |

The baseline CE was `0.243819` for native and `0.253236` for cLM on this panel. The native-recipient CE change increased from `+0.00690` after layer 16 to `+0.03069` after layer 21.

Because that initial test identified a boundary effect, the only resolution increase was a position split at layer 16. Patching cLM context positions into native changed CE by only `+0.00071`; patching target-prediction positions changed it by `+0.00635`. In the reverse direction the changes were `-0.00017` and `-0.00281`. No further layer or token sweep was run.

## 3. Exact AdamW one-step counterfactual

### Method

The audit loaded the epoch-4 cLM adapter and exact saved optimizer state at global step 320 (`lr=1e-5`, betas `0.9/0.999`, epsilon `1e-8`, weight decay `0.01`, max gradient norm `1`). It reconstructed one deterministic logical training batch of 16 in physical chunks of 2, including the active objective `2 * (MSE + 4*0.01/0.99 * SIGReg)` and the saved SIGReg step. A disjoint deterministic held-out batch of 16 supplied both the held-out NTP gradient and immediate evaluation endpoint.

For NTP only, JEPA only, and their sum, parameters and optimizer state were restored independently, one exact AdamW step was applied virtually, and held-out NTP was evaluated. The initial adapter fingerprint was restored exactly afterward. The first-order prediction is the held-out gradient dotted with the actual AdamW parameter displacement, so it incorporates adaptive moments, clipping, and weight decay in the proposed step direction.

### Results

- Raw train-gradient NTP-versus-JEPA cosine: `-0.02228` over all trainable parameters (`-0.03682` over LoRA only).
- Raw JEPA-gradient versus held-out-NTP-gradient cosine: `+0.05932`.
- AdamW NTP-update versus JEPA-update cosine: `+0.81434`.
- Combined update versus the sum of separately computed updates: cosine `+0.96905`, with norm ratio `0.579`; AdamW's adaptive map is not additive.

| Virtual update | First-order predicted held-out CE change | Observed held-out CE change |
|---|---:|---:|
| NTP only | +0.000258 | +0.000982 |
| JEPA only | +0.000127 | +0.000244 |
| NTP + JEPA | +0.000283 | +0.000541 |

The raw train-gradient cosine and AdamW-update cosine had opposite signs. Observed changes exceeded first-order predictions for all three virtual updates. The combined observed change was `0.000441` below the NTP-only observed change. Per the prespecified staged rule, MSE/SIGReg decomposition and extra step sizes were not run.

The counterfactual used 16 reactions and reports one frozen state and batch; it is not a population estimate.

## 4. Shortcut-controlled pair retrieval

For every query, the hard four-way task used its true target plus three wrong products chosen to be high-Morgan-similarity among size-matched candidates. Across 768 wrong pairs, `98.96%` satisfied the size criterion; mean heavy-atom, tokenizer-length, and character-length differences were `1.92`, `3.30`, and `3.39`. Mean Morgan Tanimoto was `0.1915`; `5.60%` shared the exact nonempty Bemis-Murcko scaffold.

| Serialization / retrieval set | Native | cLM-JEPA | cLM advantage |
|---|---:|---:|---:|
| Aligned R-SMILES, full 256 | 10.55% | 47.27% | +36.72 pp |
| Aligned R-SMILES, hard four-way | 42.19% | 76.95% | +34.77 pp |
| Independent canonical SMILES, full 256 | 10.94% | 42.58% | +31.64 pp |
| Independent canonical SMILES, hard four-way | 45.31% | 72.27% | +26.95 pp |

Independent canonicalization changed the cLM hard-four-way retrieval from `76.95%` to `72.27%` and native from `42.19%` to `45.31%`. This was a bounded shortcut audit rather than an exhaustive matched-negative framework.

## 5. Chemical proximity of existing generations

All 1,280 existing official five-view prediction records were rescored; no generation was rerun. Similarity used RDKit Morgan fingerprints (`radius=2`, 2,048 bits), and scaffold match used exact isomeric Bemis-Murcko scaffold SMILES with empty scaffolds never counted as matches.

| Endpoint | Native | cLM-JEPA | Paired change, 95% bootstrap CI |
|---|---:|---:|---:|
| Top-1 Tanimoto | 0.47346 | 0.47281 | -0.00066 `[-0.01185,+0.01061]` |
| Best-top-3 Tanimoto | 0.63968 | 0.63057 | -0.00911 `[-0.01723,-0.00091]` |
| Best-top-5 Tanimoto | 0.70068 | 0.68245 | -0.01823 `[-0.02598,-0.01047]` |
| Best-top-10 Tanimoto | 0.76050 | 0.74765 | -0.01285 `[-0.02038,-0.00546]` |
| Top-1 scaffold match | 40.94% | 40.86% | -0.08 pp `[-1.48,+1.33]` |
| Any-top-3 scaffold match | 54.84% | 52.97% | -1.88 pp `[-3.36,-0.47]` |
| Any-top-5 scaffold match | 60.55% | 58.20% | -2.34 pp `[-3.91,-0.78]` |

Among the 1,216 reactions where neither model was exactly correct at top 1, the cLM-minus-native top-1 Tanimoto difference was `+0.00441`, with CI `[-0.00560,+0.01455]`. The top-3/5/10 Tanimoto and top-3/5 scaffold differences were negative with intervals excluding zero.

## Artifacts and verification

- `runs/diagnostics/generation_mechanism/representation_pathway.json`
- `runs/diagnostics/generation_mechanism/activation_patching.json`
- `runs/diagnostics/generation_mechanism/activation_patching_position_refined.json`
- `runs/diagnostics/generation_mechanism/optimizer_counterfactual.json`
- `runs/diagnostics/generation_mechanism/chemical_similarity.json`

The representation run used 256 reactions and took 210.0 seconds. The complete refined activation run used 64 reactions, peaked at 2.03 GB CUDA allocation, and took 59.7 seconds. The exact optimizer audit peaked at 2.71 GB and took 34.7 seconds. All reported model runs identify `NVIDIA GeForce RTX 4050 Laptop GPU` and PyTorch `2.3.0+cu121` in their artifacts.
