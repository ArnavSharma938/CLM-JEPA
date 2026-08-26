# SIGReg pair-specificity audit

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
- Audit implementation: `scripts/audit_sigreg_pair_specificity.py`
- Focused tests: `tests/test_sigreg_pair_specificity_audit.py`
