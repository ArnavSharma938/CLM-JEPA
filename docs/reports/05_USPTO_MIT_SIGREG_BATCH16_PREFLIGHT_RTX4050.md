# USPTO-MIT SIGReg batch-16 preflight (RTX 4050)

## Decision

**DO NOT PROCEED with the currently proposed lone two-epoch batch-16 SIGReg run.** The implementation and `lambda=0.01` gradient scale are numerically sound, but exact SIGReg-16 necessarily halves the native optimizer-update cadence, so comparison with the existing batch-8 native checkpoint would be confounded. The 16-update smoke test also provides no evidence that SIGReg is responding: cosine JEPA improves, while SIGReg rises and native NTP is noisy rather than descending. Before a full run is interpretable, its protocol must include an exposure-matched native control using the same 16-example update cadence; the SIGReg condition should retain the fixed configuration below rather than tune against this smoke result.

No full training, checkpoint selection, beam generation, validation evaluation, k-ablation, or follow-up objective experiment was run. Calibration used no optimizer steps. The only optimization was the requested 16-update smoke test, and no smoke checkpoint was saved.

Evidence: [`runs/diagnostics/sigreg_batch16_preflight.json`](../../runs/diagnostics/sigreg_batch16_preflight.json) and [`src/diagnose_sigreg_batch16_rtx4050.py`](../../src/diagnose_sigreg_batch16_rtx4050.py).

## Fixed configuration and coefficient mapping

The preflight used the base ChemFM-1B checkpoint from which the intended run would start, the fixed 1,280-row Gate 4 USPTO-MIT training manifest, seed 533, k=0 source `<eos>`, symmetric cosine JEPA, native NTP, no stop-gradient, physical batches of two, gradient checkpointing, BF16 model computation, AdamW, LR `1e-4`, and the established 50% JEPA-branch dropout. Each SIGReg estimate jointly contained 16 source and 16 target representations.

The [LeJEPA paper](https://arxiv.org/abs/2511.08544) defines its trade-off as

`(1 - lambda_sig) L_cos + lambda_sig L_SIGReg`.

The official [LeJEPA implementation](https://github.com/galilai-group/lejepa) was inspected at commit `c293d291ca87cd4fddee9d3fffe4e914c7272052`. For the requested two-view `lambda_sig=0.01`, this project divides the LeJEPA mixture by `1-lambda_sig` so adding SIGReg cannot silently weaken the already-fixed cosine term:

`L_aux = L_cos + (0.01 / 0.99) L_SIGReg`

`L_total = L_NTP + active * 2 * L_aux`.

The outer coefficient is two because the established activity probability is 0.5 and the intended expected cosine coefficient is one. Thus:

- relative SIGReg coefficient inside the auxiliary bracket: `0.0101010101`;
- applied coefficient on an active update: `0.0202020202`;
- expected coefficient after 50% dropout: `0.0101010101`;
- cosine-JEPA strength is unchanged.

## Exact batch-16 implementation and equivalence

An exact active update uses eight physical chunks of two. Parameters remain fixed while a no-grad first pass accumulates the source and target empirical-characteristic-function cosine/sine sufficient statistics over all 16 reactions. After the global statistic and exact endpoint VJP are prepared, an RNG-replayed second pass recomputes the same chunks with graphs, applies ordinary NTP and symmetric cosine-JEPA gradients, and injects the exact SIGReg representation VJP. There is no detached queue, stale memory bank, average of batch-2 SIGReg losses, or target asymmetry.

The equivalence test used four real reactions as two identically padded ChemFM microbatches. The direct path retained both microbatch graphs and materialized the four-example distribution; the streamed path used the two-pass sufficient-statistic implementation. Keeping physical shapes identical avoids conflating SIGReg exactness with BF16 kernel/padding changes.

| Quantity | Result |
|---|---:|
| Direct SIGReg | 1.6730396748 |
| Streamed SIGReg | 1.6730396748 |
| Loss absolute error | 0 |
| Endpoint maximum absolute error | 0 |
| Applied representation-gradient maximum absolute error | 4.66e-10 |
| Applied representation-gradient relative L2 error | 1.75e-8 |
| Parameter-gradient maximum absolute error | 0 |
| Parameter-gradient relative L2 error | 0 |
| Parameter-gradient cosine | 1.0000000000 |

This passes numerical equivalence. A preliminary comparison that changed the physical shape from batch four to two batches of two was rejected because BF16 ChemFM endpoints changed by 2.6%; it was a batching-kernel confound, not a streaming-SIGReg error.

## Optimizer semantics

The established setup uses physical batch two with four-way accumulation: eight reactions per update, 160 updates per epoch, and 320 updates over two epochs. Exact SIGReg-16 requires all 16 endpoints and their gradients at one parameter snapshot. Stepping NTP after the first eight would make the remaining endpoints stale or mix parameter states; therefore no exact implementation can preserve an eight-example optimizer cadence.

The faithful batch-16 setup consequently uses 16 reactions per update, 80 updates per epoch, and 160 over two epochs. This is a twofold update reduction, compared with the batch-128 experiment's sixteenfold reduction. The smoke scheduler was configured for that prospective 160-update horizon: eight warmup steps followed by cosine decay to `1e-5`. A future causal comparison requires a native control with the same batch-16 optimizer and scheduler semantics.

## No-step gradient calibration

Three deterministic real groups of 16 were evaluated at the base checkpoint without optimizer steps. They contained 2,148, 1,921, and 1,778 active tokens. Reported norms are over all trainable ChemFM/LoRA parameters. `SIGReg applied` includes both the `0.01/0.99` mapping and active outer coefficient two; `auxiliary applied` includes cosine and SIGReg after the outer coefficient.

| Component | Mean loss | Mean gradient norm |
|---|---:|---:|
| Native NTP | 1.766991 | 2.902e15 |
| Cosine JEPA, raw | 0.444010 | 1.124e15 |
| SIGReg, raw | 6.412715 | 1.720e15 |
| SIGReg, applied | — | 3.474e13 |
| Combined auxiliary, applied | — | 2.226e15 |
| Combined total, pre-clip | — | 3.420e15 |

The raw SIGReg scalar is not used to infer optimization strength. Its **applied gradient is only 1.20% of the NTP gradient** at `lambda_sig=0.01`; the applied combined auxiliary is 76.7% of NTP, and the total is 1.179 times NTP. The auxiliary is therefore not SIGReg-dominated as batch 128 with `lambda_sig=0.05` was.

Mean gradient cosine with NTP was `-0.1998` for cosine JEPA, `+0.1367` for SIGReg, and `-0.1998` for their combined auxiliary. The three groups were heterogeneous: combined-auxiliary/NTP alignment ranged from `-0.6683` to `+0.0667`. The extremely large absolute early gradients are finite and also occur in prior ChemFM runs before gradient clipping; relative component scale and post-warmup behavior are the relevant checks here.

## Sixteen-update smoke trajectory

The fixed dropout stream activated JEPA/SIGReg on 11/16 updates (68.75%), above its nominal 50% rate but plausible for this short deterministic sequence. `—` denotes a native-only update. All gradients were finite and the established global norm clipping at one was applied before each optimizer step.

| Update | Active | NTP | Cosine | SIGReg | Total | Pre-clip grad norm | LR |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 1 | no | 1.6290 | — | — | 1.6290 | 1.302e15 | 0 |
| 2 | yes | 1.9644 | 0.5254 | 6.2754 | 3.1420 | 4.574e15 | 1.25e-5 |
| 3 | yes | 2.0882 | 0.3623 | 6.2879 | 2.9398 | 1.434e4 | 2.50e-5 |
| 4 | yes | 2.0648 | 0.4819 | 6.2176 | 3.1543 | 8.640e2 | 3.75e-5 |
| 5 | yes | 2.7483 | 0.5034 | 6.2526 | 3.8814 | 9.480e2 | 5.00e-5 |
| 6 | no | 2.2758 | — | — | 2.2758 | 1.072e3 | 6.25e-5 |
| 7 | no | 1.8345 | — | — | 1.8345 | 3.480e2 | 7.50e-5 |
| 8 | yes | 1.8784 | 0.4565 | 6.3039 | 2.9189 | 1.768e3 | 8.75e-5 |
| 9 | yes | 2.1922 | 0.4644 | 6.3350 | 3.2489 | 8.840e2 | 1.00e-4 |
| 10 | yes | 2.5782 | 0.4302 | 6.3552 | 3.5669 | 4.521e2 | 9.999e-5 |
| 11 | yes | 2.2218 | 0.3716 | 6.2621 | 3.0915 | 1.424e3 | 9.996e-5 |
| 12 | yes | 2.2702 | 0.4497 | 6.3702 | 3.2983 | 4.480e2 | 9.991e-5 |
| 13 | yes | 2.2889 | 0.4033 | 6.3860 | 3.2246 | 6.440e2 | 9.985e-5 |
| 14 | yes | 1.7798 | 0.4214 | 6.4828 | 2.7535 | 3.792e3 | 9.976e-5 |
| 15 | no | 2.2984 | — | — | 2.2984 | 4.700e2 | 9.965e-5 |
| 16 | no | 1.7436 | — | — | 1.7436 | 8.800e2 | 9.953e-5 |

The native first-four mean was `1.9366` and the last-four mean `2.0277` (`+4.70%`; linear slope `+0.00531/update`). Different reactions make this a noisy indicator, and the earlier batch-2 SIGReg run similarly rose `+4.88%` over its first/last four updates. Thus there is no evidence that NTP is uniquely overwhelmed, but there is also no short-run descending NTP signal.

Across active updates, cosine JEPA moved from `0.4565` for the first three to `0.4248` for the last three (`-6.95%`; slope `-0.00675/active update`). It did not approach the prior near-zero shortcut in 16 updates. SIGReg moved in the wrong direction, from `6.2603` to `6.4130` (`+2.44%`; slope `+0.01780/active update`). Because each point uses a different real batch and random slice draw, this short trend is not proof of divergence; it is specifically an absence of the requested evidence that SIGReg responds to optimization.

## Answers to the smoke questions

1. **NTP is not demonstrably overwhelmed, but it is not descending in this short trace.** Its noise and early huge clipped gradients resemble the established setup.
2. **SIGReg does not show a favorable response.** Its active-update trend is positive despite exact gradients and sane weighting.
3. **Cosine JEPA improves without an immediate near-zero shortcut.** At the last active update it remains `0.4214`; SIGReg is not simultaneously improving.
4. **Gradients are finite and the relative scale is sane.** Applied SIGReg is 1.20% of NTP in the no-step calibration, so the batch-128 domination failure is not repeated.
5. **The intended NTP update behavior cannot be preserved exactly.** Exact SIGReg-16 changes effective optimizer batch from 8 to 16 and halves updates, requiring a matched native control.

The numerical implementation therefore passes, while the currently proposed single-run experimental design does not. The recommendation above is based on the unavoidable optimizer-cadence confound and the missing SIGReg response—not on raw SIGReg loss magnitude, outcome-based coefficient tuning, or a claim of classical collapse.
