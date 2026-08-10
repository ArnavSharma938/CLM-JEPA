# USPTO-MIT target-side stop-gradient rescue: 512-reaction panel

## Scope and protocol

This frozen-checkpoint follow-up evaluates the selected seed-533 target-side-stop-gradient checkpoint on the same 512 unique canonical USPTO-MIT validation identities used by the earlier decoder-coupling panel. It does not retrain, tune, select a new checkpoint, change the objective, or use augmented SMILES views. The compared endpoints are:

- native seed-533 epoch 3;
- symmetric cLM-JEPA seed-533 epoch 3;
- target-side-stop-gradient cLM-JEPA seed-533 epoch 4.

Generation uses one official enumeration per identity, beam width 10, ten returned sequences, and exact top-1 as the primary metric. The target-stop-gradient pass used equal-prompt-length batches of up to four. The complete 32-reaction parity test had already shown that this execution path preserves every aggregate metric and every reaction-level top-1/3/5/10 outcome, although lower-ranked incorrect candidate identities are not bitwise identical to batch-one generation. The cached native and symmetric 512 results use the earlier batch-one protocol. Accordingly, the prespecified exact cutoff outcomes are authoritative; comparisons of arbitrary incorrect candidate identity ordering are not.

The target-stop-gradient generation required 934.7 reaction-seconds and used batches of four for 432/512 reactions. Frozen CE, representation, and intervention inference required 100.5 seconds; the four-checkpoint geometry pass required 69.7 seconds.

## Generative results

| Metric | Native epoch 3 | Symmetric cLM-JEPA epoch 3 | Target-stop-gradient epoch 4 |
|---|---:|---:|---:|
| Exact top-1 (primary) | 0.046875 (24/512) | 0.033203 (17/512) | **0.050781 (26/512)** |
| Exact top-3 | 0.140625 | 0.136719 | **0.146484** |
| Exact top-5 | 0.205078 | 0.203125 | **0.207031** |
| Exact top-10 | 0.265625 | 0.265625 | **0.283203** |
| Valid-candidate rate | 0.835352 | **0.871680** | 0.811328 |
| Target-token CE | 0.239942 | **0.238685** | 0.258398 |

Against native, 14 reactions are top-1 correct under both checkpoints, 10 are native-only, 12 are target-stop-gradient-only, and 476 are wrong under both. The net top-1 difference is +0.003906, with a paired bootstrap 95% interval of [-0.013672, 0.021484] and exact McNemar p = 0.8318. Correct-product rank improves on 71 reactions, worsens on 68, and ties on 373; mean rank improvement is 0.0996 with 95% interval [-0.1113, 0.3086].

The 32-reaction top-1 direction therefore persists but shrinks to two net reactions on the larger panel. It is not a statistically established generative advantage. Top-3/5/10 move in the same favorable direction, but the intervals and paired changes remain compatible with no effect.

Teacher-forced evidence is unfavorable. Target-stop-gradient CE is 7.69% worse than native and 8.26% worse than symmetric cLM-JEPA. Native CE is lower on 61.72% of reactions; mean per-reaction `native CE - target-stop-gradient CE` is -0.02098 with bootstrap 95% interval [-0.02811, -0.01450] and paired Wilcoxon p = 9.93e-10. Thus the small beam advantage is not accompanied by more probable true targets under teacher forcing.

## Representation geometry on the same identities

| Checkpoint | Target variance | Effective rank | Mean-direction energy | Raw pair margin | Raw retrieval top-1 |
|---|---:|---:|---:|---:|---:|
| Base ChemFM | 0.022994 | 23.948 | 0.651772 | 0.028781 | 0.343750 |
| Native epoch 3 | 0.018809 | 2.625 | 0.719319 | 0.014422 | 0.240234 |
| Symmetric cLM-JEPA epoch 3 | 0.00003376 | **44.824** | 0.999596 | 0.00009292 | **0.654297** |
| Target-stop-gradient epoch 4 | **0.00040629** | 38.749 | **0.994657** | **0.00074389** | 0.414062 |

Relative to symmetric cLM-JEPA, target stop-gradient increases variance 12.03-fold, energy outside the mean direction 13.23-fold, and raw true-versus-matched margin 8.01-fold. These large-sample results reproduce the directional 32-identity geometry rescue and support target-side JEPA gradients as one contributor to the shortcut.

The intervention remains incomplete. Target variance is still 46.3 times below native and 99.47% of target-state energy remains in the mean direction.

| Analysis-only representation | Symmetric margin / retrieval | Target-stop-gradient margin / retrieval |
|---|---:|---:|
| Raw | 0.000093 / 0.654297 | 0.000744 / 0.414062 |
| Mean centered | **0.214653** / **0.765625** | 0.125369 / 0.593750 |
| Centered, remove PC1 | **0.258458** / **0.791016** | 0.225660 / 0.755859 |
| Centered, remove top 2 PCs | **0.280830** / **0.806641** | 0.253003 / 0.804688 |
| Centered, remove top 4 PCs | **0.283874** / 0.853516 | 0.270988 / **0.876953** |

Strong multidimensional pair-specific information survives in both JEPA checkpoints. Target stop-gradient has lower centered margins but essentially matches symmetric retrieval after removing two PCs and exceeds it after removing four. This remains extreme common-direction concentration with informative residual structure, not classical representation collapse.

## Decoder coupling and source interventions

Pair strength does not robustly identify reactions helped by the target-stop-gradient checkpoint:

| Signal | Outcome | Spearman rho | Bootstrap 95% interval | p |
|---|---|---:|---:|---:|
| Raw pair margin | CE improvement | 0.0846 | [-0.0059, 0.1696] | 0.0557 |
| Raw pair margin | Rank improvement | 0.0678 | [-0.0180, 0.1520] | 0.1252 |
| Residual top-2-PC margin | CE improvement | -0.0243 | [-0.1116, 0.0634] | 0.5835 |
| Residual top-2-PC margin | Rank improvement | 0.0190 | [-0.0737, 0.1076] | 0.6680 |

Associations with native-failure-to-target-stop-gradient-success transitions are also null at top-1/3/5/10: every raw and residual bootstrap interval includes zero, with absolute rho no larger than 0.0717. The raw CE association is borderline but does not survive its uncertainty interval and is contradicted by the residual signal and the strongly worse aggregate/per-reaction CE.

| Checkpoint | Intervention | Raw `[PRED]` sensitivity | Residual sensitivity | Decoder CE change | Decoder KL |
|---|---|---:|---:|---:|---:|
| Native | contributor removal | 0.075167 | 0.480790 | 0.552835 | 0.598464 |
| Symmetric cLM-JEPA | contributor removal | 0.000337 | 0.462210 | 0.554123 | 0.598441 |
| Target-stop-gradient | contributor removal | 0.002405 | 0.551795 | 0.564621 | 0.648620 |
| Native | contributor replacement | 0.073408 | 0.519381 | 0.783876 | 0.818750 |
| Symmetric cLM-JEPA | contributor replacement | 0.000111 | 0.399096 | 0.759526 | 0.791972 |
| Target-stop-gradient | contributor replacement | 0.000736 | 0.380460 | 0.821400 | 0.897617 |
| Native | unrelated source | 0.128466 | 0.962550 | 0.950704 | 1.014829 |
| Symmetric cLM-JEPA | unrelated source | 0.000271 | 0.910555 | 0.922140 | 0.983426 |
| Target-stop-gradient | unrelated source | 0.001767 | 0.884877 | 1.031032 | 1.139998 |

Target stop-gradient raises raw `[PRED]` sensitivity roughly 6.5-7.1-fold over symmetric cLM-JEPA, confirming a partial rescue. Yet raw sensitivity remains approximately 31-100 times below native while the normal decoder responds at least as strongly as native to the same chemically meaningful interventions. The chemistry remains visible in the centered residual, but the raw auxiliary readout remains largely decoupled from the normal autoregressive pathway.

## Conclusion

The 512-reaction panel strengthens the geometry result but does not establish a generative rescue. Blocking target-side JEPA gradients causally relaxes variance contraction/common-direction concentration and preserves strong pair structure. It yields a small favorable exact-generation direction—26 versus 24 native top-1 successes—but that difference is statistically weak, while target-token CE is broadly and significantly worse and pair strength does not predict CE, rank, or cutoff transitions.

The most supported interpretation is therefore a **partial geometry rescue without demonstrated decoder coupling**. The result does not justify describing the earlier failure as collapse, nor does it justify claiming that target-side stop-gradient improves forward generation. A cross-seed replication remains necessary before treating the two net top-1 reactions as reproducible.

## Evidence artifacts

- Paired summary: `runs/diagnostics/target_sg_rescue_512/summary_512.json`
- Target-stop-gradient beam outputs: `runs/diagnostics/target_sg_rescue_512/target_sg_generation_512.jsonl`
- Per-reaction CE, pair margins, and interventions: `runs/diagnostics/target_sg_rescue_512/target_sg_diagnostics_512.json`
- Four-checkpoint geometry: `runs/diagnostics/target_sg_rescue_512/geometry_512.json`
- Cached embeddings: `runs/diagnostics/target_sg_rescue_512/geometry_cache_512/*.pt`
- Frozen native reference panel/output: `runs/diagnostics/decoder_coupling/native_generation_512.jsonl`
- Selected checkpoint: `runs/gate4_rescue/target_sg-s533-batched-checkpoints/epoch_4`
