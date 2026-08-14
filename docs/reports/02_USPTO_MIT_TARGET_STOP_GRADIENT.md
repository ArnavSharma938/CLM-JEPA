# USPTO-MIT target-side stop-gradient experiment

## Result summary

Target-side JEPA stop-gradient partially relaxed the original geometry concentration but did not establish a generation improvement. On 512 frozen reactions, native and target-stop-gradient exact top-1 were 24/512 and 26/512 (difference +0.39 percentage points; 95% bootstrap CI [-1.37, +2.15]; exact McNemar p=0.8318). Target variance increased 12.03-fold relative to symmetric cLM-JEPA but remained 46.3-fold below native. Target-token CE was 7.69% worse than native, and pair strength did not predict CE, rank, or cutoff transitions.

## 32-reaction training-selection panel

## Scope and intervention

This report covers the single targeted seed-533 cLM-JEPA rescue experiment motivated by the earlier USPTO-MIT geometry and decoder-coupling diagnosis. It is intentionally separate from that parent diagnosis.

The frozen-checkpoint 512-reaction evaluation appears later in this report.

The run changed only the JEPA gradient path:

`L = L_native + lambda * [1 - cos(z_source, sg(z_target))]`.

The target model and target LoRA were not frozen. Native next-token loss continued to update the shared ChemFM/LoRA parameters, including parameters used to form the target representation. Only the target argument of the JEPA cosine was detached; JEPA gradients continued through the k=1 `[PRED]` source branch.

The data, four-epoch budget, learning rate 1e-4, effective JEPA weight 1, 50% JEPA-loss dropout, optimizer, scheduler, batch size, seed, and exact-top-1 checkpoint selector were unchanged. Focused tests establish that the detached and symmetric paths have identical forward values, that target-vector JEPA gradient is absent, that source-vector JEPA gradient remains, and that native CE parameter gradients remain additive.

Validation retained ChemFM's five R-SMILES views and beam width/return count 10. Length-sorted left-padded prompt batching changed execution only. On the complete 32-reaction native epoch-3 panel it reproduced every aggregate metric and every reaction-level correct/incorrect outcome at top-1/3/5/10; lower-ranked wrong candidate identities were not bitwise identical. The optimized pass took 317 seconds. The rescue W&B run was `offline-run-20260810_163127-98ya9a6z` and selected epoch 4.

The training-selection evaluation used 32 unique USPTO-MIT validation identities. The later 512-reaction follow-up did not alter checkpoint selection.

## Results

### Normal generative evaluation

| Metric | Native epoch 3 | Symmetric cLM-JEPA epoch 3 | Target-stop-gradient epoch 4 |
|---|---:|---:|---:|
| Exact top-1 (primary) | 0.062500 | 0.062500 | **0.093750** |
| Exact top-3 | **0.250000** | 0.218750 | 0.218750 |
| Exact top-5 | **0.281250** | 0.250000 | **0.281250** |
| Exact top-10 | 0.343750 | 0.375000 | **0.437500** |
| Validity | 1.000000 | 1.000000 | 1.000000 |
| Native target-token CE | 0.219933 | **0.212888** | 0.239664 |

Target stop-gradient adds one exact top-1 success and loses none relative to either control: two reactions are correct under both endpoints, one is target-stop-gradient-only, and 29 are wrong under both. The comparison has one discordant pair and a two-sided exact McNemar p of 1.0; it does not establish a generative difference.

The secondary evidence is mixed: top-10 improves, top-3 does not, and selected-checkpoint target CE is 8.97% worse than native and 12.58% worse than symmetric cLM-JEPA.

### Frozen representation geometry on the same 32 identities

| Checkpoint | Target variance | Effective rank | Mean-direction energy | Raw matched margin | Raw retrieval top-1 |
|---|---:|---:|---:|---:|---:|
| Base ChemFM | 0.021413 | 12.750 | 0.683344 | 0.020759 | 0.312500 |
| Native epoch 3 | 0.014214 | 2.692 | 0.796124 | 0.024626 | 0.343750 |
| Symmetric cLM-JEPA epoch 3 | 0.00003073 | 15.869 | 0.999633 | 0.00009133 | **0.687500** |
| Target-stop-gradient epoch 4 | 0.00035328 | **17.834** | 0.995351 | 0.00071439 | 0.375000 |

The sole intervention increases target variance 11.50-fold, non-mean-direction energy 12.65-fold, and the raw matched margin 7.82-fold relative to symmetric cLM-JEPA. This supports the causal hypothesis that direct target-side JEPA gradients contributed to the easiest common-direction solution. The shortcut is not eliminated: target variance remains 40 times below native and mean-direction energy remains 0.99535.

| Transformation | Symmetric margin / retrieval | Target-stop-gradient margin / retrieval |
|---|---:|---:|
| Raw | 0.000091 / 0.687500 | 0.000714 / 0.375000 |
| Mean centered | 0.200465 / 0.750000 | 0.124864 / 0.562500 |
| Centered, remove PC1 | 0.246897 / 0.718750 | 0.234665 / **0.937500** |
| Centered, remove top 2 PCs | **0.272632** / 0.812500 | 0.258951 / 0.812500 |
| Centered, remove top 4 PCs | 0.242757 / 0.781250 | **0.254633** / 0.812500 |

Strong multidimensional pair information remains. After common-component removal, the target-stop-gradient checkpoint reaches 0.9375 four-way retrieval after removing PC1 and matches or exceeds symmetric retrieval after removing two or four PCs. The intervention therefore trades some raw-cosine retrieval for a materially larger residual scale without destroying pair structure. This is not classical representation collapse.

### Source-content sensitivity

| Checkpoint | Alternate-SMILES sensitivity | Contributor-removal sensitivity | Contributor-replacement sensitivity |
|---|---:|---:|---:|
| Base ChemFM | 0.165700 | 0.202799 | 0.138641 |
| Native epoch 3 | 0.059930 | 0.074244 | 0.056254 |
| Symmetric cLM-JEPA epoch 3 | 0.000106 | 0.000338 | 0.000129 |
| Target-stop-gradient epoch 4 | 0.000756 | 0.002460 | 0.000824 |

Target stop-gradient raises raw contributor-removal sensitivity 7.27-fold and the other source sensitivities roughly six- to seven-fold relative to symmetric cLM-JEPA. Nonetheless, `[PRED]` remains far less source-sensitive than the native pathway. The original decoder-decoupling diagnosis is weakened but not overturned.

## Causal interpretation

The measurements indicate that unrestricted two-sided JEPA optimization contributed to the geometry shortcut. Blocking only target-side JEPA gradient consistently moves variance, non-mean energy, raw margin, and source sensitivity away from the symmetric pathology, while producing the best observed exact top-1 and top-10.

It does not fully solve the problem. Extreme anisotropy remains, native CE worsens, and the top-1 advantage is one example on a 32-reaction selector panel. The experiment therefore provides evidence that two-sided JEPA gradients contributed to the pathology, but not evidence that this intervention has established a reliable generative advantage.

## Evidence and artifacts

- Parent diagnosis: `docs/reports/01_USPTO_MIT_ORIGINAL_COSINE_FAILURE.md`
- Training result: `runs/gate4_rescue/target_sg-s533-batched.json`
- Selected checkpoint: `runs/gate4_rescue/target_sg-s533-batched-checkpoints/epoch_4`
- Full 32-reaction evaluator parity evidence: `runs/gate4_rescue/beam_full_parity_b4.json`
- Four-condition 32-identity geometry: `runs/diagnostics/target_sg_geometry_32.json`
- Current-code 32-identity representation/intervention diagnostics: `runs/diagnostics/target_sg_representation_32.json`
- Implementation: `src/jepa.py`, `src/train.py`, `src/chemfm.py`, and `scripts/geometry_diagnosis.py`
- Focused tests: `tests/test_jepa.py` and `tests/test_chemfm.py`

## Frozen 512-reaction follow-up

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

The result is a **partial geometry change without demonstrated decoder coupling**. The result does not justify describing the earlier failure as collapse, nor does it justify claiming that target-side stop-gradient improves forward generation. A cross-seed replication remains necessary before treating the two net top-1 reactions as reproducible.

## Evidence artifacts

- Paired summary: `runs/diagnostics/target_sg_rescue_512/summary_512.json`
- Target-stop-gradient beam outputs: `runs/diagnostics/target_sg_rescue_512/target_sg_generation_512.jsonl`
- Per-reaction CE, pair margins, and interventions: `runs/diagnostics/target_sg_rescue_512/target_sg_diagnostics_512.json`
- Four-checkpoint geometry: `runs/diagnostics/target_sg_rescue_512/geometry_512.json`
- Cached embeddings: `runs/diagnostics/target_sg_rescue_512/geometry_cache_512/*.pt`
- Frozen native reference panel/output: `runs/diagnostics/decoder_coupling/native_generation_512.jsonl`
- Selected checkpoint: `runs/gate4_rescue/target_sg-s533-batched-checkpoints/epoch_4`
