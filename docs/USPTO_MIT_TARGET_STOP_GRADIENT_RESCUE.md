# USPTO-MIT cLM-JEPA target-side stop-gradient rescue

## Scope and intervention

This report covers the single targeted seed-533 cLM-JEPA rescue experiment motivated by the earlier USPTO-MIT geometry and decoder-coupling diagnosis. It is intentionally separate from that parent diagnosis.

The subsequent frozen-checkpoint 512-reaction evaluation is reported separately in [`USPTO_MIT_TARGET_STOP_GRADIENT_RESCUE_512.md`](USPTO_MIT_TARGET_STOP_GRADIENT_RESCUE_512.md).

The run changed only the JEPA gradient path:

`L = L_native + lambda * [1 - cos(z_source, sg(z_target))]`.

The target model and target LoRA were not frozen. Native next-token loss continued to update the shared ChemFM/LoRA parameters, including parameters used to form the target representation. Only the target argument of the JEPA cosine was detached; JEPA gradients continued through the k=1 `[PRED]` source branch.

The data, four-epoch budget, learning rate 1e-4, effective JEPA weight 1, 50% JEPA-loss dropout, optimizer, scheduler, batch size, seed, and exact-top-1 checkpoint selector were unchanged. Focused tests establish that the detached and symmetric paths have identical forward values, that target-vector JEPA gradient is absent, that source-vector JEPA gradient remains, and that native CE parameter gradients remain additive.

Validation retained ChemFM's five R-SMILES views and beam width/return count 10. Length-sorted left-padded prompt batching changed execution only. On the complete 32-reaction native epoch-3 panel it reproduced every aggregate metric and every reaction-level correct/incorrect outcome at top-1/3/5/10; lower-ranked wrong candidate identities were not bitwise identical. The optimized pass took 317 seconds. The rescue W&B run was `offline-run-20260810_163127-98ya9a6z` and selected epoch 4.

Only the requested 32 unique USPTO-MIT validation identities were used for the rescue evaluation. No 512- or 1,024-identity rescue analysis was run.

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

Target stop-gradient adds one exact top-1 success and loses none relative to either control: two reactions are correct under both endpoints, one is target-stop-gradient-only, and 29 are wrong under both. This is the first native-beating exact top-1 result in the controlled seed-533 comparison, but it is only one reaction out of 32; two-sided exact McNemar p is 1.0 with one discordant pair. It is therefore a promising directional result, not a statistically established generative improvement.

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

The most supported interpretation is that unrestricted two-sided JEPA optimization was a real contributor to the geometry shortcut. Blocking only target-side JEPA gradient consistently moves variance, non-mean energy, raw margin, and source sensitivity away from the symmetric pathology, while producing the best observed exact top-1 and top-10.

It does not fully solve the problem. Extreme anisotropy remains, native CE worsens, and the top-1 advantage is one example on a 32-reaction selector panel. The experiment therefore provides evidence that two-sided JEPA gradients contributed to the pathology, but not evidence that this intervention has established a reliable generative advantage.

## Exactly one recommended next experiment

Run exactly one seed-917 replication of this unchanged target-stop-gradient configuration. Do not tune or add another intervention. Its sole purpose is to determine whether the directional geometry improvement and native-beating exact top-1 reproduce across the already-prespecified second seed. No such replication was run here.

## Evidence and artifacts

- Parent diagnosis: `docs/USPTO_MIT_GEOMETRY_DIAGNOSIS.md`
- Training result: `runs/gate4_rescue/target_sg-s533-batched.json`
- Selected checkpoint: `runs/gate4_rescue/target_sg-s533-batched-checkpoints/epoch_4`
- Full 32-reaction evaluator parity evidence: `runs/gate4_rescue/beam_full_parity_b4.json`
- Four-condition 32-identity geometry: `runs/diagnostics/target_sg_geometry_32.json`
- Current-code 32-identity representation/intervention diagnostics: `runs/diagnostics/target_sg_representation_32.json`
- Implementation: `src/jepa.py`, `src/train.py`, `src/chemfm.py`, and `src/geometry_diagnosis.py`
- Focused tests: `tests/test_jepa.py` and `tests/test_chemfm.py`
