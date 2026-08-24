# Generation-pathway mechanism audit

## Direct answers

1. **Is JEPA information present in the actual generation computation?** Yes, but only weakly. At the final layer, cLM-JEPA increased source-to-target-only CKA from `0.128` to `0.547`, while source-to-teacher-forced-product CKA increased only from `0.182` to `0.278`. Full-panel source-to-target retrieval reached `47.27%`, but retrieval against the actual token-predicting representation was only `3.52%`. The learned relationship reaches the causal computation in attenuated form; most of its isolated-view structure does not.
2. **If present, is it causally useful?** Not as a complete cLM residual state. Patching cLM activations into native downstream layers never improved aggregate target-token CE at the tested boundaries. The decisive effect was harmful after layer 21: `+0.03069` CE, with a reaction-level bootstrap 95% CI of `[+0.02149,+0.04750]`. Native activations directionally rescued cLM, although the 64-reaction confidence intervals crossed zero. A targeted position refinement showed that the layer-16 effect came almost entirely from target-prediction positions, not source/context positions.
3. **Is damage caused locally by the JEPA update despite weak raw gradient conflict?** Not in the tested exact step. Raw NTP and JEPA gradients had cosine `-0.0223`, but their saved-state AdamW parameter updates had cosine `+0.8143`: optimizer state radically changed the relationship. A JEPA-only virtual AdamW step did worsen held-out CE by `+0.000244`, despite the raw auxiliary gradient pointing weakly toward held-out improvement. However, adding JEPA to NTP reduced the observed one-step worsening from `+0.000982` to `+0.000541`. The audit therefore shows that raw gradients miss optimizer effects, but it does not identify JEPA as the local cause of combined-update damage at this checkpoint and batch.
4. **Is the strong pair signal genuine reaction information or largely a shortcut?** It survives both controls. With three molecularly and length-matched hard wrong products per query, cLM retrieval was `76.95%` versus native `42.19%`. After independently canonicalizing each molecule and sorting source components to break paired R-SMILES alignment, it remained `72.27%` versus `45.31%`. Full-256 retrieval remained `42.58%` versus `10.94%`. The signal is not largely explained by the tested molecular-size or serialization shortcuts.
5. **Does JEPA improve a chemically meaningful aspect of generated products?** No. Across the existing 1,280-reaction five-view predictions, top-1 Morgan Tanimoto changed by `-0.00066` with a 95% CI spanning zero. Best-top-3, top-5, and top-10 Tanimoto were worse by `-0.00911`, `-0.01823`, and `-0.01285`; their intervals excluded zero. Top-1 scaffold match was unchanged, while any-top-3 and any-top-5 scaffold match were worse. Restricting to reactions where neither model was top-1 exact did not reveal a hidden gain.

## Scope and controls

This was a frozen, local diagnostic audit. It used the existing epoch-4 native checkpoint at `runs/sigreg_batch16_pilot/matched_b4/native_checkpoints/epoch_4`, the direct MSE+SIGReg checkpoint at `runs/mse_ablation/stage1/mse_sigreg_checkpoints/epoch_4`, their existing AdamW state, frozen manifests, and existing official prediction artifacts. No model was trained and no cloud service was used. All model diagnostics ran on the local RTX 4050.

The implementation is `scripts/audit_generation_mechanism.py`. Focused semantic tests are in `tests/test_generation_mechanism_audit.py`. Machine-readable outputs are under `runs/diagnostics/generation_mechanism/`.

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

The CKA and linear-prediction increases establish that some learned relation is present in the actual autoregressive path. The retrieval gap establishes the more important qualification: the strong isolated-view correspondence is mostly not expressed as pair-discriminative structure at the token-predicting states. At layer 16, cLM greatly improves isolated source-target retrieval while slightly reducing source-to-AR-product retrieval. This is a view-integration failure, not absence of a learned pair representation.

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

The baseline CE was `0.243819` for native and `0.253236` for cLM on this panel. The one clear causal result is that the cLM final residual state is harmful even when native norm/head components consume it. Harm grows substantially between the layer-16 and layer-21 boundaries, matching report 04's independently obtained parameter-block localization.

Because that initial test identified a boundary effect, the only resolution increase was a position split at layer 16. Patching cLM context positions into native changed CE by only `+0.00071`; patching target-prediction positions changed it by `+0.00635`. In the reverse direction the changes were `-0.00017` and `-0.00281`. No further layer or token sweep was run.

## 3. Exact AdamW one-step counterfactual

### Method

The audit loaded the epoch-4 cLM adapter and exact saved optimizer state at global step 320 (`lr=1e-5`, betas `0.9/0.999`, epsilon `1e-8`, weight decay `0.01`, max gradient norm `1`). It reconstructed one deterministic logical training batch of 16 in physical chunks of 2, including the active objective `2 * (MSE + 4*0.01/0.99 * SIGReg)` and the saved SIGReg step. A disjoint deterministic held-out batch of 16 supplied both the held-out NTP gradient and immediate evaluation endpoint.

For NTP only, JEPA only, and their sum, parameters and optimizer state were restored independently, one exact AdamW step was applied virtually, and held-out NTP was evaluated. The initial adapter fingerprint was restored exactly afterward. The first-order prediction is the held-out gradient dotted with the actual AdamW parameter displacement, so it incorporates adaptive moments, clipping, and weight decay in the proposed step direction.

### Results

- Raw train-gradient NTP-versus-JEPA cosine: `-0.02228` over all trainable parameters (`-0.03682` over LoRA only).
- Raw JEPA-gradient versus held-out-NTP-gradient cosine: `+0.05932`. A plain negative-gradient JEPA step would therefore point weakly toward held-out improvement.
- AdamW NTP-update versus JEPA-update cosine: `+0.81434`.
- Combined update versus the sum of separately computed updates: cosine `+0.96905`, with norm ratio `0.579`; AdamW's adaptive map is not additive.

| Virtual update | First-order predicted held-out CE change | Observed held-out CE change |
|---|---:|---:|
| NTP only | +0.000258 | +0.000982 |
| JEPA only | +0.000127 | +0.000244 |
| NTP + JEPA | +0.000283 | +0.000541 |

Raw-gradient reasoning misses a direction reversal introduced by accumulated AdamW state: the JEPA gradient looks weakly favorable against held-out NTP, but its actual adaptive update is weakly harmful. Curvature/nonlinearity amplifies the predicted harm in all three cases. Crucially, the combined update is less harmful than NTP alone on this frozen sample. This does not support a claim that instantaneous JEPA interference causes the cLM generation gap. Per the prespecified staged rule, MSE/SIGReg decomposition and extra step sizes were not run.

The 16-reaction counterfactual is a mechanism probe, not an estimator of average training benefit. Its reliable conclusion is about the optimizer's transformation of directions, not the population mean CE change.

## 4. Shortcut-controlled pair retrieval

For every query, the hard four-way task used its true target plus three wrong products chosen to be high-Morgan-similarity among size-matched candidates. Across 768 wrong pairs, `98.96%` satisfied the size criterion; mean heavy-atom, tokenizer-length, and character-length differences were `1.92`, `3.30`, and `3.39`. Mean Morgan Tanimoto was `0.1915`; `5.60%` shared the exact nonempty Bemis-Murcko scaffold.

| Serialization / retrieval set | Native | cLM-JEPA | cLM advantage |
|---|---:|---:|---:|
| Aligned R-SMILES, full 256 | 10.55% | 47.27% | +36.72 pp |
| Aligned R-SMILES, hard four-way | 42.19% | 76.95% | +34.77 pp |
| Independent canonical SMILES, full 256 | 10.94% | 42.58% | +31.64 pp |
| Independent canonical SMILES, hard four-way | 45.31% | 72.27% | +26.95 pp |

Breaking alignment causes modest attenuation, not disappearance. This is a deliberately small shortcut audit rather than an exhaustive matched-negative framework, but the remaining advantages are too large to characterize the learned signal as largely serialization or simple molecular-similarity leakage.

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

Among the 1,216 reactions where neither model was exactly correct at top 1, cLM top-1 Tanimoto was only `+0.00441` higher with CI `[-0.00560,+0.01455]`; top-3/5/10 Tanimoto and top-3/5 scaffold coverage remained significantly worse. JEPA does not merely exchange exact rank for chemically closer candidates.

## Smallest remaining mechanism and next experiment

The remaining mechanism is **view-to-decoder integration mismatch with upper-layer co-adaptation**:

1. cLM learns a large, genuine source-product relation in the isolated JEPA views.
2. Only a smaller fraction appears at the causal token-prediction states.
3. The complete cLM state is not useful to native downstream computation, and clear harm accumulates by the final upper-layer boundary specifically at product-prediction positions.
4. The saved optimizer does alter raw-gradient geometry, but the exact combined step does not show JEPA-specific incremental harm.
5. The representation gain produces no exact or chemistry-aware behavioral gain.

This narrows the problem beyond generic collapse or gradient conflict. There is a clear reason for exactly one new full-training experiment when compute is next authorized: retain the validated direct MSE+SIGReg objective and all existing controls, but **zero only the auxiliary-gradient contribution to LoRA blocks 17-21 while leaving their NTP gradients unchanged**. Do not change the loss, weight, cadence, optimizer, or data. This is the smallest intervention jointly motivated by report 04's favorable 12-16/harmful 17-21 parameter swaps and the present final-state activation harm. Its gate must be autoregressive CE/generation, with isolated-view retrieval only as a mechanism check.

No such training was run in this audit. If that single layer-restricted condition fails, the evidence favors abandoning final-endpoint JEPA integration in this architecture rather than another loss-weight, regularizer, projector, or gradient-combiner sweep.

## Artifacts and verification

- `runs/diagnostics/generation_mechanism/representation_pathway.json`
- `runs/diagnostics/generation_mechanism/activation_patching.json`
- `runs/diagnostics/generation_mechanism/activation_patching_position_refined.json`
- `runs/diagnostics/generation_mechanism/optimizer_counterfactual.json`
- `runs/diagnostics/generation_mechanism/chemical_similarity.json`

The representation run used 256 reactions and took 210.0 seconds. The complete refined activation run used 64 reactions, peaked at 2.03 GB CUDA allocation, and took 59.7 seconds. The exact optimizer audit peaked at 2.71 GB and took 34.7 seconds. All reported model runs identify `NVIDIA GeForce RTX 4050 Laptop GPU` and PyTorch `2.3.0+cu121` in their artifacts.
