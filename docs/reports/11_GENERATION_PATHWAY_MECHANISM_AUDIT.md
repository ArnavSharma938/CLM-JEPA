# Generation-pathway mechanism audit

## Measured summary

1. At the final layer, source-to-target-only CKA was `0.128` native and `0.547` cLM-JEPA. Source-to-teacher-forced-product CKA was `0.182` and `0.278`. Full-panel source-to-target retrieval was `47.27%` for cLM-JEPA; retrieval against its token-predicting representation was `3.52%`.
2. Patching cLM activations into native downstream layers changed aggregate target-token CE by `+0.00834`, `+0.00690`, and `+0.03069` after layers 11, 16, and 21. The layer-21 reaction-level bootstrap 95% CI was `[+0.02149,+0.04750]`. Reverse-patch changes were `-0.00722`, `-0.00330`, and `-0.00613`, with all 64-reaction confidence intervals crossing zero.
3. Raw NTP/JEPA gradient cosine was `-0.0223`; their saved-state AdamW update cosine was `+0.8143`. Observed held-out CE changes after NTP-only, JEPA-only, and combined virtual steps were `+0.000982`, `+0.000244`, and `+0.000541`.
4. Hard-four-way retrieval was `76.95%` cLM-JEPA and `42.19%` native. After independent canonicalization and source-component sorting, it was `72.27%` and `45.31%`. Full-256 canonicalized retrieval was `42.58%` and `10.94%`.
5. Across 1,280 existing five-view predictions, the cLM-minus-native changes in top-1 and best-top-3/5/10 Morgan Tanimoto were `-0.00066`, `-0.00911`, `-0.01823`, and `-0.01285`. The top-1 confidence interval crossed zero; the top-3/5/10 intervals did not. Scaffold-match changes are reported below.

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
