# USPTO-MIT cLM-JEPA representation geometry diagnosis

**Scope.** Existing checkpoints only; no optimizer steps or retraining. The assay uses the frozen 1,024-identity `data/gate3/uspto_mit_synthesis.csv` sample, seed 533, and the Gate 4/5-faithful k=1 predictor-token initialization. It compares base ChemFM, native epoch 3, and cLM-JEPA epoch 3. Model inference took 150.54 seconds total on the RTX 4050; peak allocated VRAM was 2.69 GB.

## Results

### Raw k=1 geometry

| Checkpoint | Target variance | Effective rank | Mean-direction energy | Correct-minus-matched cosine | Matched retrieval top-1 |
|---|---:|---:|---:|---:|---:|
| Base ChemFM | 0.022171 | 26.164 | 0.674849 | 0.025028 | 0.343750 |
| Native epoch 3 | 0.015871 | 3.050 | 0.779364 | 0.016019 | 0.236328 |
| cLM-JEPA epoch 3 | 0.00003206 | 56.499 | 0.999617 | 0.00009372 | 0.722656 |

The base model is anisotropic, but not extremely so: 32.5% of target-state energy remains outside its mean direction and effective rank is 26.2. Native fine-tuning worsens the raw geometry even without JEPA: mean-direction energy rises to 0.779, effective rank falls to 3.05, and raw retrieval falls slightly below the four-way 0.25 chance baseline.

cLM-JEPA changes the scale much more sharply. Target variance is 692 times lower than base and 495 times lower than native, while 99.9617% of energy lies in the mean direction. The raw matched cosine margin is 267 times smaller than base. This extreme common-direction concentration therefore **emerges during cLM-JEPA training**; it is not merely inherited from ChemFM.

This is not evidence of dimensional or constant-vector collapse. cLM-JEPA target effective rank is 56.5, higher than both controls, and its raw four-way retrieval is already 0.723. The across-example residual has very small amplitude but contains strong, multidirectional pair information.

### Cached-embedding residual tests

One shared mean and shared PCs were fitted to the concatenated source and target embeddings for each checkpoint. The same transformation was then applied to both views, preserving their common coordinate system. No model calls were made for these tests.

| Checkpoint and transformation | Correct-minus-matched cosine | Matched retrieval top-1 |
|---|---:|---:|
| Base: raw | 0.025028 | 0.343750 |
| Base: mean centered | 0.058230 | 0.346680 |
| Base: centered, remove PC1 | 0.040880 | 0.318359 |
| Base: centered, remove top 2 PCs | 0.047363 | 0.336914 |
| Base: centered, remove top 4 PCs | 0.059808 | 0.348633 |
| Native: raw | 0.016019 | 0.236328 |
| Native: mean centered | 0.044349 | 0.345703 |
| Native: centered, remove PC1 | 0.122120 | 0.571289 |
| Native: centered, remove top 2 PCs | 0.133880 | 0.571289 |
| Native: centered, remove top 4 PCs | 0.081659 | 0.397461 |
| cLM-JEPA: raw | 0.00009372 | 0.722656 |
| cLM-JEPA: mean centered | 0.213060 | 0.762695 |
| cLM-JEPA: centered, remove PC1 | 0.251651 | 0.816406 |
| cLM-JEPA: centered, remove top 2 PCs | **0.285808** | 0.847656 |
| cLM-JEPA: centered, remove top 4 PCs | 0.273486 | **0.859375** |

Mean centering increases cLM-JEPA's matched margin 2,273-fold. Centering plus removing two PCs increases it 3,050-fold, to 0.286, while retrieval rises from 0.723 to 0.848. The result is not a generic benefit of deleting PCs: base retrieval is essentially unchanged and removing PC1 slightly hurts it. Native benefits substantially from removing one or two dominant PCs, but cLM-JEPA's recovered margin and retrieval are much stronger.

Full whitening was not run. With 2,048 centered observations in a roughly 2,048-dimensional hidden space, the joint covariance is rank-deficient by construction. A stable whitening result would require an arbitrary shrinkage or eigenvalue-floor hyperparameter, which was not frozen and is unnecessary to answer the causal question.

## Most supported causal diagnosis

The evidence supports a combination of explanations **(2) and (3)**, with (2) producing the near-zero raw cosine loss and (3) explaining why pair information remains recoverable:

1. ChemFM starts with moderate anisotropy, so explanation (1) is background context rather than the main cause.
2. Ordinary native fine-tuning also worsens geometry, particularly effective rank, showing that some dominant-direction growth is not JEPA-specific.
3. cLM-JEPA uniquely drives target variance down by roughly three orders of magnitude and mean-direction energy to 0.999617. Positive-pair raw cosine can consequently approach one without requiring a large reaction-specific component.
4. The reaction-specific component is useful rather than absent: raw cLM-JEPA retrieval is high, and centering/PC removal reveals a very large matched margin and still higher retrieval.

The precise failure is therefore **objective-induced extreme anisotropy that lets raw cosine underweight a chemically informative residual**. It should not be called representation collapse: rank and retrieval directly contradict that stronger claim. It also does not look like a k=1 readout failure. The target EOS geometry is independent of k, and the k=1 source residual becomes the best of the three checkpoints after removing common components. The conditional 256-example readout assay was therefore not run.

Alternative explanations remain bounded:

- Checkpoint-wide generative fine-tuning contributes to anisotropy, as native demonstrates, but cannot explain cLM-JEPA's additional 495-fold variance reduction.
- PCA uses the diagnostic sample and is explanatory, not a deployable transformation or checkpoint-selection rule.
- High residual retrieval does not prove that the decoder can exploit the residual; the paired 512-reaction decoder-coupling assay below finds no generative benefit.
- Endpoint gradients were not measured because they cannot reconstruct the optimization path and the cached-state intervention already distinguishes the three proposed causes.

## Comparison with the prior protein-LM JEPA failure mode

The 2026 [ProteinJEPA paper](https://arxiv.org/abs/2605.07554) reports that JEPA-only training loses identity-sensitive downstream performance across nearly every setting, while retaining masked-language-model cross-entropy and adding collapse-prevention regularization is critical. Its failure is behavioral loss of identity information when latent prediction replaces token supervision.

This cLM-JEPA result is related but not the same. cLM-JEPA retained native token supervision and preserves exceptionally strong pair identity in the residual, but the paired 512-reaction assay below finds lower exact top-1 than native. Its problem is not loss of identity information; it is that an unregularized, positive-only endpoint cosine objective can be minimized through a dominant common direction while leaving useful chemistry too small for raw cosine to weight strongly or the decoder to exploit. ProteinJEPA's use of variance/covariance regularization or an EMA-style target addresses the same family of geometric shortcut, but the present evidence does not justify importing its stronger word “collapse” here.

## Frozen decoder-coupling diagnostic: 512 reactions

**Protocol.** The selected seed-533 epoch-3 native and fully correct cLM-JEPA checkpoints were kept frozen. The assay uses 512 unique canonical USPTO-MIT validation reactions, one official enumeration per reaction, beam 10, and generation batch size one. The first 512 unique identities in the native deterministic prompt-length traversal were frozen before metric inspection. Two identities that remained late in the cLM traversal were replaced, before any metric was read, by the lowest panel-index identities already present in both streams (indices 69 and 72). This retained 510 frozen identities and produced 512 exactly paired identities without correctness- or loss-based selection.

### Forward generation

| Metric | Native | cLM-JEPA | cLM minus native |
|---|---:|---:|---:|
| Exact top-1 (primary) | **0.046875** | 0.033203 | -0.013672 |
| Exact top-3 | **0.140625** | 0.136719 | -0.003906 |
| Exact top-5 | **0.205078** | 0.203125 | -0.001953 |
| Exact top-10 | 0.265625 | 0.265625 | 0.000000 |
| Valid beam fraction | 0.835352 | **0.871680** | +0.036328 |

There were 12 native-only exact top-1 successes and 5 cLM-JEPA-only successes. The paired top-1 difference was -1.37 percentage points with a bootstrap 95% interval of [-2.93, +0.20] points; exact McNemar p = 0.143. Thus the sample does not establish a statistically precise harm, but it rules out the proposed explanation that the 32-reaction tie merely hid an observable cLM-JEPA advantage. Rank improved on 60 reactions, worsened on 65, and tied on 387; mean rank improvement was -0.043 (95% bootstrap interval [-0.231, 0.139]). Top-10 was exactly tied.

### Target-token cross-entropy

Token-weighted target CE was 0.239942 for native and 0.238685 for cLM-JEPA, a 0.524% relative cLM-JEPA improvement. The effect was not broad or precise: 52.54% of reactions improved and 47.46% worsened; mean length-normalized per-reaction improvement was 0.001304 with 95% bootstrap interval [-0.003427, 0.006174], and paired Wilcoxon p = 0.367. The larger-sample result therefore does not reproduce the earlier approximately 3.2% advantage. It leaves only weak evidence for a small teacher-forced benefit and no forward-generation benefit.

### Does JEPA pair strength predict decoder improvement?

The cLM-JEPA raw pair margin averaged 0.0000895; after joint centering and shared top-2-PC removal, the analysis-only residual margin averaged 0.288953.

| Signal versus outcome | Spearman rho | Bootstrap 95% interval |
|---|---:|---:|
| Raw margin versus CE improvement | -0.056 | [-0.140, 0.033] |
| Residual margin versus CE improvement | -0.054 | [-0.142, 0.037] |
| Raw margin versus rank improvement | 0.006 | [-0.076, 0.095] |
| Residual margin versus rank improvement | 0.003 | [-0.080, 0.085] |

Top-minus-bottom residual-signal quartiles differed by -0.0074 CE improvement (95% interval [-0.0218, 0.0073]) and +0.039 rank positions (95% interval [-0.422, 0.516]). A residual-margin association with cLM-only top-1 transitions was small (rho = 0.091, p = 0.040) and rests on only five such transitions; it did not reproduce at top-3, top-5, or top-10 and is not persuasive after considering the family of cutoff analyses.

### Source interventions

| Intervention | Native raw predictor sensitivity | cLM raw predictor sensitivity | cLM residual sensitivity | cLM target CE change | cLM target KL |
|---|---:|---:|---:|---:|---:|
| Contributor removal | 0.07517 | 0.000337 | 0.4622 | +0.5541 | 0.5984 |
| Contributor replacement | 0.07341 | 0.000111 | 0.3991 | +0.7595 | 0.7920 |
| Unrelated source | 0.12847 | 0.000271 | 0.9106 | +0.9221 | 0.9834 |

The cLM raw `[PRED]` vector is extraordinarily insensitive to source content even when the normal generative pathway changes strongly. Removing, replacing, or fully substituting the source produces large target CE and token-distribution KL changes, comparable to the native decoder, while raw cLM predictor sensitivity remains approximately 0.0001–0.00034. Centering/top-PC removal reveals large residual sensitivity, confirming that chemistry is present but carried at a scale that the raw cosine/readout largely ignores.

### Causal interpretation

The strongest diagnosis is **C, with D as supporting evidence**:

- The cLM-JEPA auxiliary residual contains strong pair-specific chemistry, but neither raw nor residual pair strength consistently predicts per-reaction CE or generated-product rank improvement.
- Normal generation is highly source-sensitive while the raw cLM `[PRED]` representation is nearly invariant; this is direct evidence that the useful residual is largely decoupled from the normal autoregressive pathway.
- The small 0.524% aggregate CE advantage is uncertain and occurs on only a slight majority of reactions. Exact top-1 is lower, not higher. The earlier apparent native-loss benefit largely disappears on the paired 512 sample.
- Explanation A is not supported. Explanation B receives, at most, one fragile cutoff-specific signal that is inconsistent with the CE, rank, and other-cutoff evidence.

This remains **extreme anisotropy plus decoder decoupling**, not demonstrated representation collapse. Effective rank, residual retrieval, residual source sensitivity, and the large centered margin all show that information survives.

## Exactly one recommended next experiment

Run **one seed-533 readout-only coupling pilot that replaces the isolated k=1 `[PRED]` source readout with the normal pre-EOS generative source state for the same JEPA prediction and loss, changing nothing else**. Compare only with the existing seed-533 native checkpoint. Require a native-beating exact top-1 result as the primary criterion and a positive association between pair margin and CE/rank improvement as the mechanism criterion. Do not combine this pilot with centering, InfoNCE, EMA targets, variance regularization, or any other objective change; the purpose is to test decoder coupling as one variable.

## Artifacts

- Results: `runs/diagnostics/uspto_mit_geometry_diagnosis.json`
- Cached embeddings: `runs/diagnostics/geometry_cache/{base,native_epoch3,clm_jepa_epoch3}.pt`
- Reproduction code: `src/geometry_diagnosis.py`
- Paired decoder-coupling summary: `runs/diagnostics/decoder_coupling/summary_512.json`
- Paired beam outputs: `runs/diagnostics/decoder_coupling/{native,clm_jepa}_generation_512.jsonl`
- Per-reaction CE, pair margins, and interventions: `runs/diagnostics/decoder_coupling/{native,clm_jepa}_diagnostics.json`
- Frozen 1,024-identity parent panel: `data/gate5_decoder_coupling/uspto_mit_validation_1024.csv`
