# cLM-JEPA Gate 5 tie diagnostic

**Status:** diagnostic completed for the selected USPTO-MIT cLM-JEPA endpoint on 2026-08-09. This is a reduced pilot, not a claim of success across all three tasks.

## Executive conclusion

The completed evidence shows a literal primary-metric tie, not a cLM-JEPA win. On the same 32-reaction validation set and seed 533, native ChemFM-LoRA and cLM-JEPA each recovered exactly the same two reactions at rank 1: exact top-1 was 2/32 (0.0625) for both, with zero native-only and zero cLM-JEPA-only top-1 successes. cLM-JEPA improved exact top-10 by one reaction and validation teacher-forced loss by 0.00705, but lost one exact top-3 reaction. These secondary movements are mixed and do not override the frozen top-1 selector.

The most likely mechanism is a **mean-direction shortcut in the JEPA state objective**. At the selected cLM-JEPA checkpoint, the JEPA loss became extremely small, yet corrected Section 11 diagnostics show that 99.9630% of target-state energy lies in one mean direction. Correct-pair cosine exceeds a size-matched wrong-pair cosine by only 0.0000998. Removing the source component with the largest ring-aware MCS overlap to the product changes the source state by only 0.000338 cosine distance; replacing it with a size-matched component from another reaction changes it by only 0.000129. Thus, the auxiliary objective is easy to satisfy without learning a relationship representation whose pair-specific amplitude is large enough to improve decoding.

Pair information is not completely absent: four-way retrieval top-1 is 0.6875 versus 0.25 chance, and residual directions have nontrivial rank. The diagnosis is therefore **near-common-vector concentration with a small informative residual**, not total constant-vector collapse. This reconciles the rapidly vanishing JEPA loss with a generative tie.

## What was and was not tested

Completed fine-tuning comparisons cover only reduced USPTO-MIT forward synthesis:

- native, seed 533, four epochs;
- cLM-JEPA, seeds 533 and 917, four epochs;
- fixed reliable hyperparameters, with no learning-rate search;
- primary paired comparison: native seed 533 versus cLM-JEPA seed 533.

Human metabolism and USPTO-50K retrosynthesis have not yet been fine-tuned. Seed 917 cannot establish an improvement because only cLM-JEPA was run at that seed; its 0.09375 exact top-1 has no matched native control. The monitor-only run requested for diagnosis was cancelled during epoch 2 because its full four-epoch beam-evaluation runtime was disproportionate to the added evidence on the 6 GB GPU. Its lone epoch-1 checkpoint is incomplete and excluded. The shuffled control was not launched.

## Exact extent of the tie

Both selected checkpoints were epoch 3 under the same frozen validation selector, exact top-1.

| Metric | Native seed 533 | cLM-JEPA seed 533 | cLM-JEPA minus native |
|---|---:|---:|---:|
| Exact top-1 | 0.0625 (2/32) | 0.0625 (2/32) | 0.0000 |
| Exact top-3 | 0.2500 (8/32) | 0.21875 (7/32) | -0.03125 |
| Exact top-5 | 0.28125 (9/32) | 0.2500 (8/32) | -0.03125 |
| Exact top-10 | 0.34375 (11/32) | 0.3750 (12/32) | +0.03125 |
| Valid rate | 1.0000 | 1.0000 | 0.0000 |
| Duplicate rate | 0.0000 | 0.0000 | 0.0000 |
| Mean unique valid outputs | 10.0 | 10.0 | 0.0 |
| Validation native loss | 0.219933 | 0.212888 | -0.007046 (-3.20%) |

The paired top-1 contingency is especially important:

| Outcome on one validation reaction | Count |
|---|---:|
| Both correct at top-1 | 2 |
| Native only correct at top-1 | 0 |
| cLM-JEPA only correct at top-1 | 0 |
| Neither correct at top-1 | 30 |

Across the ten-beam rank lists, cLM-JEPA improved the first exact-match rank on four reactions, worsened it on five, and tied it on 23. It gained two top-10 hits and lost one, yielding the net +1 top-10 result. There are no discordant top-1 pairs on which a paired test could support superiority. The pilot is also small: a 2/32 proportion has a wide approximate 95% Wilson interval (about 0.017 to 0.201). Exact top-1 being low is a red flag for absolute task performance and statistical power, even though it is not evidence of a scoring bug: validity is 100%, beam outputs are unique, and both independently trained conditions recover the same two top-1 products.

## Learning dynamics

### Generative validation by epoch

| Condition | Epoch | Native loss | Exact@1 | Exact@3 | Exact@10 |
|---|---:|---:|---:|---:|---:|
| Native | 1 | 0.232695 | 0.00000 | 0.06250 | 0.28125 |
| Native | 2 | 0.214255 | 0.03125 | 0.21875 | 0.37500 |
| Native | 3 (selected) | 0.219933 | 0.06250 | 0.25000 | 0.34375 |
| Native | 4 | 0.236518 | 0.06250 | 0.31250 | 0.34375 |
| cLM-JEPA | 1 | 0.233298 | 0.00000 | 0.06250 | 0.21875 |
| cLM-JEPA | 2 | 0.208141 | 0.03125 | 0.15625 | 0.40625 |
| cLM-JEPA | 3 (selected) | 0.212888 | 0.06250 | 0.21875 | 0.37500 |
| cLM-JEPA | 4 | 0.229846 | 0.06250 | 0.18750 | 0.31250 |

Both models improve top-1 along the same 0, 1, 2 correct-reaction trajectory. The cLM-JEPA advantage in native validation loss does not convert into a top-1 advantage. Higher-rank results fluctuate by one to three examples, consistent with a small validation sample.

### Training losses

Mean per-step losses by epoch were:

| Condition | Epoch | Native training loss | JEPA loss on active batches | JEPA-active batches |
|---|---:|---:|---:|---:|
| Native | 1 | 0.707046 | n/a | 0 |
| Native | 2 | 0.200289 | n/a | 0 |
| Native | 3 | 0.122972 | n/a | 0 |
| Native | 4 | 0.090330 | n/a | 0 |
| cLM-JEPA | 1 | 0.870817 | 0.152563 | 89 |
| cLM-JEPA | 2 | 0.206672 | 0.000723 | 81 |
| cLM-JEPA | 3 | 0.133379 | 0.000276 | 85 |
| cLM-JEPA | 4 | 0.098451 | 0.000238 | 82 |

The JEPA loss falls about 211-fold from epoch 1 to epoch 2 and about 641-fold from epoch 1 to epoch 4. Meanwhile, cLM-JEPA's native training loss remains slightly worse than native at every epoch. This is the first sign that the auxiliary objective is being optimized through an easy geometric solution rather than providing a useful generative inductive bias.

cLM-JEPA required 3,978 seconds versus 2,969 seconds for native on this hardware, about 34.0% more wall time. Peak allocated VRAM was essentially unchanged (4.882 GB versus 4.879 GB).

## Corrected Section 11 representation evaluation

### Diagnostic correction and provenance

The representation metrics embedded in the original run JSONs are not final evidence. The old sampler took the first 32 augmented validation rows. Those rows represented only seven canonical target identities, while randomized SMILES strings were treated as different identities. In addition, the old necessary-component assay removed the final serialized component rather than a chemically motivated contributor. These issues do **not** affect the beam-generation metrics above, which group randomized-SMILES views and canonicalize targets correctly.

The corrected evaluation:

- selects one row for each of 32 distinct canonical target identities;
- keys targets by canonical molecular identity;
- computes actual tokenizer lengths and heavy-atom counts for matched negatives;
- defines the contributor proxy as the source component with maximum ring-aware MCS atom overlap to any product component, with heavy-atom count and original order as tie-breakers;
- tests both removal and replacement of that component;
- evaluates the already-selected cLM-JEPA epoch-3 checkpoint without retraining or checkpoint reselection.

The MCS rule is a transparent contributor proxy, not atom-mapped proof that a component is mechanistically necessary. The user-directed reduced scope evaluates only the trained cLM-JEPA endpoint; therefore this is not the full five-condition Section 11 comparison specified in the plan.

### Results for selected cLM-JEPA epoch 3

| Section 11 diagnostic | Value | Interpretation |
|---|---:|---|
| Correct-pair cosine | 0.999598 | Absolute cosine is nearly saturated. |
| Random-pair cosine | 0.999464 | Random target is almost equally aligned. |
| Matched-shuffle cosine | 0.999498 | Size-matched wrong target is almost equally aligned. |
| Correct minus random | 0.0001342 | Pair-specific angular margin is tiny. |
| Correct minus matched shuffle | 0.00009984 | Central matched-control margin is tiny. |
| Target variance | 0.00003091 | Very low across-example amplitude. |
| Target effective rank | 15.9009 | The small residual is multidirectional, not rank-one. |
| Target mean-direction energy | 0.999630 | Only about 0.0370% of energy remains outside the common mean direction. |
| Held-out ridge explained variance | 0.02931 | Source states explain little centered target variance linearly. |
| Four-way retrieval top-1 | 0.6875 | Above the 0.25 size-matched chance level. |
| Four-way retrieval MRR | 0.8021 | Tiny residuals still order candidates usefully. |
| Source-target difference effective rank | 18.9504 | Difference residuals span multiple directions. |
| Top singular energy of centered differences | 0.18125 | No single difference direction dominates. |
| Alternate-SMILES cosine | 0.999888 | Strong serialization invariance, partly expected under common-vector concentration. |
| Contributor-removal cosine | 0.999662 | Removing the best-overlap component barely moves the state. |
| Contributor-removal sensitivity | 0.0003384 | Far too small to indicate strong component dependence. |
| Contributor-replacement cosine | 0.999871 | Cross-reaction replacement moves the state even less. |
| Contributor-replacement sensitivity | 0.0001289 | The representation is weakly conditional on contributor identity. |

The retrieval result and effective ranks prevent the stronger claim of total collapse. However, effective rank measures the distribution of residual energy, not its magnitude. Here the residual can carry enough information to rank three matched negatives while still being too small relative to the common vector to impose a useful constraint on the decoder. Similarly, alternate-SMILES invariance is desirable in isolation but is not strong evidence of chemical invariance when random, shuffled, ablated, and replaced inputs are also nearly parallel.

## Causal diagnosis

The following evidence supports the mean-direction-shortcut explanation:

1. **Optimization succeeds numerically.** JEPA loss falls to 0.000238 while remaining active on roughly half the batches.
2. **Pair discrimination barely exceeds controls.** Correct versus matched-shuffle margin is only 0.0000998.
3. **State amplitude collapses around a common direction.** Mean-direction energy is 0.999630 and target variance is 0.00003091.
4. **Reaction-content interventions barely matter.** Removal and replacement sensitivities are 0.000338 and 0.000129.
5. **Generation follows native rather than separating from it.** The same two examples are top-1 correct, with mixed changes deeper in the beam.
6. **Some residual identity signal survives.** Retrieval exceeds chance, explaining why the representation is not literally constant and why the failure is subtle.

This evidence is consistent with the predictor and target states satisfying cosine alignment mainly through a dominant shared direction. Because cosine loss does not reward a large pair-specific margin or variance, the tiny informative residual is sufficient to drive the auxiliary loss down. The generative objective continues to do the useful work, yielding behavior close to native fine-tuning rather than a meaningful improvement.

Alternative explanations are less consistent with the complete pattern:

- **Evaluation/scoring failure:** unlikely as the sole cause, because canonical beam grouping, validity, uniqueness, and paired exact hits are coherent. The discovered bug was isolated to the old diagnostic sampler, not generative scoring.
- **JEPA never activated:** contradicted by 337 active microbatches and the observed JEPA curve.
- **Pure representational rank collapse:** contradicted by effective rank and retrieval; the problem is primarily scale/common-direction dominance.
- **Insufficient training:** top-1 saturates for both models by epoch 3, while JEPA loss is already near zero by epoch 2. More epochs under the same objective are unlikely to fix the mechanism.
- **Definitive proof of architectural failure:** not supported. This is one small forward-synthesis pilot, without completed monitor/shuffled controls or matched native representation diagnostics under the corrected sampler.

## Decision and efficient next action

Gate 5 should remain **not passed**. cLM-JEPA did not beat native on the primary metric, and the mechanism assay supplies a concrete warning that the auxiliary objective found a low-variance shortcut. The result does justify testing transportability on the two missing tasks with small, faithful native-versus-cLM-JEPA pilots, because task structure may change whether the residual signal is useful. It does not justify broad hyperparameter search, longer USPTO-MIT runs, or claiming a cLM-JEPA gain.

For efficiency, the next tests should use the fixed reliable configuration and a single paired seed first:

1. construct the required parent-grouped, scaffold-disjoint MetaTrans validation split before any metabolism fine-tuning;
2. run a small metabolism native/cLM-JEPA pair and assess recall@5 with the plan's lower-bound precision reporting;
3. run a small USPTO-50K product-to-precursor native/cLM-JEPA pair and assess exact precursor-set top-1;
4. add the second fixed seed only if cLM-JEPA improves the prespecified primary metric without validity or native-loss regression.

## Evidence files

- Generative native result: `runs/gate5/runs/native-s533.json`
- Generative cLM-JEPA result: `runs/gate5/runs/clm_jepa-s533.json`
- Corrected cLM-JEPA Section 11 result: `runs/diagnostics/uspto_mit_clm_jepa_section11.json`
- Selected native checkpoint: `runs/gate5/checkpoints/native-s533/epoch_3`
- Selected cLM-JEPA checkpoint: `runs/gate4_v2/reliable/clm_jepa-s533-checkpoints/epoch_3`
- Corrected evaluation implementation: `src/representation_eval.py` and `src/train.py`
