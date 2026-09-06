# Native oracle next-token transition decomposition

**Supported conclusion: genuinely unresolved future-state structure under the locked probe classes.**

Revealing the realized token helps, but does not dominate the missing information.
Reaction-first residual removal is 10.05% [9.09%, 11.03%] for ridge and
20.53% [19.20%, 21.88%] for the residual MLP. Nonlinear C improves paired-mean
full-state R2 from A's .640 to .715, yet leaves .285 normalized MSE.
The train-target rank-256 PCA retains 98.62%/98.68% variance, so its small
training projection loss does not explain a residual this large. Training
coverage is not itself a measured test-set ceiling.

Four-state D does not improve on C: R2 falls to .559 for ridge and .648
for the MLP; reaction-first history partial reductions are -.132 and -.235.
Because D contains C's features, this deterioration reflects the performance
of the fixed finite-data, regularized, early-stopped probes, not negative
information in history. These results do not establish that h[t] is a
sufficient state or rule out information in longer prefixes/KV state.

The recovered information is decoder-functional. Nonlinear A-to-C reduces JS
from .125 to .097, raises top-1 agreement from .784 to .824, improves the
x[t+2] rank from 2.81 to 2.10, and raises its logit margin from 3.913 to 4.423.
Nonlinear D worsens JS and top-1 relative to C. Ridge D has a narrower
decoder benefit despite worse hidden reconstruction, so hidden R2 and decoder
preservation should remain separate endpoints.

Token-only B is substantially weaker (ridge R2 .259 versus C .611);
causal state removes 47.59% [46.45%, 48.73%] of B's residual. Shuffled-token C
reaches only .539 R2, versus .611 with the correct token. The shuffle uses
the existing cross-reaction derangement with nearest product position within
the assigned donor reaction; length matching is approximate, not exact.

Thus neither token-uncertainty dominance, demonstrated latent-state
insufficiency, nor functionally irrelevant reconstructed variance is supported
as the single explanation. The remaining structure is unresolved by this
experiment, not proven intrinsically unpredictable. The supported next step
is the predictable plus decoder-relevant subspace audit; no new ChemFM
training or additional audit is performed here.


Frozen Native r8 seeds 533/917; product final post-RMSNorm; k=1. All models and seeds use identical t-3..t+2 eligible rows. Primary metrics average within reaction, then across reactions. MLP rows average metrics across probe seeds 20260904/5/6, without ensembling predictions.

Train-target rank-256 PCA coverage: 533=0.986221; 917=0.986794. PCA and input standardizers use only training data. The original Report-02 .641 is not a matched-row comparator.

| Seed | Probe/input | R2 | nMSE | cos / centered | P token∣h | P h∣token | P history | JS | top-1 | x[t+2] rank | margin |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 533 | ridge A | 0.563 | 0.437 | 0.816 / 0.742 | — | — | — | 0.159 | 0.735 | 3.23 | 2.843 |
| 533 | ridge B | 0.259 | 0.741 | 0.660 / 0.479 | — | — | — | 0.363 | 0.440 | 7.49 | -0.869 |
| 533 | ridge C | 0.608 | 0.392 | 0.837 / 0.775 | 0.102 | 0.473 | — | 0.143 | 0.755 | 2.82 | 3.040 |
| 533 | ridge D | 0.556 | 0.444 | 0.821 / 0.752 | — | — | -0.135 | 0.133 | 0.766 | 2.56 | 3.334 |
| 533 | ridge C_shuffle | 0.539 | 0.461 | 0.804 / 0.727 | — | — | — | 0.169 | 0.717 | 3.42 | 2.684 |
| 533 | mlp A | 0.638 | 0.362 | 0.850 / 0.792 | — | — | — | 0.125 | 0.782 | 2.75 | 3.939 |
| 533 | mlp C | 0.712 | 0.288 | 0.883 / 0.838 | 0.203 | — | — | 0.097 | 0.823 | 2.08 | 4.404 |
| 533 | mlp D | 0.644 | 0.356 | 0.857 / 0.802 | — | — | -0.239 | 0.103 | 0.814 | 2.11 | 4.207 |
| 917 | ridge A | 0.570 | 0.430 | 0.820 / 0.746 | — | — | — | 0.158 | 0.732 | 3.46 | 2.804 |
| 917 | ridge B | 0.260 | 0.740 | 0.664 / 0.480 | — | — | — | 0.363 | 0.441 | 8.68 | -0.913 |
| 917 | ridge C | 0.613 | 0.387 | 0.840 / 0.777 | 0.099 | 0.478 | — | 0.143 | 0.756 | 2.90 | 2.997 |
| 917 | ridge D | 0.563 | 0.437 | 0.825 / 0.756 | — | — | -0.129 | 0.133 | 0.766 | 2.67 | 3.301 |
| 917 | ridge C_shuffle | 0.539 | 0.461 | 0.807 / 0.728 | — | — | — | 0.168 | 0.720 | 3.50 | 2.667 |
| 917 | mlp A | 0.642 | 0.358 | 0.853 / 0.795 | — | — | — | 0.124 | 0.785 | 2.86 | 3.888 |
| 917 | mlp C | 0.718 | 0.282 | 0.886 / 0.842 | 0.208 | — | — | 0.096 | 0.824 | 2.12 | 4.442 |
| 917 | mlp D | 0.653 | 0.347 | 0.862 / 0.809 | — | — | -0.231 | 0.102 | 0.814 | 2.15 | 4.186 |
| paired mean | ridge A | 0.566 | 0.434 | 0.818 / 0.744 | — | — | — | 0.159 | 0.734 | 3.35 | 2.823 |
| paired mean | ridge B | 0.259 | 0.741 | 0.662 / 0.480 | — | — | — | 0.363 | 0.440 | 8.08 | -0.891 |
| paired mean | ridge C | 0.611 | 0.389 | 0.839 / 0.776 | 0.101 | 0.476 | — | 0.143 | 0.756 | 2.86 | 3.019 |
| paired mean | ridge D | 0.559 | 0.441 | 0.823 / 0.754 | — | — | -0.132 | 0.133 | 0.766 | 2.62 | 3.318 |
| paired mean | ridge C_shuffle | 0.539 | 0.461 | 0.805 / 0.727 | — | — | — | 0.168 | 0.719 | 3.46 | 2.676 |
| paired mean | mlp A | 0.640 | 0.360 | 0.852 / 0.793 | — | — | — | 0.125 | 0.784 | 2.81 | 3.913 |
| paired mean | mlp C | 0.715 | 0.285 | 0.884 / 0.840 | 0.205 | — | — | 0.097 | 0.824 | 2.10 | 4.423 |
| paired mean | mlp D | 0.648 | 0.352 | 0.859 / 0.805 | — | — | -0.235 | 0.103 | 0.814 | 2.13 | 4.197 |

Paired conditional information (mean; crossed checkpoint/reaction 95% bootstrap CI):

- ridge:P_token_given_h: 0.1005 [0.0909, 0.1103].
- ridge:P_history_given_h_token: -0.1321 [-0.1430, -0.1210].
- ridge:P_h_given_token: 0.4759 [0.4645, 0.4873].
- mlp:P_token_given_h: 0.2053 [0.1920, 0.2188].
- mlp:P_history_given_h_token: -0.2346 [-0.2460, -0.2226].

Paired functional and reconstruction differences (improved minus baseline):

- ridge:A->C:r2: 0.0442 [0.0395, 0.0490].
- ridge:A->C:js: -0.0155 [-0.0182, -0.0128].
- ridge:A->C:top1_agreement: 0.0221 [0.0151, 0.0290].
- ridge:A->C:gold_rank: -0.4902 [-0.7458, -0.2423].
- ridge:A->C:gold_margin: 0.1952 [0.1507, 0.2428].
- ridge:C->D:r2: -0.0514 [-0.0560, -0.0468].
- ridge:C->D:js: -0.0105 [-0.0130, -0.0080].
- ridge:C->D:top1_agreement: 0.0108 [0.0042, 0.0176].
- ridge:C->D:gold_rank: -0.2399 [-0.4487, -0.0425].
- ridge:C->D:gold_margin: 0.2991 [0.2425, 0.3540].
- mlp:A->C:r2: 0.0748 [0.0693, 0.0805].
- mlp:A->C:js: -0.0280 [-0.0309, -0.0252].
- mlp:A->C:top1_agreement: 0.0402 [0.0344, 0.0459].
- mlp:A->C:gold_rank: -0.7083 [-0.8995, -0.5173].
- mlp:A->C:gold_margin: 0.5098 [0.4282, 0.5920].
- mlp:C->D:r2: -0.0665 [-0.0704, -0.0624].
- mlp:C->D:js: 0.0061 [0.0038, 0.0085].
- mlp:C->D:top1_agreement: -0.0098 [-0.0154, -0.0044].
- mlp:C->D:gold_rank: 0.0338 [-0.0835, 0.1446].
- mlp:C->D:gold_margin: -0.2263 [-0.2911, -0.1626].
- ridge:C_shuffle->ridge:C:r2: 0.0715 [0.0660, 0.0773].
- ridge:C_shuffle->ridge:C:js: -0.0248 [-0.0276, -0.0221].
- ridge:C_shuffle->ridge:C:top1_agreement: 0.0370 [0.0301, 0.0437].
- ridge:C_shuffle->ridge:C:gold_rank: -0.6006 [-0.8537, -0.3612].
- ridge:C_shuffle->ridge:C:gold_margin: 0.3430 [0.2874, 0.3972].
- ridge:A->mlp:A:r2: 0.0737 [0.0705, 0.0771].
- ridge:A->mlp:A:js: -0.0340 [-0.0365, -0.0314].
- ridge:A->mlp:A:top1_agreement: 0.0500 [0.0425, 0.0581].
- ridge:A->mlp:A:gold_rank: -0.5396 [-0.7250, -0.3446].
- ridge:A->mlp:A:gold_margin: 1.0898 [1.0411, 1.1413].
- ridge:C->mlp:C:r2: 0.1043 [0.1009, 0.1079].
- ridge:C->mlp:C:js: -0.0465 [-0.0494, -0.0437].
- ridge:C->mlp:C:top1_agreement: 0.0682 [0.0615, 0.0753].
- ridge:C->mlp:C:gold_rank: -0.7577 [-0.9702, -0.5660].
- ridge:C->mlp:C:gold_margin: 1.4044 [1.3267, 1.4815].
- ridge:D->mlp:D:r2: 0.0892 [0.0859, 0.0924].
- ridge:D->mlp:D:js: -0.0299 [-0.0324, -0.0274].
- ridge:D->mlp:D:top1_agreement: 0.0476 [0.0415, 0.0540].
- ridge:D->mlp:D:gold_rank: -0.4840 [-0.6465, -0.3281].
- ridge:D->mlp:D:gold_margin: 0.8790 [0.8348, 0.9273].

The bootstrap uses 4,000 paired draws, resampling reactions jointly across checkpoints and checkpoint indices jointly across reactions. Inference across training seeds remains limited by having only two checkpoints. Conditional quantities use each reaction's SSE ratio first.

All KL, JS, top-1/5/10, teacher probability/log-probability/rank/margin, latent cosine, centered cosine, SSE/SST, and per-reaction results are retained in metrics_*.json and table.csv. Actual-state decoder references are in the ridge metrics files. Saved BF16 LM-head weights are used unchanged over all saved outputs (402, including reserved IDs). Unlike Report 02's decoder tables, there is no native-vocabulary truncation or renormalization.

Ridge: AdamW lr=.003, weight_decay=.001; residual MLP: frozen ridge plus Linear(input,128)-GELU-Linear(128,256), weight_decay=.0001. Both use batch size 512, at most 20 epochs, validation-only early stopping (patience 3, improvement 1e-8). Train sampling is reaction-balanced and capped at 16,384 positions. Initialization is explicitly seeded. No ChemFM weights are trained or tokens generated.
