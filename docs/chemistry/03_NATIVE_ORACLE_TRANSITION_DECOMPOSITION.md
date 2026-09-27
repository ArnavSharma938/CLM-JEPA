# Native oracle next-token transition decomposition

**Supported conclusion: next-token revelation explains a real, decoder-functional minority of the one-step residual; most future-state structure remains unresolved under these probes.**

Reaction-first residual removal is 10.05% [9.09%, 11.03%] for ridge and
20.53% [19.20%, 21.88%] for the residual MLP. Nonlinear C improves paired-mean
full-state R2 from A's .640 to .715, yet leaves .285 normalized MSE.
The actual test PCA ceilings are .984584/.985172 for seeds 533/917, with
projection nMSE .015416/.014828. These measured test projection errors,
rather than training variance coverage, establish that the fixed target
basis accounts for only a small part of the remaining reconstruction error.

The capacity-matched nonlinear shuffle control confirms that the correct
realized token matters. C_shuffle MLP reaches .626 R2 versus C's .715:
correct-minus-shuffled R2 is +.0891 [.0835, .0947], native-vocabulary JS
changes by -.03558 [-.03884, -.03235], and top-1 agreement increases by
.05294 [.04671, .05954]. Both checkpoints agree in direction. Extra input
dimensionality alone does not reproduce the oracle benefit under this control.

The recovered information is decoder-functional. With the corrected native
392-token vocabulary, nonlinear A-to-C reduces JS from .12439 to .09648,
raises top-1 agreement from .78344 to .82390, improves x[t+2] rank from
2.671 to 2.024, and raises its logit margin from 3.913 to 4.423.
The original 402-output decoder results included added predictor/reserved
tokens and are now explicitly robustness-only, not the primary comparison.

Four-state D underperforms C: R2 falls to .559 for ridge and .648 for
the MLP; reaction-first history partial reductions are -.132 and -.235.
D has 10,240 input dimensions versus C's 4,096 under the same finite-data,
regularized, early-stopped probe budget. D contains C's features, so this
performance loss does not establish h[t] sufficiency and must not be read
as negative information in history. Longer-prefix/KV information and more
effective estimators remain untested.

Ridge C-to-D shows a meaningful reconstruction/function dissociation:
hidden R2 worsens (.611 to .559), while native decoder JS improves
(.14278 to .13242), top-1 rises (.75532 to .76631), rank improves
(2.715 to 2.501), and margin rises (3.018 to 3.318). Whole-state
reconstruction and decoder-functional fidelity are distinct objectives.
Nonlinear D does not reproduce that functional advantage: its JS and
top-1 are worse than nonlinear C's, and rank does not clearly improve.

Token-only B is substantially weaker (ridge R2 .259 versus C .611);
causal state removes 47.59% [46.45%, 48.73%] of B's residual. The
cross-reaction token shuffle is approximately position/length matched,
not exact; its measured donor gaps are reported below.

The controls support a specific, functional oracle-token contribution,
not token-uncertainty dominance, proof of latent-state sufficiency or
insufficiency, or intrinsically unpredictable hidden structure. A
predictable, decoder-relevant subspace audit remains a reasonable next
question; no such additional audit or ChemFM training is performed here.


Frozen Native r8 seeds 533/917; product final post-RMSNorm; k=1. All models and seeds use identical t-3..t+2 eligible rows. Primary metrics average within reaction, then across reactions. MLP rows average metrics across probe seeds 20260904/5/6, without ensembling predictions.

Train-target rank-256 PCA coverage: 533=0.986221; 917=0.986794. PCA and input standardizers use only training data. The original Report-02 .641 is not a matched-row comparator.

| Seed | Probe/input | R2 | nMSE | cos / centered | P token∣h | P h∣token | P history | JS | top-1 | x[t+2] rank | margin |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 533 | ridge A | 0.563 | 0.437 | 0.816 / 0.742 | — | — | — | 0.159 | 0.735 | 3.07 | 2.843 |
| 533 | ridge B | 0.259 | 0.741 | 0.660 / 0.479 | — | — | — | 0.362 | 0.440 | 6.98 | -0.873 |
| 533 | ridge C | 0.608 | 0.392 | 0.837 / 0.775 | 0.102 | 0.473 | — | 0.143 | 0.755 | 2.68 | 3.040 |
| 533 | ridge D | 0.556 | 0.444 | 0.821 / 0.752 | — | — | -0.135 | 0.133 | 0.766 | 2.45 | 3.334 |
| 533 | ridge C_shuffle | 0.539 | 0.461 | 0.804 / 0.727 | — | — | — | 0.168 | 0.717 | 3.24 | 2.685 |
| 533 | mlp A | 0.638 | 0.362 | 0.850 / 0.792 | — | — | — | 0.125 | 0.782 | 2.63 | 3.939 |
| 533 | mlp C | 0.712 | 0.288 | 0.883 / 0.838 | 0.203 | — | — | 0.097 | 0.823 | 2.01 | 4.403 |
| 533 | mlp D | 0.644 | 0.356 | 0.857 / 0.802 | — | — | -0.239 | 0.103 | 0.814 | 2.04 | 4.207 |
| 533 | mlp C_shuffle | 0.623 | 0.377 | 0.843 / 0.782 | — | — | — | 0.134 | 0.769 | 2.76 | 3.755 |
| 917 | ridge A | 0.570 | 0.430 | 0.820 / 0.746 | — | — | — | 0.158 | 0.732 | 3.26 | 2.804 |
| 917 | ridge B | 0.260 | 0.740 | 0.664 / 0.480 | — | — | — | 0.361 | 0.441 | 8.02 | -0.914 |
| 917 | ridge C | 0.613 | 0.387 | 0.840 / 0.777 | 0.099 | 0.478 | — | 0.143 | 0.756 | 2.75 | 2.997 |
| 917 | ridge D | 0.563 | 0.437 | 0.825 / 0.756 | — | — | -0.129 | 0.132 | 0.766 | 2.55 | 3.301 |
| 917 | ridge C_shuffle | 0.539 | 0.461 | 0.807 / 0.728 | — | — | — | 0.167 | 0.720 | 3.30 | 2.667 |
| 917 | mlp A | 0.642 | 0.358 | 0.853 / 0.795 | — | — | — | 0.124 | 0.785 | 2.72 | 3.888 |
| 917 | mlp C | 0.718 | 0.282 | 0.886 / 0.842 | 0.208 | — | — | 0.096 | 0.824 | 2.04 | 4.442 |
| 917 | mlp D | 0.653 | 0.347 | 0.862 / 0.809 | — | — | -0.231 | 0.102 | 0.813 | 2.07 | 4.186 |
| 917 | mlp C_shuffle | 0.628 | 0.372 | 0.847 / 0.786 | — | — | — | 0.131 | 0.773 | 2.90 | 3.771 |
| paired mean | ridge A | 0.566 | 0.434 | 0.818 / 0.744 | — | — | — | 0.158 | 0.733 | 3.17 | 2.823 |
| paired mean | ridge B | 0.259 | 0.741 | 0.662 / 0.480 | — | — | — | 0.362 | 0.440 | 7.50 | -0.893 |
| paired mean | ridge C | 0.611 | 0.389 | 0.839 / 0.776 | 0.101 | 0.476 | — | 0.143 | 0.755 | 2.72 | 3.018 |
| paired mean | ridge D | 0.559 | 0.441 | 0.823 / 0.754 | — | — | -0.132 | 0.132 | 0.766 | 2.50 | 3.318 |
| paired mean | ridge C_shuffle | 0.539 | 0.461 | 0.805 / 0.727 | — | — | — | 0.167 | 0.719 | 3.27 | 2.676 |
| paired mean | mlp A | 0.640 | 0.360 | 0.852 / 0.793 | — | — | — | 0.124 | 0.783 | 2.67 | 3.913 |
| paired mean | mlp C | 0.715 | 0.285 | 0.884 / 0.840 | 0.205 | — | — | 0.096 | 0.824 | 2.02 | 4.423 |
| paired mean | mlp D | 0.648 | 0.352 | 0.859 / 0.805 | — | — | -0.235 | 0.103 | 0.814 | 2.05 | 4.197 |
| paired mean | mlp C_shuffle | 0.626 | 0.374 | 0.845 / 0.784 | — | — | — | 0.132 | 0.771 | 2.83 | 3.763 |

Paired conditional information (mean; crossed checkpoint/reaction 95% bootstrap CI):

- ridge:P_token_given_h: 0.1005 [0.0909, 0.1103].
- ridge:P_history_given_h_token: -0.1321 [-0.1430, -0.1210].
- ridge:P_h_given_token: 0.4759 [0.4645, 0.4873].
- mlp:P_token_given_h: 0.2053 [0.1920, 0.2188].
- mlp:P_history_given_h_token: -0.2346 [-0.2460, -0.2226].

Paired functional and reconstruction differences (improved minus baseline):

- ridge:A->C:r2: 0.0442 [0.0395, 0.0490].
- ridge:A->C:normalized_mse: -0.0442 [-0.0490, -0.0395].
- ridge:A->C:kl_true_predicted: -0.1029 [-0.1281, -0.0780].
- ridge:A->C:js: -0.0154 [-0.0181, -0.0126].
- ridge:A->C:top1_agreement: 0.0219 [0.0149, 0.0289].
- ridge:A->C:top5_overlap: 0.0238 [0.0204, 0.0272].
- ridge:A->C:top10_overlap: 0.0199 [0.0156, 0.0238].
- ridge:A->C:gold_log_probability: 0.0959 [0.0701, 0.1220].
- ridge:A->C:gold_probability: 0.0192 [0.0147, 0.0235].
- ridge:A->C:gold_rank: -0.4505 [-0.6765, -0.2305].
- ridge:A->C:gold_margin: 0.1951 [0.1504, 0.2430].
- ridge:C->D:r2: -0.0514 [-0.0560, -0.0468].
- ridge:C->D:normalized_mse: 0.0514 [0.0468, 0.0560].
- ridge:C->D:kl_true_predicted: -0.0533 [-0.0737, -0.0338].
- ridge:C->D:js: -0.0104 [-0.0129, -0.0079].
- ridge:C->D:top1_agreement: 0.0110 [0.0044, 0.0179].
- ridge:C->D:top5_overlap: -0.0064 [-0.0099, -0.0031].
- ridge:C->D:top10_overlap: -0.0133 [-0.0168, -0.0097].
- ridge:C->D:gold_log_probability: 0.0570 [0.0356, 0.0802].
- ridge:C->D:gold_probability: 0.0201 [0.0155, 0.0247].
- ridge:C->D:gold_rank: -0.2146 [-0.4005, -0.0393].
- ridge:C->D:gold_margin: 0.2992 [0.2427, 0.3543].
- mlp:A->C:r2: 0.0748 [0.0693, 0.0805].
- mlp:A->C:normalized_mse: -0.0748 [-0.0805, -0.0693].
- mlp:A->C:kl_true_predicted: -0.2170 [-0.2471, -0.1878].
- mlp:A->C:js: -0.0279 [-0.0308, -0.0251].
- mlp:A->C:top1_agreement: 0.0405 [0.0346, 0.0462].
- mlp:A->C:top5_overlap: 0.0383 [0.0350, 0.0416].
- mlp:A->C:top10_overlap: 0.0338 [0.0310, 0.0366].
- mlp:A->C:gold_log_probability: 0.2162 [0.1863, 0.2477].
- mlp:A->C:gold_probability: 0.0390 [0.0349, 0.0431].
- mlp:A->C:gold_rank: -0.6473 [-0.8181, -0.4740].
- mlp:A->C:gold_margin: 0.5097 [0.4281, 0.5920].
- mlp:C->D:r2: -0.0665 [-0.0704, -0.0624].
- mlp:C->D:normalized_mse: 0.0665 [0.0624, 0.0704].
- mlp:C->D:kl_true_predicted: 0.0244 [0.0065, 0.0420].
- mlp:C->D:js: 0.0061 [0.0038, 0.0085].
- mlp:C->D:top1_agreement: -0.0102 [-0.0158, -0.0048].
- mlp:C->D:top5_overlap: -0.0176 [-0.0214, -0.0137].
- mlp:C->D:top10_overlap: -0.0239 [-0.0270, -0.0209].
- mlp:C->D:gold_log_probability: -0.0242 [-0.0439, -0.0036].
- mlp:C->D:gold_probability: -0.0112 [-0.0149, -0.0076].
- mlp:C->D:gold_rank: 0.0307 [-0.0741, 0.1303].
- mlp:C->D:gold_margin: -0.2262 [-0.2911, -0.1624].
- ridge:C_shuffle->ridge:C:r2: 0.0715 [0.0660, 0.0773].
- ridge:C_shuffle->ridge:C:normalized_mse: -0.0715 [-0.0773, -0.0660].
- ridge:C_shuffle->ridge:C:kl_true_predicted: -0.1771 [-0.2044, -0.1497].
- ridge:C_shuffle->ridge:C:js: -0.0247 [-0.0274, -0.0219].
- ridge:C_shuffle->ridge:C:top1_agreement: 0.0368 [0.0298, 0.0435].
- ridge:C_shuffle->ridge:C:top5_overlap: 0.0346 [0.0308, 0.0381].
- ridge:C_shuffle->ridge:C:top10_overlap: 0.0302 [0.0271, 0.0332].
- ridge:C_shuffle->ridge:C:gold_log_probability: 0.1719 [0.1435, 0.2004].
- ridge:C_shuffle->ridge:C:gold_probability: 0.0336 [0.0293, 0.0379].
- ridge:C_shuffle->ridge:C:gold_rank: -0.5574 [-0.7802, -0.3443].
- ridge:C_shuffle->ridge:C:gold_margin: 0.3427 [0.2873, 0.3969].
- mlp:C_shuffle->mlp:C:r2: 0.0891 [0.0835, 0.0947].
- mlp:C_shuffle->mlp:C:normalized_mse: -0.0891 [-0.0947, -0.0835].
- mlp:C_shuffle->mlp:C:kl_true_predicted: -0.2814 [-0.3109, -0.2519].
- mlp:C_shuffle->mlp:C:js: -0.0356 [-0.0388, -0.0323].
- mlp:C_shuffle->mlp:C:top1_agreement: 0.0529 [0.0467, 0.0595].
- mlp:C_shuffle->mlp:C:top5_overlap: 0.0423 [0.0383, 0.0461].
- mlp:C_shuffle->mlp:C:top10_overlap: 0.0393 [0.0359, 0.0426].
- mlp:C_shuffle->mlp:C:gold_log_probability: 0.2775 [0.2492, 0.3069].
- mlp:C_shuffle->mlp:C:gold_probability: 0.0493 [0.0446, 0.0540].
- mlp:C_shuffle->mlp:C:gold_rank: -0.8059 [-1.0219, -0.5950].
- mlp:C_shuffle->mlp:C:gold_margin: 0.6599 [0.6037, 0.7180].
- ridge:A->mlp:A:r2: 0.0737 [0.0705, 0.0771].
- ridge:A->mlp:A:normalized_mse: -0.0737 [-0.0771, -0.0705].
- ridge:A->mlp:A:kl_true_predicted: -0.1744 [-0.1971, -0.1512].
- ridge:A->mlp:A:js: -0.0338 [-0.0363, -0.0312].
- ridge:A->mlp:A:top1_agreement: 0.0501 [0.0424, 0.0581].
- ridge:A->mlp:A:top5_overlap: 0.0229 [0.0198, 0.0257].
- ridge:A->mlp:A:top10_overlap: 0.0266 [0.0241, 0.0290].
- ridge:A->mlp:A:gold_log_probability: 0.1751 [0.1495, 0.1997].
- ridge:A->mlp:A:gold_probability: 0.0599 [0.0560, 0.0640].
- ridge:A->mlp:A:gold_rank: -0.4945 [-0.6579, -0.3260].
- ridge:A->mlp:A:gold_margin: 1.0898 [1.0411, 1.1414].
- ridge:C->mlp:C:r2: 0.1043 [0.1009, 0.1079].
- ridge:C->mlp:C:normalized_mse: -0.1043 [-0.1079, -0.1009].
- ridge:C->mlp:C:kl_true_predicted: -0.2885 [-0.3147, -0.2634].
- ridge:C->mlp:C:js: -0.0463 [-0.0492, -0.0435].
- ridge:C->mlp:C:top1_agreement: 0.0686 [0.0620, 0.0757].
- ridge:C->mlp:C:top5_overlap: 0.0373 [0.0340, 0.0406].
- ridge:C->mlp:C:top10_overlap: 0.0405 [0.0378, 0.0434].
- ridge:C->mlp:C:gold_log_probability: 0.2954 [0.2691, 0.3237].
- ridge:C->mlp:C:gold_probability: 0.0797 [0.0751, 0.0845].
- ridge:C->mlp:C:gold_rank: -0.6912 [-0.8795, -0.5201].
- ridge:C->mlp:C:gold_margin: 1.4045 [1.3264, 1.4818].
- ridge:D->mlp:D:r2: 0.0892 [0.0859, 0.0924].
- ridge:D->mlp:D:normalized_mse: -0.0892 [-0.0924, -0.0859].
- ridge:D->mlp:D:kl_true_predicted: -0.2108 [-0.2330, -0.1879].
- ridge:D->mlp:D:js: -0.0298 [-0.0323, -0.0273].
- ridge:D->mlp:D:top1_agreement: 0.0474 [0.0412, 0.0538].
- ridge:D->mlp:D:top5_overlap: 0.0262 [0.0215, 0.0308].
- ridge:D->mlp:D:top10_overlap: 0.0300 [0.0243, 0.0359].
- ridge:D->mlp:D:gold_log_probability: 0.2142 [0.1899, 0.2387].
- ridge:D->mlp:D:gold_probability: 0.0485 [0.0445, 0.0525].
- ridge:D->mlp:D:gold_rank: -0.4460 [-0.5893, -0.3102].
- ridge:D->mlp:D:gold_margin: 0.8791 [0.8348, 0.9271].

The bootstrap uses 4,000 paired draws, resampling reactions jointly across checkpoints and checkpoint indices jointly across reactions. Inference across training seeds remains limited by having only two checkpoints. Conditional quantities use each reaction's SSE ratio first.

All KL, JS, top-1/5/10, teacher probability/log-probability/rank/margin, latent cosine, centered cosine, SSE/SST, and per-reaction results are retained in metrics_*.json and table.csv. Actual-state decoder references are in the ridge metrics files. Primary decoder metrics use the unchanged BF16 lm_head[:native_vocab] (392 outputs), matching Report 02. The original 402-output metrics are retained separately as a full-head robustness check; their extra predictor/reserved tokens change the probability normalization.

Ridge: AdamW lr=.003, weight_decay=.001; residual MLP: frozen ridge plus Linear(input,128)-GELU-Linear(128,256), weight_decay=.0001. Both use batch size 512, at most 20 epochs, validation-only early stopping (patience 3, improvement 1e-8). Train sampling is reaction-balanced and capped at 16,384 positions. Initialization is explicitly seeded. No ChemFM weights are trained or tokens generated.

## Amendment checks

The test projection ceiling uses the true held-out target projected through the saved train-only PCA. It is the maximum full-state R2 attainable within that fixed affine target space, under the same reaction-balanced SSE/SST reduction; no PCA is fitted on test.

| Native seed | Test PCA ceiling R2 | Test projection nMSE |
|---|---:|---:|
| 533 | 0.984584 | 0.015416 |
| 917 | 0.985172 | 0.014828 |
| paired mean | 0.984878 | 0.015122 |

Eligible train/validation/test reactions: 640/192/192; positions: 23,702/7,000/7,287. Training uses the same 16,384 reaction-balanced positions in all arms and seeds. Confirmation chemical-pair overlap is zero. Cached input IDs, product positions, split assignments and all evaluated row identities match across checkpoints and arms.

The token shuffle is cross-reaction and approximately product-position/length matched, not exact. Diagnostics below use the actual capped training set and complete validation/test sets.

| Seed | Split | Self-reaction matches | Same-token fraction | Mean absolute position gap | Mean absolute length gap |
|---|---|---:|---:|---:|---:|
| 533 | train | 0 | 0.1390 | 3.056 | 18.135 |
| 533 | validation | 0 | 0.1414 | 3.661 | 19.042 |
| 533 | test | 0 | 0.1333 | 3.086 | 17.773 |
| 917 | train | 0 | 0.1390 | 3.056 | 18.135 |
| 917 | validation | 0 | 0.1414 | 3.661 | 19.042 |
| 917 | test | 0 | 0.1333 | 3.086 | 17.773 |

The capacity-matched nonlinear correct-minus-shuffled contrast is:

| Metric | Seed 533 | Seed 917 | Paired mean | Paired 95% CI |
|---|---:|---:|---:|---|
| r2 | 0.088615 | 0.089540 | 0.089077 | [0.083515, 0.094712] |
| normalized_mse | -0.088615 | -0.089540 | -0.089077 | [-0.094712, -0.083515] |
| kl_true_predicted | -0.284065 | -0.278663 | -0.281364 | [-0.310914, -0.251883] |
| js | -0.036604 | -0.034560 | -0.035582 | [-0.038840, -0.032349] |
| top1_agreement | 0.054864 | 0.051008 | 0.052936 | [0.046715, 0.059542] |
| top5_overlap | 0.043322 | 0.041239 | 0.042280 | [0.038316, 0.046127] |
| top10_overlap | 0.040573 | 0.038033 | 0.039303 | [0.035932, 0.042638] |
| gold_log_probability | 0.279845 | 0.275189 | 0.277517 | [0.249169, 0.306941] |
| gold_probability | 0.050506 | 0.048088 | 0.049297 | [0.044589, 0.054005] |
| gold_rank | -0.758228 | -0.853549 | -0.805888 | [-1.021892, -0.595046] |
| gold_margin | 0.648535 | 0.671218 | 0.659877 | [0.603743, 0.718032] |

The prior full-head decoder results remain robustness-only in `native_vocab_decoder_summary.json`; the main table and all primary comparisons now use 392 tokens. The maximum absolute native-versus-full cell-mean discrepancy is 0.0014426527 for JS and 0.0004338321 for top-1. All nine decoder metrics and both vocabulary summaries are retained, not just JS/top-1.

Compact committed audit artifacts: [summary](../../runs/oracle_transition/summary.json), [table](../../runs/oracle_transition/table.csv), [eligibility](../../runs/oracle_transition/eligibility.json), [PCA ceiling](../../runs/oracle_transition/pca_ceiling.json), [native/full decoder summaries](../../runs/oracle_transition/native_vocab_decoder_summary.json), [nonlinear shuffle contrasts](../../runs/oracle_transition/nonlinear_c_shuffle_summary.json), [seed 533 shuffle](../../runs/oracle_transition/shuffle_diagnostics_533.json), [seed 917 shuffle](../../runs/oracle_transition/shuffle_diagnostics_917.json), and [hash manifest](../../runs/oracle_transition/artifact_manifest.json). Hidden-state caches, fitted weights and full per-reaction metric files remain local and are not committed.
