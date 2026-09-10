# Report 08: Frozen Objective-Coupling and Candidate-Target Diagnostics

## Scope and decision

These were frozen diagnostics only. No ChemFM parameter was updated, no
optimizer was constructed for ChemFM, no model was retrained, and no beam or
other generation was run. The only fitted parameters were equal-capacity
linear probes in Experiment B.

**Decision.** Experiment A identifies ordinary multi-objective interference as
a major credible failure mechanism: the faithful auxiliary gradient is much
larger than NTP in every LoRA group and its reaction-averaged direction opposes
NTP throughout most middle/late layers. The effect is not uniformly visible in
the global gradient because the embedding/head result is Native-seed dependent,
so interference is a major contributor rather than a complete explanation.
Experiment B finds no candidate with materially stronger decoder-behavioral
coupling than adjacent-state prediction. The present evidence therefore
supports **neither** a selective semantic JEPA objective nor a
cross-serialization objective.

## Experiment A: NTP versus faithful-NextLat gradients

### Frozen design

The locally retained Native rank-8 confirmation checkpoints (seeds 2027 and
3163) were crossed with both trained faithful-NextLat auxiliary heads (seeds 533
and 917), producing four robustness cells. This crossing is important because
the original Native rank-8 533/917 adapters used for earlier hidden caches are
no longer local. The Native checkpoints have SHA-256
`efc31efc...dff0c` and `7c924752...e1f54`; the auxiliary states have SHA-256
`f7aaccca...c983` and `d4fa2aa6...d87d7`.

Each cell used the identical eight physical batches of four rows (32 distinct
reaction groups) selected deterministically across the empirical sequence-length
distribution of the existing 1,280-row pilot training manifest. The manifest
SHA-256 is `b5900bc7...c8dba`; selected-group hash is
`3c8f6f97...cd593`. Gradients were obtained separately with
`torch.autograd.grad` from ordinary product NTP and from faithful auxiliary
SmoothL1 + teacher/student KL on the same forward graph. Only the 7,954,432
trainable ChemFM parameters were measured; auxiliary-predictor parameters were
excluded. Optimizer steps: zero.

For each group, `negative dot fraction` is
`sum(max(0,-g_ntp*g_aux))/sum(abs(g_ntp*g_aux))`. `Active sign conflict` is
the fraction of jointly nonzero scalar coordinates having opposite signs.
`NTP-direction retention` is the projection of `g_ntp + g_aux` onto `g_ntp`,
divided by the NTP self-projection: `1 + ||g_aux||/||g_ntp|| * cosine`.

| Native / auxiliary seed | group | NTP norm | auxiliary norm | ratio | cosine | negative dot fraction | active sign conflict | NTP-direction retention |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2027 / 533 | all | 1.378 | 2.771 | 2.01 | 0.011 | 0.489 | 0.522 | 1.02 |
| 2027 / 533 | embeddings + head | 1.252 | 2.142 | 1.71 | 0.040 | 0.464 | 0.490 | 1.07 |
| 2027 / 533 | attention LoRA | 0.309 | 1.110 | 3.60 | -0.043 | 0.540 | 0.503 | 0.85 |
| 2027 / 533 | MLP LoRA | 0.487 | 1.363 | 2.80 | -0.074 | 0.570 | 0.534 | 0.79 |
| 2027 / 917 | all | 1.378 | 3.060 | 2.22 | 0.006 | 0.494 | 0.532 | 1.01 |
| 2027 / 917 | embeddings + head | 1.252 | 2.374 | 1.90 | 0.037 | 0.467 | 0.489 | 1.07 |
| 2027 / 917 | attention LoRA | 0.309 | 1.230 | 3.98 | -0.048 | 0.545 | 0.515 | 0.81 |
| 2027 / 917 | MLP LoRA | 0.487 | 1.488 | 3.06 | -0.091 | 0.586 | 0.542 | 0.72 |
| 3163 / 533 | all | 1.312 | 2.736 | 2.09 | -0.317 | 0.757 | 0.532 | 0.34 |
| 3163 / 533 | embeddings + head | 1.228 | 2.192 | 1.78 | -0.399 | 0.797 | 0.493 | 0.29 |
| 3163 / 533 | attention LoRA | 0.261 | 1.089 | 4.17 | -0.085 | 0.578 | 0.530 | 0.65 |
| 3163 / 533 | MLP LoRA | 0.381 | 1.223 | 3.21 | -0.088 | 0.582 | 0.534 | 0.72 |
| 3163 / 917 | all | 1.312 | 3.093 | 2.36 | -0.288 | 0.742 | 0.535 | 0.32 |
| 3163 / 917 | embeddings + head | 1.228 | 2.520 | 2.05 | -0.351 | 0.775 | 0.498 | 0.28 |
| 3163 / 917 | attention LoRA | 0.261 | 1.205 | 4.62 | -0.105 | 0.597 | 0.531 | 0.52 |
| 3163 / 917 | MLP LoRA | 0.381 | 1.326 | 3.49 | -0.097 | 0.590 | 0.537 | 0.66 |

Across the 32 individual physical-batch cells, global cosine has mean -0.026,
median -0.003, and is negative in 16/32. Attention and MLP per-batch medians are
0.003 and -0.010. Thus arbitrary small batches are often near orthogonal; the
stronger conflict appears in the stable reaction-averaged component.

The layer result is more systematic than the global result. Mean cosine across
all four crossings is negative in 20/22 layers. The largest conflicts are layer
18 (-0.147; auxiliary/NTP ratio 5.58), layer 14 (-0.139; 2.68), layer 4
(-0.128; 2.47), layer 5 (-0.120; 2.53), layer 9 (-0.117; 3.12), and layer 16
(-0.111; 3.55). All four crossings are negative in each of these layers.
Layers 19--21 have smaller angular conflict but extreme norm ratios of 8.44,
15.71, and 27.31, so small adverse components can still dominate an NTP update.

**Interpretation.** The consistent LoRA opposition plus 2.8--4.6x group norm
ratios is large enough to remove 15--48% of the NTP-aligned attention update and
21--34% of the NTP-aligned MLP update under unit faithful weighting. For Native
3163, head conflict reduces the global NTP-direction projection to 28--29%.
This makes coupling/weighting a major mechanism. Native 2027's compatible head
and near-zero global cosine show that it is not a universal scalar explanation;
target mismatch or representation shaping may still account for residual
degradation.

## Experiment B: frozen candidate-target certificate

### Frozen design and causal contract

This experiment uses the existing pilot panel because it retains five actual
R-SMILES serializations per reaction, which the 1,024-reaction canonical cache
does not. One frozen final-post-norm extraction was made with Native rank-8 seed
2027; the resulting reusable cache contains 256 reactions x five views and has
SHA-256 `f663d550...94786`. No logits were generated autoregressively.

A stable hash split assigns 160/48/48 complete reactions to
train/validation/test (split hash `0d9b2433...d869`); no view or position crosses
reaction splits. View 0 supplies the causal input `h[t]`. Only product
transitions whose next token maps to an RDKit atom are included. The predictor
never receives a future token/state, alternate view, graph label, or target.

The three targets are:

1. adjacent: the view-0 `h[t+1]`;
2. semantic: the view-0 mean atom state over the next atom's radius-1 graph
   neighborhood, expanded by the smallest named functional-group SMARTS match
   containing that atom;
3. cross-serialization: the same canonical graph unit pooled separately in
   existing views 1--4, then averaged.

All targets use graph/atom correspondence rather than token spans. Every probe
is the same 2048-to-128 linear ridge (262,272 parameters), with train-only input
standardization and train-only rank-128 target PCA. Validation-only early
stopping is capped at 20 epochs. The common matrices contain 3,949/1,140/1,189
train/validation/test positions from 160/48/48 reactions.

| target | held-out R2 | normalized MSE | cosine | centered cosine | NTP-loss coupling | gold-rank coupling | gold-margin coupling |
|---|---:|---:|---:|---:|---:|---:|---:|
| adjacent state | 0.652 | 0.348 | 0.869 | 0.807 | 0.450 | 0.431 | 0.119 |
| semantic graph pool | 0.636 | 0.364 | 0.904 | 0.783 | 0.417 | 0.390 | 0.168 |
| cross-serialization graph pool | 0.559 | 0.441 | 0.899 | 0.725 | 0.462 | 0.400 | 0.244 |

Coupling is reaction-mean Spearman correlation, direction-adjusted so positive
always means lower target-prediction error accompanies better decoder behavior.
NTP-loss coupling is significant within every target (adjacent p=.00135,
semantic p=.00321, cross-serialization p=.00095), but the relevant comparison
is paired against adjacent state. Semantic-minus-adjacent NTP coupling is
-0.027 (reaction bootstrap 95% CI [-0.213, 0.157]); cross-serialization is
+0.018 [-0.278, 0.298]. Gold-rank differences are -0.050 [-0.242, 0.122] and
-0.040 [-0.286, 0.202]. Cross-serialization's larger point estimate for margin
coupling (+0.133) is uncertain [-0.148, 0.418].

**Interpretation.** The graph-pooled targets do generalize across held-out
reactions and, for the cross target, across four unseen serializations of each
object. That is not enough: neither has stronger held-out R2 than adjacent state,
and neither has a material or statistically resolved advantage in decoder
coupling. Choosing cross-serialization from its slightly higher NTP/margin point
estimates would be exactly the unsupported “pick the best latent statistic” move
this certificate was intended to prevent.

## Reproduction and assumptions

Run `scripts/run_frozen_objective_diagnostics.py extract`, then `certificate`,
then `gradient`. Existing completed artifacts are reused unless `--overwrite`
is passed for the certificate. Raw gradient tensors are reduced online; all
per-batch and per-layer metrics are retained in the four gradient JSON files.

Unavoidable assumptions are: (1) trained auxiliary heads from development seeds
533/917 are evaluated on later Native rank-8 endpoints because their matched
Native adapters are absent locally; the full 2x2 crossing guards against an
arbitrary pairing; (2) Experiment B anchors view 0 and averages views 1--4;
permuting the anchor was not added because the request disallowed new variants;
(3) unmapped USPTO chemistry prevents exact atom-mapped reaction-center labels,
so the semantic unit uses transparent RDKit graph neighborhoods plus the
repository's existing functional-group SMARTS; and (4) B is an atom-token
certificate, not a claim about syntax-token targets.

Primary artifacts: [runner](../../scripts/run_frozen_objective_diagnostics.py),
[B results](../../runs/frozen_objective_diagnostics/certificate/results.json),
[A 2027/533](../../runs/frozen_objective_diagnostics/gradient/seed_2027.json),
[A 2027/917](../../runs/frozen_objective_diagnostics/gradient/native_2027_aux_917.json),
[A 3163/533](../../runs/frozen_objective_diagnostics/gradient/native_3163_aux_533.json),
and [A 3163/917](../../runs/frozen_objective_diagnostics/gradient/seed_3163.json).
