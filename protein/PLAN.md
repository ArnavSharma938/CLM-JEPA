Implement the following calibration study end-to-end in the current repository. Treat this as an execution task, not a proposal or literature review. Inspect the existing codebase first, preserve its conventions where reasonable, and add only the training, evaluation, diagnostics, analysis, checkpointing, and reproducibility infrastructure actually necessary to execute this pilot. Do not silently substitute different models, datasets, metrics, budgets, hypotheses, or approximations. When a specified quantity cannot be computed exactly, implement the stated fallback/bound and record that limitation.

The goal is to determine why LoRA diverges from full fine-tuning during genuinely generative protein-language-model adaptation before introducing any new PEFT method. The calibration study distinguishes target-coverage limitations from four candidate LoRA-specific mechanisms:

* **H1 — capacity misallocation:** LoRA may have enough total low-rank capacity, but uniform rank allocation may place that capacity in the wrong modules.
* **H2 — adaptation-subspace choice:** the allowed rank may be sufficient, but vanilla LoRA may occupy poorly oriented row/column spaces relative to useful supervised adaptation directions.
* **H3 — structural insufficiency:** even with favorable allocation, orientation, and optimization, a pure low-rank update family at the specified budget may be unable to reach the required function.
* **H4 — factorized optimization pathology:** the desired effective update may be representable, but optimizing it as factors `BA` may depend pathologically on arbitrary equivalent factor coordinates.

These are different causal explanations. The experiment below is ordered to separate them rather than treating every LoRA failure as “insufficient rank.”

# 1. Model and task

Use released ESM-IF1 checkpoint:

`esm_if1_gvp4_t16_142M_UR50`

Report the true loaded parameter count programmatically:

```python
sum(p.numel() for p in model.parameters())
```

Do not trust inconsistent published parameter-count metadata.

Use the curated Megascale supervised fine-tuning setting associated with ProteinDPO. The collection is approximately 660k measured variants from ~403 protein domains after filtering and provides structurally separated train/validation/test domains.

Task: stability-directed, structure-conditioned autoregressive protein sequence generation.

Input:

* native protein backbone `X`

Target:

* experimentally measured stabilizing sequence variant `y`

Training subset:

* `ΔΔG < 0`
* exclude indels
* retain valid single- and double-substitution variants

Objective:

```math
L_SFT = -Σ_t log pθ(y_t | X, y_<t)
```

Do not introduce preference optimization in this calibration study.

# 2. Structural OOD variable

Use query-normalized FoldSeek TM-score.

For each validation/test backbone `x`:

```math
S_max(x) = max_{z ∈ D_train} TM_FoldSeek^(q)(x,z)
d_struct(x) = 1 - S_max(x)
```

`d_struct` is the sole primary OOD axis.

Keep it continuous for statistical analysis.

For visualization and fixed subset construction only, split test domains into equal-sized near/middle/far tertiles.

Domain analyses must control for:

* sequence length
* mutation count

Do not vary decoding parameters, evaluation methods, or sample counts across OOD tiers.

# 3. Core adaptation conditions

Implement:

1. Base

   * frozen pretrained ESM-IF1

2. Full FT

   * all ESM-IF1 parameters trainable

3. Dense-target FT / DTFT

   * every entry of exactly the matrices targeted by LoRA trainable
   * all other parameters frozen

4. LoRA-low

   * rank `r=8`
   * `alpha=8`

5. LoRA-high

   * rank `r=64`
   * `alpha=64`

Use:

```text
LoRA dropout = 0
adapter weight decay = 0
alpha / r = 1
```

Standard LoRA initialization:

* `A`: random/Kaiming
* `B = 0`

## LoRA targets

Target every dense transformer matrix.

Encoder:

* Q
* K
* V
* O
* FC1
* FC2

Decoder self-attention:

* Q
* K
* V
* O

Decoder cross-attention:

* Q
* K
* V
* O

Decoder FFN:

* FC1
* FC2

Do NOT target:

* GVP layers
* embeddings
* output head
* layer norms

This complete transformer-matrix coverage is deliberate; do not reduce it to a Q/V-only configuration.

Programmatically derive parameter counts from matrix shapes and verify expected counts are approximately:

```text
LoRA r=8:    1,441,792 trainable scalars
LoRA r=64:  11,534,336 trainable scalars
DTFT:       58,720,256 trainable dense weights
```

Abort with an informative error if architecture-derived counts materially disagree.

# 4. Budget definitions

Primary adaptation budget:

```text
trainable scalar count
```

For conventional LoRA matrix `m × n` at rank `r`:

```math
B_train = r(m+n)
```

Also report intrinsic manifold dimension:

```math
B_intrinsic = r(m+n-r)
```

For sparse methods, report mask/index storage separately.

Always distinguish:

1. trainable scalar count
2. intrinsic dimensionality
3. serialized storage

Never use “same parameters,” “same degrees of freedom,” and “same storage” interchangeably.

# 5. Training protocol

Use:

* effective batch size: 32
* coordinate noise: 0.1 Å
* bf16
* maximum training budget: 30 epochs
* validation every 0.5 epoch
* paired data ordering within seeds
* no truncation of valid Megascale proteins

30 epochs is a cap, not an assumed convergence point.

Primary model for each run:

* checkpoint with minimum validation autoregressive NLL

## Learning-rate HPO

Use HPO seed:

`7`

Do NOT use seed 7 for uncertainty estimates.

FT/DTFT LR candidates:

```text
3e-8
1e-7
3e-7
```

LoRA LR candidates:

```text
1e-6
3e-6
1e-5
3e-5
1e-4
```

Select LR by validation autoregressive NLL.

Do not tune rank.

Evidence seeds:

```text
11, 23, 37
```

Only after a mechanism survives the pilot, add confirmation seeds:

```text
53, 71
```

Save:

* epoch 0
* epoch 0.25
* epoch 1
* epoch 3
* epoch 10
* best-validation checkpoint
* terminal checkpoint if run reaches epoch 30

Early checkpoints are intentionally oversampled for H2/H4 dynamics.

# 6. Primary gap decomposition and coverage gate

For a performance metric `P`, define:

```math
G_total    = P_FT   - P_LoRA
G_coverage = P_FT   - P_DTFT
G_LoRA     = P_DTFT - P_LoRA
```

Only `G_LoRA` cleanly isolates limitations of the low-rank parameterization/optimization conditional on target coverage.

The first mechanistic decision is whether target coverage dominates.

If FT→DTFT explains more than 50% of total FT→LoRA gap on BOTH:

1. the primary held-out likelihood/task outcome
2. the primary cheap functional outcome

then stop H1–H4 attribution and report:

`target coverage is currently the dominant adaptation bottleneck`

The target set must then be reconsidered before deeper LoRA-mechanism claims.

Also, if LoRA ≈ DTFT, there is no substantive LoRA-specific failure to diagnose.

# 7. Broad evaluation

Across the complete held-out set compute:

* autoregressive NLL
* normalized native-vs-variant likelihood
* experimental-stability ranking
* stabilizing-vs-destabilizing classification
* OOD-stratified metrics

For every test variant:

```math
Δlog p =
log p(y_variant | X)
-
log p(y_native | X)
```

Define experimental fitness direction:

```math
f = -ΔΔG
```

Primary experimental-stability outcome:

```math
ρ(Δlog p, -ΔΔG)
```

Compute per domain and summarize appropriately across domains.

Also report:

* held-out NLL
* sequence recovery
* stabilizing-vs-destabilizing AUROC
* likelihood calibration across stability bins

# 8. Fixed diagnostic and generation panels

Before adaptation, select performance-independent panels.

Train diagnostic panel:

* 48 training domains
* stratified by sequence length/domain size

Test diagnostic panel:

* 48 test domains
* 16 near
* 16 middle
* 16 far

For each domain select where available:

* at most 8 stabilizing variants
* at most 8 destabilizing variants

Expensive generation subset:

* exactly 18 test backbones
* 6 near
* 6 middle
* 6 far

Select these using ONLY:

* structural distance
* sequence length

Never use model performance to choose the panel.

The 18 generation backbones must be a fixed subset of the 48-domain test diagnostic panel.

# 9. Generation evaluation

At each run's best checkpoint generate:

```text
16 sequences / backbone / model / seed
```

At epochs 1 and 10 generate:

```text
4 sequences / backbone / model / seed
```

Use:

* ancestral sampling
* temperature = 0.5
* no top-k
* no top-p
* identical paired RNG seeds
* no post-generation filtering

Expensive metrics:

## Rosetta energetic score

```math
ΔE = E_generated - E_native
```

Report:

* median `ΔE`
* fraction with `ΔE < 0`

Call this only:

`in-silico energetic improvement`

Do not call it experimentally validated stability.

## Structural consistency

* ESMFold TM-score to conditioning backbone
* mean pLDDT

## Novelty

* maximum sequence identity to training sequences

## Diversity

* pairwise identity
* unique fraction
* sequence-cluster count

Do not collapse these metrics into a composite score.

# 10. H1 — capacity misallocation across modules

**Question being tested:** assuming LoRA has a fixed total trainable-scalar budget, does performance depend materially on *where* rank capacity is placed across transformer matrices? H1 is specifically about allocation of a fixed low-rank budget, not about whether low rank itself is sufficient.

Let module ranks be:

```math
r = (r_1,...,r_L)
```

and LoRA trainable-scalar cost:

```math
C(r) = Σ_l r_l(m_l+n_l)
```

H1 means that, holding total trainable-scalar budget and update family fixed, different allocation vectors produce systematically different attainable task performance because uniform allocation places capacity poorly.

Do NOT assume each module has a context-independent scalar “rank demand.”

Rank value may be:

* contextual
* non-additive
* dependent on ranks elsewhere

## H1-A: exact DTFT reconstruction allocation

This first H1 diagnostic asks how rank would be allocated if the sole objective were reconstructing the dense DTFT update as accurately as possible at a fixed LoRA parameter budget.

For each targeted DTFT matrix:

```math
ΔW_l^D = W_l,DTFT - W_l,0
```

Let singular values be:

```math
σ_l,1 ≥ σ_l,2 ≥ ...
```

For local rank `k`:

```math
value_l,k = Σ_{j=1}^k σ_l,j²
cost_l,k  = k(m_l+n_l)
```

Solve exactly:

```math
max_{k_1,...,k_L} Σ_l value_l,k_l
```

subject to:

```math
Σ_l cost_l,k_l ≤ B
```

Use exact dynamic programming/discrete knapsack.

Do NOT greedily sort by energy-per-parameter and call it globally optimal.

This solution is optimal ONLY for Frobenius reconstruction of DTFT. It is not a functional optimum. Failure of this allocation to improve task performance does NOT falsify H1.

## H1-B: contextual equal-budget functional rank swaps

The actual causal H1 test asks whether moving rank capacity from one module to another, while preserving the total trainable-scalar budget exactly, changes attainable task performance.

Do not define a context-independent module value `V_i(+ΔB)`.

Instead evaluate explicit reallocations:

```math
ΔP_{i←j}
=
P(r + δ_i - δ_j) - P(r)
```

with exact budget equality:

```math
C(r + δ_i - δ_j) = C(r)
```

Same-shape modules:

If:

```math
m_i+n_i = m_j+n_j
```

transfer rank directly:

```math
r_i → r_i+k
r_j → r_j-k
```

Different-shape modules:

```math
c_i = m_i+n_i
c_j = m_j+n_j
```

smallest exact exchange:

```math
a_i = c_j / gcd(c_i,c_j)
a_j = c_i / gcd(c_i,c_j)
```

so:

```math
a_i c_i = a_j c_j
```

If the minimum exact exchange is too large to remain scientifically local or violates rank bounds, do NOT approximate the budget. Compare complete preregistered allocation vectors with exactly equal `C(r)` instead.

At minimum compare:

1. uniform allocation
2. exact DTFT-reconstruction allocation
3. shape-matched shuffled allocation controls
4. preregistered diagnostic-guided capacity-transfer allocations

Estimate contextual swap matrix:

```math
S_ij = ΔP_{i←j}
```

If:

```math
S_ij ≠ -S_ji
```

or effects change under different baseline allocations, report interactions instead of forcing an additive rank-demand model.

## H1 optimization control

Functional allocation comparisons must not be confounded by H4.

Do NOT use vanilla-LoRA Adam as the sole optimizer for H1 causal comparisons.

Use the same strong factorization-controlled fixed-rank optimization procedure used in H3 constrained searches.

All allocation vectors must receive:

* same supervised objective
* same data
* same optimization-step budget
* same restart count
* same validation-selection rule
* zero factor weight decay

The objective is attainable function under each allocation, not optimizer accidents.

## H1 spectral predictors

These diagnostics ask whether properties of the dense supervised update/gradient identify modules where additional rank is actually useful.

Per module/checkpoint measure:

* stable rank
* effective rank
* r90
* r95
* r99
* spectral entropy

Treat these only as candidate predictors of contextual swap benefit, not direct evidence of misallocation.

## H1 adaptive spectrum correctness

Start randomized SVD at:

`k=128`

Compute independently:

```math
||G||_F²
```

Captured energy:

```math
E_k = Σ_{i=1}^k σ_i² / ||G||_F²
```

If required, increase:

```text
128 → 256 → 512
```

where dimensions permit.

If the threshold remains unresolved, report bounds such as:

```text
r95 > 128
r99 > 256
```

Do not invent exact ranks.

Report full spectral entropy only when:

* full spectrum is available, or
* a justified tail estimator/bound is implemented

## H1 decision logic

H1 is weakened if:

* exact-budget swaps have negligible effects
* complete nonuniform vectors do not materially outperform uniform
* effects lack reproducible structure across seeds
* diagnostics do not predict contextual swap effects

Strong H1 evidence requires:

```text
diagnostic
→ predicted beneficial equal-budget allocation change
→ observed reproducible functional improvement
```

Failure of the SVD allocation alone does not falsify H1.

# 11. H2 — adaptation-subspace choice

**Question being tested:** for a fixed rank allocation, is LoRA's representational capacity adequate but pointed in the wrong directions? H2 concerns the orientation of the low-dimensional row/column spaces used by LoRA, not how much rank each module receives.

H2: for a fixed rank allocation, a useful low-dimensional adaptation geometry exists, but vanilla LoRA occupies poorly oriented row/column spaces relative to supervised adaptation directions.

Keep distinct from:

* H1: wrong capacity quantity by module
* H3: no sufficient low-rank function
* H4: representable directions but factor optimization fails

## H2-A: initialization geometry

At standard LoRA initialization, `B=0`, so only changes to `B` can produce a first-order effective weight update. Therefore the initial row space of random `A` restricts which gradient directions are immediately reachable.

Standard LoRA begins with:

```math
B = 0
BA = 0
```

The ordinary smooth fixed-rank manifold tangent formula does not apply at initialization.

First-order effective update:

```math
d(BA) = dB A
```

Reachable first-order space:

```math
T_0(A) = {dB A}
```

Let `P_A` project onto row space of `A`.

Actual initial gradient capture:

```math
C_init(A;G) =
||G P_A||_F² / ||G||_F²
```

Exact best rank-r initial row-space ceiling:

```math
C_init,r* =
Σ_{i=1}^{min(r,rank G)} σ_i(G)²
/
||G||_F²
```

Define:

```math
D_init = C_init,r* - C_init
```

## H2-B: mature tangent geometry

After both LoRA factors become full numerical rank, compare the actual local tangent directions available to the trained LoRA state with the mathematically best tangent capture possible at the same rank.

Once both factors have numerical rank `r`, define:

```math
U = col(B)
V = row(A)
```

Tangent projection:

```math
P_T(G)
=
UUᵀG + GVVᵀ - UUᵀGVVᵀ
```

Actual capture:

```math
C_T(G) =
||P_T(G)||_F² / ||G||_F²
```

Exact capacity-matched rank-r tangent ceiling:

```math
C_T,r* =
Σ_{i=1}^{min(2r,rank G)} σ_i(G)²
/
||G||_F²
```

Use this exact top-`2r` singular-energy ceiling. Do not perform unnecessary numerical Grassmann optimization.

Define:

```math
D_orientation = C_T,r* - C_T
```

This is the primary mature H2 diagnostic.

## H2-C: longitudinal geometry

Track:

* `D_init` at initialization
* `D_orientation` after full numerical rank
* principal angles between LoRA row/column spaces and supervised-gradient singular spaces
* principal angles to DTFT update spaces
* rate of orientation improvement

Interpret separately:

Initialization problem:

* high `D_init`
* mature geometry later becomes good

Persistent subspace problem:

* high mature `D_orientation`
* predicts poor adaptation

Structural problem:

* even best rank-constrained function is insufficient

Optimization problem:

* useful directions are representable but normal factor training fails

## H2-D: spectrum-matched aligned initialization

This is the causal H2 intervention: rotate only the initial right-singular orientation of `A` toward useful supervised gradient directions while holding the entire singular spectrum and zero initial effective update fixed.

Only if initial orientation deficit predicts failure, run the causal initialization intervention.

Given vanilla `A`:

```math
A = U_A Σ_A V_Aᵀ
```

Let `V_G,r` contain top `r` right singular vectors of the supervised gradient.

Construct:

```math
A_aligned = U_A Σ_A V_G,rᵀ
B_aligned = 0
```

This must preserve exactly:

* singular values of A
* Frobenius norm
* spectral norm
* rank
* zero initial effective update

Only the right-singular orientation changes.

## H2 decision logic

H2 is not a major final-performance limitation if:

* poor initial orientation rapidly disappears
* mature tangent capture approaches exact `C_T,r*`
* aligned initialization changes only early convergence
* final task/generative function is unchanged

Strong evidence requires:

* orientation deficit predicts failure
* orientation-only intervention selectively rescues high-deficit cases

# 12. H3 — structural insufficiency

**Question being tested:** after controlling for target coverage, rank allocation, subspace orientation, and strong optimization, does a sufficiently good function actually exist inside the allowed low-rank update family? H3 is therefore about the representational structure of low-rank updates themselves.

H3 asks whether the pure low-rank update family itself is insufficient at the specified trainable-scalar budget, even with favorable allocation, orientation, and strong optimization.

The rank constraint:

```math
rank(BA) ≤ r
```

is exact.

Its functional importance is not.

There is no tractable global oracle for this nonlinear neural-network problem, so the experiment can strongly falsify structural insufficiency by finding a good low-rank solution, but failed searches cannot mathematically prove that none exists.

## H3-A: post-hoc SVD existence falsifier

This first test asks whether simply compressing the successful dense-target update into the allowed low-rank family already yields a high-performing low-rank solution.

Using:

```math
ΔW^D = W_DTFT - W_0
```

construct each matrix's best allowed low-rank Frobenius approximation by truncated SVD.

This solves:

```math
min_rank(ΔW_hat)≤r
||ΔW^D - ΔW_hat||_F
```

It does NOT solve:

```math
min_rank(ΔW_hat)≤r
L_task(W_0 + ΔW_hat)
```

Interpret one-sidedly.

If SVD compression preserves nearly all DTFT functionality, a sufficient low-rank solution demonstrably exists and H3 is strongly falsified.

If SVD compression fails, no strong conclusion follows.

## H3-B: strong function-space rank-constrained search

If SVD compression fails, directly optimize within the fixed-rank function family with much stronger search than ordinary LoRA training. The purpose is to search for any high-performing low-rank function, not necessarily one close to DTFT in weight space.

Do NOT call this an oracle.

Call it:

`strong function-space rank-constrained search`

Use multiple complementary initializations:

1. truncated-SVD DTFT initialization
2. random balanced low-rank initialization
3. gradient-informed initialization

Use:

* substantially larger optimization budget than ordinary LoRA
* multiple restarts
* zero factor weight decay
* balanced/retracted fixed-rank optimization OR direct Riemannian fixed-rank optimization
* validation-based search selection

Primary objective:

```math
L_search = L_task
```

Do not require imitation of DTFT.

A low-rank solution that solves the adaptation task differently from DTFT still falsifies H3.

Secondary search route only:

```math
L =
L_task
+
λ KL(p_DTFT || p_LR)
```

This is distillation-assisted search, not the definition of low-rank success.

## H3 interpretation asymmetry

Maintain:

```text
finding a sufficient low-rank solution
⇒ strong falsification of H3
```

but NEVER:

```text
failure to find one
⇒ proof that no good low-rank solution exists
```

Failed strong search provides only empirical evidence consistent with structural insufficiency.

## H3-C: alternative update structures

If low-rank search remains consistently inferior, test whether other update structures achieve better function under the same trainable-scalar budget. This helps determine whether any limitation is specifically associated with pure low-rank structure rather than merely insufficient parameter count.

Only if strong low-rank searches remain inferior, compare under equal trainable-scalar budget:

1. low-rank
2. sparse
3. low-rank + sparse

For hybrid updates preregister fraction of trainable scalar budget assigned to low-rank factors:

```text
q ∈ {0.25, 0.5, 0.75}
```

Sparse mask/index metadata is excluded from primary trainable-scalar budget but reported separately as serialized storage.

Search fairness:

* same supervised objective
* same examples
* same maximum gradient evaluations
* same restart count
* same validation-selection budget
* comparable HPO budget

If sparse methods require extra discrete mask search, explicitly report the additional computational burden.

Do not claim one structural family is intrinsically superior merely because it received stronger optimization/search.

## H3 functional evaluation

Weight-reconstruction error:

```math
E_W =
||ΔW^D - ΔW_hat||_F
/
||ΔW^D||_F
```

This is descriptive.

Primary outcomes are functional:

* held-out NLL
* stability ranking
* generated-protein outcomes

Optionally report token-distribution divergence from DTFT descriptively.

DTFT behavioral similarity is NOT itself the objective.

## H3 decision logic

Strong falsification:

* find a carefully validated rank-constrained solution retaining roughly `≥90%` of DTFT's useful task/generative improvement

Empirical support requires convergence of evidence:

1. SVD compression fails
2. multiple strong task-loss low-rank searches fail
3. failure reproduces across seeds
4. matched-budget alternative structures consistently perform substantially better under comparable search effort

Even then do not treat search failure as proof of nonexistence.

# 13. H4 — factorized optimization pathology

**Question being tested:** does LoRA optimization depend on the arbitrary factorization coordinates `A,B`, even when two factorizations represent exactly the same effective weight update `BA`? If equivalent parameterizations subsequently train differently under otherwise identical conditions, the optimizer is sensitive to coordinates that have no functional meaning.

LoRA represents:

```math
ΔW = BA
```

For invertible:

```math
R ∈ R^{r×r}
```

the transformation:

```math
A' = RA
B' = BR^{-1}
```

preserves:

```math
B'A' = BA
```

Thus the represented model is exactly unchanged.

H4 asks whether ordinary optimization nevertheless depends on arbitrary factor coordinates.

Adapter weight decay must remain exactly 0 because factor-wise decay is itself not invariant to equivalent rescaling.

## H4-A: equivalent-factor continuation

At epochs 1 and 10, create mathematically equivalent LoRA factorizations representing the exact same effective update, then continue each under identical future training conditions.

At epochs 1 and 10, branch from exactly the same LoRA state.

Primary branches:

1. Original:

```text
(A,B)
```

2. Orthogonal transformation:
   choose deterministic orthogonal `R`:

```math
RᵀR = I
```

then:

```math
A' = RA
B' = BRᵀ
```

3. Invertible scaling:
   choose diagonal `R=diag(s_1,...,s_r)` with:

```math
κ(R) = 16
```

while remaining safely nonsingular, then:

```math
A' = RA
B' = BR^{-1}
```

All branches must begin with numerically identical:

* BA
* model logits
* loss
* representational capacity

## H4-B: balanced factorization

Do not zero-pad a rank-deficient SVD factorization and call it an equivalent coordinate transformation.

If:

```math
rank(BA) < r
```

skip balanced branch.

If:

```math
rank(BA) = r
```

numerically, balanced representation may be included only if obtained through an invertible transformation of the existing factors:

```math
A_bal = RA
B_bal = BR^{-1}
```

Verify:

```math
||B_bal A_bal - BA||
```

is at floating-point tolerance.

Balanced branch is secondary.

Primary H4 tests are orthogonal and invertible-scaling transformations.

## H4-C: controlled continuation

For every branch:

* reset optimizer state
* same LR
* same minibatch sequence
* same RNG state
* same dropout realization
* zero weight decay
* same number of steps

Continue exactly:

```text
500 optimizer steps
```

Measure:

* `||ΔW_t^(a)-ΔW_t^(b)||_F`
* effective-update cosine similarity
* NLL
* task outcome

If large divergence develops, run generation on the small fixed generation panel.

## H4-D: causal rescue

If coordinate-equivalent branches diverge reproducibly under Adam, test whether an optimizer designed to be invariant to LoRA factor transformations removes that sensitivity.

Only if equivalent coordinates produce large reproducible divergence, run a transformation-invariant optimizer such as RITE as the matched intervention.

The intended causal sequence is:

```text
coordinate sensitivity under Adam
→ sensitivity substantially removed by invariant optimization
```

This is more informative than simply comparing final Adam and RITE performance.

## 14. Cross-model adaptation and weight-space analysis

In addition to testing whether LoRA underperforms dense adaptation, characterize **how LoRA and dense fine-tuning alter each protein language model internally**, including cases where their downstream performance is similar.

For every model and adaptation condition, analyze the geometry of the learned update and the resulting weights using:

* singular-value spectra, effective rank, and spectral entropy of \(\Delta W\);
* layerwise distribution of update magnitude and spectral energy;
* principal-angle/subspace overlap between LoRA, DTFT, and full-FT updates;
* alignment of each learned update with the dominant gradient and dense-update directions already measured under H1–H3;
* changes in the singular spectrum of the final adapted weights \(W+\Delta W\), including whether adaptation introduces new dominant singular directions relative to the pretrained model;
* functional agreement between methods on the same examples, so similar predictions can be distinguished from similar internal adaptation.

Treat **functional similarity and weight-space similarity as separate outcomes**. In particular, do not infer that LoRA and dense fine-tuning learn equivalent solutions merely because their task performance is similar.

When the study is extended across PLMs, apply these diagnostics consistently across model architectures and scales. Use them to determine whether architecture, parameter scale, or adaptation regime changes:

1. the intrinsic rank and compressibility of useful fine-tuning updates;
2. where adaptation capacity is required across the network;
3. which subspaces LoRA can access relative to dense fine-tuning;
4. whether similar downstream behavior is reached through substantially different weight-space solutions.

These analyses extend H1–H4 from explaining a performance gap to characterizing the broader **adaptation geometry of protein language models**, including regimes where no substantial LoRA performance deficit is observed.


# 15. Statistical hierarchy and valid units of evidence

The highest-level independent replication is training run/seed.

Do not treat modules, checkpoints, domains, or generated sequences as independent method replicates.

Generation hierarchy:

```text
training run
→ backbone
→ generated sequence
```

Longitudinal hierarchy:

```text
training run
→ checkpoint
→ backbone
→ generated sequence
```

Module hierarchy:

```text
training run
→ module
```

Checkpoints are repeated observations from a trajectory.

Modules are clustered within runs.

Generated sequences are nested outcomes, not independent method replicates.

With only 3 evidence seeds:

* run-level predictive regression is exploratory only
* do not claim convincing predictive generalization from three runs

Richer primary predictive analyses should use quantities with genuine within-run variation.

Examples:

H1:

* does a module-level diagnostic predict contextual `ΔP_{i←j}`?

H2:

* does module/group `D_init` predict benefit of spectrum-matched aligned initialization?

OOD:

* does a genuinely domain-specific diagnostic predict that domain's LoRA–DTFT gap as `d_struct` increases?

Never copy one run-level diagnostic across many domains and treat that as domain-level evidence.

Use hierarchical/mixed or cluster-robust analyses where appropriate.

# 16. Instrumentation

At every saved checkpoint:

## FT/DTFT

Store:

* target weights
* `ΔW = W_t - W_0`
* validation metrics

## LoRA

Store:

* A
* B
* merged BA
* scaling
* optimizer states

## Gradients

On the fixed diagnostic panel calculate per target module:

* mean gradient
* Frobenius norm
* spectral norm
* adaptive singular spectrum
* microbatch variability

Do not store enormous per-example full-gradient tensors.

## Adaptive spectra

Start:

* `k=128`

Compute:

* top-k singular values
* independent `||G||_F²`
* captured energy

Increase to:

* 256
* 512

as needed and feasible.

Report inequalities when unresolved.

Do not infer exact full-spectrum entropy from truncated unresolved spectra.

## Activations

Maintain streaming top-64 input principal components per target module.

Do not store raw token activations.

Treat activation PCs as secondary/exploratory.

## Domain/example metadata

Store:

* domain
* sequence ID
* ΔΔG
* mutation count
* sequence length
* max query-normalized FoldSeek TM-score to training
* `d_struct`
* token losses
* whole-sequence likelihood
* normalized native/variant likelihood

## Generation metadata

Store:

* generated sequence
* conditioning backbone
* model
* training run/seed
* checkpoint
* generation RNG seed
* decoding configuration
* Rosetta score
* ESMFold pLDDT
* TM-score
* nearest-training sequence identity
* diversity metrics

# 17. Preregistered decision rules

## Gate 0: target coverage

If FT→DTFT accounts for >50% of FT→LoRA gap on BOTH primary cheap outcomes:

```text
STOP H1–H4 attribution
```

until target coverage is redesigned.

## H1 support

Support H1 only if contextual exact-budget reallocation gives reproducible functional gains and preregistered diagnostics predict which reallocations help.

Reject H1 as a major mechanism if performance is approximately insensitive to exact-budget rank placement.

Failure of the Frobenius/SVD allocation alone is explicitly non-decisive.

## H2 support

Support H2 if:

* actual reachable/tangent capture is substantially below mathematically exact capacity-matched ceiling
* deficit predicts adaptation failure
* orientation-only intervention selectively helps high-deficit cases

Reject as major mechanism if orientation deficits disappear during training and orientation controls do not change final function.

## H3

Strong falsification:

* sufficient rank-constrained solution found

Empirical support:

* repeated strong low-rank searches fail while matched-budget alternatives succeed

Never infer proof of nonexistence from failed optimization.

## H4 support

Support H4 if invertibly equivalent LoRA coordinates follow reproducibly different future trajectories and invariant optimization suppresses that difference.

Reject as major mechanism if equivalent coordinate systems remain geometrically/functionally indistinguishable under controlled continuation.

# 18. Exact execution order

Follow this order because later tests depend on earlier identifiability gates.

Stage 1 — establish genuine adaptation gap

* Base
* FT
* DTFT
* LoRA-8
* LoRA-64

If LoRA ≈ DTFT, stop LoRA-specific mechanism diagnosis.

Stage 2 — target-coverage gate

* compare FT−DTFT against DTFT−LoRA
* if coverage dominates, stop H1–H4

Stage 3 — H3 SVD existence falsifier

* compress DTFT into appropriate rank family
* success strongly falsifies H3
* failure is non-decisive

Stage 4 — H1 reconstruction allocation

* exact discrete SVD-energy knapsack
* characterize nonuniform DTFT spectra
* never call this functional optimum

Stage 5 — H1 causal functional reallocations

* exact equal-budget swaps / complete allocation vectors
* strong factorization-controlled optimizer
* determine whether placement causally matters

Stage 6 — H2 geometry

* calculate `D_init`
* calculate mature `D_orientation`
* run spectrum-matched aligned-A intervention only if indicated

Stage 7 — H3 strong constrained search

* only if SVD compression did not already falsify H3
* optimize fixed-rank task performance directly
* interpret search failure cautiously

Stage 8 — H3 alternative structures

* only after repeated low-rank inferiority
* sparse and low-rank+sparse
* equal scalar count and comparable search effort

Stage 9 — H4 equivalent-coordinate test

* orthogonal and invertible-scaling continuations
* if positive, invariant-optimizer rescue

Stage 10 — OOD and temporal analysis

* only after candidate mechanisms survive causal tests
* test whether severity increases with structural OOD
* whether early diagnostics predict later failure
* how mechanism severity evolves during training

# 19. Compute plan

Primary trained adaptation runs:

```text
FT
DTFT
LoRA-8
LoRA-64
```

across seeds:

```text
11, 23, 37
```

Total:

```text
12 primary adaptation runs
```

Base requires no adaptation training.

HPO uses separate seed 7.

Most mechanistic tests should reuse:

* saved checkpoints
* gradients
* weight deltas
* short continuation branches

Restrict expensive generation to the predetermined 18-backbone subset.

Only mechanisms that survive pilot causal tests receive:

* seeds 53 and 71
* larger intervention studies

# 20. Pilot implementation requirements

Implement only the infrastructure required to run this calibration study cleanly in the current repository. Reuse existing dataset loading, model wrappers, training loops, evaluation utilities, checkpoint infrastructure, external-tool integrations, and logging wherever they already exist; do not build production-grade preprocessing, orchestration, generalized frameworks, or abstractions that are unnecessary for this pilot.

Add the minimum new components needed for the specified experiments, including as applicable:

* configuration for Base/FT/DTFT/LoRA-8/LoRA-64
* correct LoRA matrix targeting and DTFT freezing
* parameter-count assertions
* LR sweeps and evidence-seed runs
* required checkpoint capture
* held-out likelihood/stability evaluation
* fixed diagnostic and generation panels
* spectrum/SVD utilities
* exact H1 knapsack and equal-budget allocation tests
* H2 initialization/tangent diagnostics and aligned initialization
* H3 SVD compression and strong fixed-rank search
* sparse/hybrid comparisons only if the H3 decision sequence reaches them
* H4 equivalent-factor continuations and invariant-optimizer branch only if triggered
* the logging needed to recover all quantities specified above

Do not implement later-stage branches before they are scientifically triggered if doing so would add substantial unnecessary work. It is acceptable for expensive external evaluations such as FoldSeek, Rosetta, or ESMFold to use simple scripts/interfaces around existing installations or separately generated input/output files rather than a generalized integration layer.

Add lightweight correctness checks for the mechanistically important invariants, especially:

* targeted/trainable parameter sets and expected counts
* DTFT targeting exactly the same dense matrices as LoRA
* exact equality of H1 trainable-scalar budgets
* H2 formula correctness on small numerical cases
* preservation of `A`'s singular spectrum in aligned initialization
* preservation of `BA`, logits, and loss under H4 equivalent transformations
* deterministic paired RNG where required
* spectrum-energy accounting

Store results in a structured format sufficient for later scientific analysis. Provide concise commands/documentation for running the pilot stages that are actually implemented; do not turn this into a production experiment-management project.

# 21. Results report

At the end, produce a purely scientific and objective report of the observed results, measurements, comparisons, uncertainty, and any experimental limitations. Do not provide an overall interpretation, narrative conclusion, mechanistic verdict, or speculative explanation beyond what is directly established by the measured results and preregistered statistical comparisons.


**NO EDITING OF THIS PLAN.MD IS PERMITTED UNLESS EXPLICITLY AUTHORIZED**