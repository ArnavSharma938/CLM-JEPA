# JEPA–NTP gradient-interaction experiment

## Question

Does direct raw-endpoint MSE+SIGReg fail primarily because its auxiliary
gradient destructively interferes with ChemFM's next-token-prediction (NTP)
gradient?  The controlled test keeps the data, ChemFM checkpoint, LoRA
configuration, optimizer, schedule, logical batch, JEPA statistic, cadence,
and evaluation pipeline fixed.  It changes only the way the NTP and JEPA
gradients are combined on trainable LoRA parameters.

Representation geometry is diagnostic, not the success criterion.  A method
must improve autoregressive target-token CE, product rank, and generation.

## Primary-source implementation decisions

The implementations were checked against the primary papers and official
code before integration:

- [Yu et al., *Gradient Surgery for Multi-Task Learning*](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf)
  (NeurIPS 2020), and the [official PCGrad repository](https://github.com/tianheyu927/PCGrad)
  plus the [named PyTorch implementation](https://github.com/WeiChengTseng/Pytorch-PCGrad).
  The experiment
  uses the requested asymmetric form: if the JEPA and NTP gradients conflict,
  only the conflicting component of JEPA is removed.
- [Liu et al., *Conflict-Averse Gradient Descent for Multi-task Learning*](https://openreview.net/pdf?id=_61Qh8tULj_)
  (NeurIPS 2021), and the [official CAGrad repository](https://github.com/Cranial-XIX/CAGrad).
  The two-objective bounded dual uses
  the official `c=0.5`, `1e-4` numerical constants, and final `1/(1+c)`
  rescaling.
- [Du et al., *Adapting Auxiliary Losses Using Gradient Similarity*](https://www.gatsby.ucl.ac.uk/~balaji/CL-NeurIPS2018-adapt.pdf)
  (NeurIPS 2018).  The implemented weighted published rule is
  `g_N + max(0, cosine(g_N,g_J)) g_J`; this is not a project-designed cosine
  threshold or binary gate.

The implementation computes a three-scalar Gram matrix over the aligned LoRA
gradient tensors instead of allocating flattened multi-million-element
vectors.  This is algebraically identical to applying the published vector
formulas to a concatenated parameter gradient.

## Production cleanup

The active model/training path contains one unambiguous direct objective:

\[
L_{JEPA}=\operatorname{MSE}(h_s,h_t)
+ \frac{4(0.01)}{0.99}\operatorname{SIGReg}(\{h_s,h_t\}),
\]

\[
L=L_{NTP}+\lambda_{active}L_{JEPA}.
\]

The following were removed from `src/`:

- PCSF loss/statistic, reference caching, collation, VJP hooks, configuration,
  and metrics;
- projector construction, optimization, checkpointing, evaluation, and the
  projection-space active condition;
- experimental gradient hooks unrelated to the four reported combination
  rules.

Historical PCSF and projection scripts, reports, checkpoints, and artifacts
remain available.  The old projection-head definition was moved to
`scripts/historical_projection.py`; it is not imported by production code.
A case-insensitive active-path scan found no PCSF/projector/rho/reference-spread
terms under `src/`.

## Exact gradient construction

For each logical batch, one cadence draw is shared by all conditions.  On an
active step:

1. A no-gradient endpoint-only pass collects all 16 source and target EOS
   states without invoking the vocabulary projection.
2. Raw MSE and the exact 16-sample-per-view SIGReg statistic are evaluated on
   that complete logical batch.  Autograd obtains their endpoint VJPs.
3. The physical four-example chunks are replayed with their exact RNG states.
   Separate `torch.autograd.grad` calls accumulate
   `g_N = grad(L_NTP)` and `g_J = grad(L_JEPA)`.
4. Frozen parameters are never included.  The selected interaction rule is
   applied only to 308 LoRA A/B tensors (6,307,840 parameters), as required.
   ChemFM's two trainable PEFT `modules_to_save` token-I/O tensors retain the
   ordinary weighted-sum gradient.
5. The established global-norm clipping, fused AdamW update, and cosine
   schedule run unchanged.

The no-gradient statistics pass and gradient-bearing replay use the same
dropout RNG stream.  Checkpoints preserve Python, NumPy, CPU/CUDA, dataloader,
JEPA-cadence, and SIGReg-slice states.

`lambda_eff` is the cadence-adjusted expected coefficient used throughout the
earlier controlled reports.  With 50% JEPA activity,
`lambda_active=lambda_eff/0.5`; therefore the four controls have active-step
coefficients 0.5, 1, 2, and 4, respectively.  PCGrad, CAGrad, and auxiliary
similarity use the historical `lambda_eff=1` control, hence active coefficient
2.  Both the raw unweighted and applied weighted JEPA/NTP norm ratios are
logged so this convention cannot be mistaken for an unreported scale change.

## Frozen protocol

| Setting | Value |
|---|---|
| Model | ChemFM-1B |
| Dataset | USPTO-MIT synthesis |
| Train / internal training-time validation | 1,280 / 2 frozen rows, matching the historical controlled jobs |
| External diagnostic / official generation panels | 256 / 256 frozen reactions |
| Seed | 533 |
| LoRA / trainable model state | unchanged controlled configuration |
| Epochs / optimizer updates | 4 / 320 |
| Physical / accumulation / logical batch | 4 / 4 / 16 |
| JEPA readout | k=0 source and target EOS, final transformer state |
| JEPA cadence | Bernoulli 0.5, one draw per logical update |
| SIGReg | exact 16 samples/view, 1,024 slices, 17 knots on [0,3] |
| Optimizer | fused AdamW, lr 1e-4, betas (0.9,0.999), eps 1e-8, wd 0.01 |
| Scheduler | cosine with 5% warmup, min lr 1e-5 |
| Attention | SDPA |
| Hardware | one retained RTX A6000 48 GB / six-vCPU instance, pinned PyTorch 2.3.0+cu121 |

The seven training conditions are four raw weighted sums (`lambda_eff` 0.25,
0.5, 1, 2), then PCGrad, CAGrad, and Du auxiliary similarity at
`lambda_eff=1`. Following the explicit cost constraint, the retained instance
executes the matrix sequentially. A completed `lambda_eff=1`
artifact from a briefly provisioned snapshot-identical A6000 was copied onto
the retained instance before that clone and the reusable snapshot were
deleted; all unfinished conditions were discarded and restarted sequentially.
Every condition runs 320 serial optimizer updates. The full test suite passed
on the retained instance before its first optimizer step.

After all training and representation/gradient diagnostics completed, the user
explicitly narrowed the costly behavioral endpoint: do not run further weight
conditions and limit every retained official comparison to 256 reactions. The
behavioral set is therefore native, historical direct MSE+SIGReg,
`lambda_eff=0.25` as a low-weight control, PCGrad, CAGrad, and auxiliary
similarity. The 256 reactions are rows 0-255 of the already frozen
`prespecified_stage1_1280` manifest, selected without reference to any model
outcome. Existing native, direct, and lambda-0.25 predictions are identity-
checked and sliced to that exact order; the three gradient-interaction methods
are generated directly on the same manifest. The lambda-0.25 1,280-reaction
run completed just before this scope change and is retained as an out-of-scope
artifact, not mixed into the 256-reaction primary table. A lambda-0.5 official
run was stopped after 52 reactions and is likewise excluded. This user-directed
mid-execution scope revision reduces endpoint power and is reported explicitly.

## Verification and evaluation

Before full training:

- local and A6000 suites passed 84 tests with one intentional skip;
- the A6000 two-update physical-4 preflight exercised one inactive and one
  active step, peaked at 10.27 GB, and emitted finite diagnostics;
- its active PCGrad step observed cosine -0.4952, raw unweighted JEPA/NTP norm
  ratio 0.04112, and a 4.23% modification relative to the raw summed gradient;
- direct/streamed logical-batch objective and VJP tests remained green.

Representation evaluation avoids repeated tokenization inside a condition,
batches endpoint extraction, requests the backbone endpoint without the LM
vocabulary projection, vectorizes prototype/retrieval scoring into one matrix
multiply, and computes effective rank from the equivalent smaller sample Gram
eigenspectrum.  On a 256x2048 RTX 4050 benchmark it ran in 0.289 s versus
1.139 s (3.94x).  Retrieval outputs were exact; other scalar differences were
at most 5.7e-6, except effective rank's 1.07e-4 SVD/eigendecomposition
roundoff.

Official beam-10 generation was reprofiled after its original four-worker
path produced only about 0.11 complete five-view reactions/s early in the
first endpoint.  The optimization followed NVIDIA's
[profile-first guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#application-profiling),
[CUDA Graph guidance](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/best-practices.html),
and Hugging Face's description of the
[dynamic/static KV-cache tradeoff](https://github.com/huggingface/transformers/blob/main/docs/source/en/cache_explanation.md).
One warmed baseline view took 3.0835 s: 2.0575 s of model-forward CUDA work
and an upper bound of 1.0260 s between forwards.  CPU profiling exposed about
10,000 tiny graph replays per view plus scalar-synchronizing beam scoring.

The exact final evaluator therefore:

- keeps dynamic sequence-length views but appends into a double-buffered,
  preallocated KV cache instead of concatenating the full prefix each layer;
- transfers each beam candidate vector in one operation instead of calling
  `.item()` for each candidate;
- captures each decoder layer as two graphs around the only dynamic-shape
  operation (cache update/repeat and SDPA), preserving the original operation
  order inside both segments; and
- uses three independent batch-1 workers, the fastest exact configuration
  after three-versus-four tie-breaking.

The final warmed view took 1.6014 s (1.93x faster), with model-forward CUDA
time 1.0776 s (1.91x faster).  On the same 24-reaction parity panel used by
the prior endpoint optimization, three-worker steady-state throughput was
0.28181 reactions/s and end-to-end throughput was 0.25230 reactions/s;
mean GPU utilization was 76.7%.  Every ordered raw candidate list, canonical
candidate list, ranked list, and exact flag matched the original evaluator for
all 24 reactions, and the merged prediction SHA-256 was identical.  This is
1.82x the prior report's selected 0.15496 reactions/s benchmark rate.

Rejected candidates were not used: static cache, merged LoRA, left-padded
batch-2, and equal-length batch-2 all changed raw candidate lists; static
cache and merged LoRA were also slower on the frozen eight-reaction screen.
The final generation code path changes evaluation runtime only, not prompts,
beam semantics, candidate canonicalization, or scoring.

Training received a separate A6000 profile rather than assuming that decoder
optimizations transferred.  A synchronized 80-update epoch spent 419.69 s
(86.8%) in the separate NTP/JEPA forward-backward path, 58.92 s (12.2%) in the
logical-batch endpoint-statistics/VJP pass, 4.13 s (0.85%) in optimizer work,
and only 0.66 s (0.14%) loading data.  Thirty-one half-second samples averaged
35.4% GPU utilization while retaining 22.47 GB, with 77% peak utilization.
This rules out tokenization, loader workers, and host-to-device transfer as
meaningful remedies.  The one-active-update PyTorch trace also showed
`aten::_scaled_dot_product_efficient_attention`; SDPA was already selecting a
fused memory-efficient backend rather than mathematical attention.  This
matches NVIDIA's advice to identify launch gaps before applying graphs
([performance troubleshooting](https://docs.nvidia.com/dl-cuda-graph/latest/troubleshooting/performance-issues.html)).

Five-update, cadence-matched training candidates were then tested sequentially
and compared at every saved adapter tensor.  Removing five inadvertent
synchronization barriers per update was bit exact but throughput-neutral
(35.37 s synchronized versus 35.44 s asynchronous).  Reusing the first or last
gradient-bearing microbatch as one endpoint pass was also bit exact but gained
less than 1% (35.25/35.06 s), so neither reuse branch remains in production.
PyTorch's documented
[batched VJP](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html)
was materially faster (28.25 s, 1.252x), but changed adapter elements by as much
as 1.831e-4 and changed the tiny validation generation panel after only five
updates; it was rejected under the no-correctness-trade rule.  The production
trainer therefore keeps sequential objective VJPs, makes synchronization
conditional on explicit profiling, avoids a redundant preliminary write of all
308 LoRA gradients, and applies final coefficients with exact multi-tensor CUDA
operations.  No rejected VJP/reuse switch remains in the active trainer.

Each epoch-4 training condition receives:

- source/target variance, effective rank, pair-center spread,
  mean-direction energy, cosine margins, retrieval/MRR, PCA spectrum, and
  top-two-PC residual retrieval;
- held-out LoRA-gradient audits at epochs 1, 2, and 4 for raw MSE, raw SIGReg,
  full active-weighted auxiliary, and the selected combiner.

The retained behavioral conditions additionally receive one-view 256-reaction
beam-10 generation with per-reaction normalized target-token CE/rank and a
256-reaction, five-official-R-SMILES-view, beam-10 endpoint. Paired native and
direct-MSE+SIGReg comparisons use the identical ordered reaction identities.

## Results

All seven four-epoch training runs, all seven representation evaluations, and
all 21 held-out-gradient audits completed.  The cost-revised behavioral
endpoint also completed for every retained condition.  Each of the six
official prediction files contains 256 unique reactions in exact manifest
order (`panel_index` 0 through 255).  The consolidated, machine-readable table
is `runs/gradient_interaction/a6000/endpoint_256/summary.json`.

### Training-time gradient interaction

These statistics cover the same 172 JEPA-active optimizer updates in each
run.  `raw ratio` is the unweighted JEPA/NTP LoRA-gradient norm ratio;
`modification` is the final combiner's change relative to ordinary summed
gradients.

| Condition | Mean cosine | Conflict | Raw ratio | Mean modification | Mean Du gate |
|---|---:|---:|---:|---:|---:|
| lambda_eff 0.25 | +0.0035 | 54.1% | 0.0983 | 0.0% | - |
| lambda_eff 0.5 | +0.0047 | 64.5% | 0.0991 | 0.0% | - |
| lambda_eff 1.0 | -0.0264 | 80.8% | 0.0899 | 0.0% | - |
| lambda_eff 2.0 | -0.0377 | 83.7% | 0.0689 | 0.0% | - |
| PCGrad | -0.0300 | 82.6% | 0.0912 | 1.14% | - |
| CAGrad | -0.0706 | 86.6% | 0.0593 | 67.86% | - |
| Du auxiliary similarity | -0.0050 | 52.3% | 0.1188 | 21.63% | 0.0153 |

The apparent tension between frequent negative cosine and PCGrad's small
modification is real: the conflicting component was usually only a small
projection of JEPA onto NTP, even though its sign was negative.  PCGrad thus
changed the summed update by only 1.14% on average.  CAGrad was not a mild
filter: the original two-task rule changed it by 67.86%.  Du adaptation
accepted only a very small positively aligned JEPA component on average and
set it to zero on conflicting steps.

The independently replayed held-out NTP audit agreed that late auxiliary
pressure remained weakly adverse.  At epoch 4, the cosine triplets below are
`MSE / SIGReg / full auxiliary`; the last column is the active-weighted full
auxiliary norm relative to held-out NTP.

| Condition | Epoch-4 held-out cosines | Full-aux/NTP norm | Combiner change |
|---|---:|---:|---:|
| lambda_eff 0.25 | +0.020 / -0.015 / -0.011 | 0.058 | 0.0% |
| lambda_eff 0.5 | +0.013 / -0.014 / -0.011 | 0.085 | 0.0% |
| lambda_eff 1.0 | -0.014 / -0.039 / -0.048 | 0.186 | 0.0% |
| lambda_eff 2.0 | -0.327 / -0.022 / -0.092 | 0.202 | 0.0% |
| PCGrad | +0.039 / -0.032 / -0.019 | 0.164 | 0.31% |
| CAGrad | -0.040 / -0.144 / -0.145 | 0.106 | 69.26% |
| Du auxiliary similarity | +0.009 / -0.008 / -0.005 | 0.207 | 20.27% (gate 0) |

Across epochs 1, 2, and 4, SIGReg was negatively aligned in 19 of 21 audits;
the two positive measurements occurred only on the Du trajectory.  The full
auxiliary was negative in 20 of 21 audits.  This confirms interference exists,
but does not establish that removing its one-dimensional conflicting
projection is sufficient.

### Representation geometry

All entries use the same frozen 256-reaction diagnostic.  Variance and
pair-center spread are in raw final-transformer space.  `Mean energy` is the
source/target fraction in the mean direction; lower is less common-direction
dominated.  Retrieval is four-way raw pair retrieval.

| Condition | Variance S / T | Center spread | Eff. rank S / T | Mean energy S / T | Cosine margin | Retrieval |
|---|---:|---:|---:|---:|---:|---:|
| Native ChemFM | .02488 / .02233 | .11622 | 25.79 / 23.04 | .6238 / .6625 | .06211 | 40.6% |
| lambda_eff 0.25 | .00167 / .00201 | .03549 | 43.18 / 25.17 | .9766 / .9716 | .00792 | 74.2% |
| lambda_eff 0.5 | .00133 / .00338 | .03888 | 38.61 / 11.56 | .9788 / .9437 | .00870 | 73.8% |
| lambda_eff 1.0 | .00188 / .00192 | .03854 | 35.92 / 29.35 | .9605 / .9576 | .01881 | 85.2% |
| lambda_eff 2.0 | .00098 / .00082 | .02585 | 44.08 / 38.49 | .9672 / .9702 | .01237 | 85.2% |
| PCGrad | .00188 / .00197 | .03817 | 40.27 / 34.48 | .9617 / .9583 | .01813 | 84.0% |
| CAGrad | .00071 / .00064 | .02226 | 50.60 / 41.13 | .9498 / .9499 | .01751 | 82.4% |
| Du auxiliary similarity | .00166 / .00153 | .03342 | 39.54 / 33.11 | .9789 / .9804 | .00628 | 66.4% |

None preserved native geometry.  All trained representations lost roughly an
order of magnitude of raw variance and became strongly mean-direction
dominated.  Pair retrieval improved, but the raw cosine margin shrank.  CAGrad
contracted variance and pair-center spread most severely; Du filtering reduced
raw retrieval as it suppressed JEPA, yet did not recover native variance.
This is why geometry alone cannot support a success claim.

### One-view autoregressive behavior

This endpoint directly measures the ordinary decoder, normalized target-token
CE, and beam-10 product rank.  CE delta is relative to native, so positive is
worse.  The historical direct condition is the validated epoch-4 MSE+SIGReg
baseline from report 02 on the same 256 reactions.

| Condition | Top-1 | Top-3 | Top-5 | Top-10 | Valid candidates | Target-token CE | CE delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native | 2.34% | 10.16% | 15.63% | 20.31% | 78.20% | .240683 | - |
| Direct MSE+SIGReg | 2.34% | 9.38% | 13.28% | 19.14% | 86.60% | .248779 | +3.36% |
| PCGrad | 2.73% | 8.59% | 14.06% | 17.58% | 80.39% | .257099 | +6.82% |
| CAGrad | 2.34% | 5.86% | 12.11% | 19.14% | 86.76% | .295247 | +22.67% |
| Du auxiliary similarity | 1.17% | 8.98% | 13.67% | 18.75% | 86.76% | .243893 | +1.33% |

PCGrad made aggregate CE significantly worse: mean per-reaction native-minus-
PCGrad CE was -0.01681, bootstrap 95% CI [-0.02333,-0.01051], Wilcoxon
`p=6.44e-8`.  CAGrad was decisively adverse (-0.06015,
[-0.06650,-0.05405], `p=3.89e-39`).  Du filtering came closest to native CE:
-0.00168, [-0.00743,+0.00388], `p=0.962`; it reduced the historical direct
condition's CE damage but did not improve NTP over native.

### Official five-view generation

The table is the final frozen 256-reaction comparison.  `Delta` and the paired
bootstrap CI are against native top-1.  Exact McNemar tests use the same
reaction identities.

| Condition | Top-1 | Top-3 | Top-5 | Top-10 | Top-1 delta [95% CI] | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| Native | 4.30% | 19.53% | 26.95% | 39.06% | - | - |
| Historical direct MSE+SIGReg | 3.52% | 18.75% | 26.17% | 37.11% | -0.78 pp [-3.13,+1.56] | .754 |
| lambda_eff 0.25 | 1.95% | 15.23% | 24.22% | 36.72% | -2.34 pp [-4.69,-0.39] | .070 |
| PCGrad | 3.91% | 16.41% | 25.00% | 35.55% | -0.39 pp [-2.73,+1.95] | 1.000 |
| CAGrad | 2.34% | 13.67% | 25.78% | 35.94% | -1.95 pp [-4.30,0.00] | .125 |
| Du auxiliary similarity | 3.91% | 19.53% | 26.56% | 35.94% | -0.39 pp [-2.73,+1.95] | 1.000 |

PCGrad and Du were each +0.39 pp versus historical direct top-1, with the same
paired CI [-1.17,+1.95] pp and McNemar `p=1.0`; this is one reaction and not
evidence of improvement.  CAGrad was -1.17 pp versus direct (`p=.453`).  Du
matched native top-3, but remained below native at top-1, top-5, and top-10.
All trained conditions had 100% valid aggregated ranked candidates, so the
negative accuracy/CE conclusion is not an invalid-SMILES artifact.

The three newly generated official endpoints took 954.9-982.6 seconds each
(15.9-16.4 minutes), including model load.  End-to-end throughput was
0.2605-0.2681 reactions/s, mean GPU utilization 91.1-91.6%, and mean power
193.1-194.1 W.  Native, historical direct, and lambda-0.25 were identity-sliced
from completed exact five-view runs; they were not wastefully regenerated.

## Verdict

Gradient conflict is measurable, but it is not the dominant removable failure
under this integration.

1. Lower auxiliary strength did not rescue generation: the retained
   lambda-0.25 control was worse than both native and historical direct on the
   official panel.  Because the requested endpoint scope stopped the other
   weight generations, this is a low-weight control rather than a full
   behavioral dose-response curve.
2. PCGrad removed the formal negative projection on 82.6% conflicting updates,
   but changed the actual sum by only 1.14%; geometry remained contracted and
   target-token CE became more adverse than direct MSE+SIGReg.
3. CAGrad strongly altered the update, but produced the worst CE and strongest
   contraction.  Pareto-style conflict aversion did not help this objective
   pair.
4. Du's literature-standard auxiliary adaptation substantially suppressed
   JEPA and came closest to native CE, but did not beat native CE or official
   generation.  Its result is consistent with avoiding some harm by mostly
   declining the auxiliary, not preserving a useful JEPA gain.

Therefore none of the tested interaction rules succeeds.  The evidence points
away from simple over-weighting or a correctable conflicting-gradient
component and toward a deeper mismatch in where/how raw-endpoint JEPA is
integrated with autoregressive ChemFM.  The next experiment should change the
JEPA integration location (for example, the separately motivated disposable
projection space) rather than add another gradient-combination heuristic.
