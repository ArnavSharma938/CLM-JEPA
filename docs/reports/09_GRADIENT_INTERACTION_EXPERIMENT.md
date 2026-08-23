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
| External diagnostic / official generation panels | 256 / 1,280 frozen reactions |
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

The seven conditions are four raw weighted sums (`lambda_eff` 0.25, 0.5, 1,
2), then PCGrad, CAGrad, and Du auxiliary similarity at `lambda_eff=1`.
Following the explicit cost constraint, the retained instance executes the
matrix and frozen evaluations sequentially.  A completed `lambda_eff=1`
artifact from a briefly provisioned snapshot-identical A6000 was copied onto
the retained instance before that clone and the reusable snapshot were
deleted; all unfinished conditions were discarded and restarted sequentially.
Every condition still runs 320 serial optimizer updates and its own complete
frozen evaluations.  The full test suite passed on the retained instance
before its first optimizer step.

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

Each epoch-4 condition receives:

- source/target variance, effective rank, pair-center spread,
  mean-direction energy, cosine margins, retrieval/MRR, PCA spectrum, and
  top-two-PC residual retrieval;
- one-view 256-reaction beam-10 generation and per-reaction normalized target
  token CE/rank diagnostics against the frozen native panel;
- the unchanged 1,280-reaction, five-official-R-SMILES-view, beam-10 endpoint
  against the frozen native predictions;
- held-out LoRA-gradient audits at epochs 1, 2, and 4 for raw MSE, raw SIGReg,
  full active-weighted auxiliary, and the selected combiner.

## Results

The controlled A6000 matrix and frozen evaluations are in progress.  This
section will be replaced with final measured tables, paired statistics,
mechanism checks, and the autoregressive verdict after all artifacts are
retrieved and verified.
