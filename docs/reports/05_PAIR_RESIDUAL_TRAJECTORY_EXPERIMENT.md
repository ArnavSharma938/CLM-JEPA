# Persistent pair-specific residual JEPA trajectory

## Preregistered protocol

This section was fixed on 2026-08-27 before any residual-trajectory outcome
was generated. Results will be appended without changing the locked design.

### Scientific intervention

For one logical batch of 16 reactions, let

```text
g_true    = grad_LoRA MSE(source_EOS, correct_target_EOS)
g_shuffle = grad_LoRA MSE(source_EOS, matched_shuffled_target_EOS)
g_pair    = g_true - g_shuffle
```

On each step selected by the established 50% auxiliary cadence, training uses

```text
g_LoRA = g_NTP + 2 * g_pair
```

The factor two is the historical active coefficient whose cadence-adjusted
expectation is one. This is exactly the active-weighted residual measured in
the frozen contraction/NTP directional audit. SIGReg is absent because its
marginal statistic and gradient are invariant to target permutation, so it
cancels from the true-minus-shuffled auxiliary contrast. The scalar
`MSE_true - MSE_shuffle` is logged only as a generator of this gradient; it is
not treated as a new bounded JEPA objective.

The residual is applied only to the 308 LoRA A/B tensors (6,307,840
parameters), matching the scope in which the favorable held-out-NTP direction
was established. PEFT token-I/O `modules_to_save` receive ordinary NTP only.
Native NTP is computed in its own bit-matched forward/backward pass. Auxiliary
endpoint computation restores the post-NTP RNG state, so it cannot alter the
future native dropout stream. A deterministic within-logical-batch unequal-
target, token-length-matched derangement uses seed `1907 + zero-based global
step`.

### Matched trajectories

| Setting | Locked value |
|---|---|
| Conditions | ordinary native NTP; NTP plus pair-specific residual JEPA |
| Model | ChemFM-1B revision `f99dc2e89726539bb9cf31b2e2b4360650bac6a8` |
| Train / validation manifests | existing frozen USPTO-MIT 1,280-row pilot / first two rows of the frozen length-stratified 256 validation panel |
| Seeds | `533, 917, 1301` |
| Epochs / updates | 4 / 320 |
| Physical / accumulation / logical batch | 4 / 4 / 16 on one Thunder A6000 |
| Optimizer | fused AdamW, LR `1e-4`, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `0.01` |
| Scheduler | cosine, 5% warmup, minimum LR `1e-5` |
| Precision / attention | BF16 / SDPA without gradient checkpointing |
| Checkpoint | fixed epoch 4; no outcome-based selection |
| LoRA / vocabulary | maintained ChemFM LoRA and historical extended endpoint vocabulary |

The initialized trainable-state SHA-256 is recorded by each run and must match
within a seed. The train manifest, data-loader seed/order, serialization,
optimizer budget, and native stochastic stream are otherwise identical.

### Endpoint and diagnostics

The expensive endpoint is exactly the existing frozen 256-reaction manifest
`data/clm_jepa_uspto_mit_official_endpoint/prespecified_stage1_256.jsonl`
(SHA-256 `5b87bce1e75ed1ebf1a2a9091e0367aedaa8600a5621bc960352cf45b18e1865`).
Every seed/condition uses all five official R-SMILES views, beam width 10, ten
candidates per view, official canonicalization, and reciprocal-rank
aggregation. Generated exact aggregated top-1 is primary. Top-3/5/10,
candidate validity, and exact top-1 for each individual view are secondary.

Mechanism diagnostics are limited to:

- five-view teacher-forced target-token CE, correct-token logit margin, and
  teacher-forced correct-token rate on the same 256 identities;
- active-step residual/NTP LoRA gradient cosine and norm ratio over the
  trajectory;
- true and shuffled MSE, residual scalar, endpoint true/shuffle gradient
  cosine, and derangement length cost.
- maintained representation geometry/sensitivity statistics on the same fixed
  256 canonical reaction pairs.

Each seed receives a reaction-paired bootstrap interval and exact McNemar
comparison. Across seeds, report every seed, the mean paired effect and sample
SD, a seed-level t interval, and a crossed seed-by-reaction bootstrap interval.
The latter resamples both seeds and reaction identities and is descriptive
with only three seeds; pooled seed-reaction McNemar significance will not be
claimed.

### Verdict rule

- **PASS:** positive aggregated exact top-1 effect in all three
  seeds, positive mean effect, and the crossed paired 95% interval excludes
  zero.
- **FAIL:** the mean exact top-1 effect is nonpositive and no more than one
  seeds improve. This falsifies persistent utility under this controlled
  ChemFM-LoRA trajectory, not the existence of the previously measured local
  first-order direction.
- **INCONCLUSIVE:** all other outcomes, including a small positive mean with
  mixed seed directions or uncertainty spanning zero.

No coefficient, cadence, seed, checkpoint, training duration, evaluation
identity, or decoding change will be made after inspecting outcomes.

### Pre-outcome execution amendment

After local one- and two-update implementation smoke tests, but before any
complete trajectory or 256-reaction endpoint was run, the requested execution
hardware changed from the local RTX 4050 to one Thunder A6000 with 6 vCPUs,
100 GB storage, and no template. The physical batch was therefore frozen at
the repository's established A6000 setting of 4 with accumulation 4, keeping
the logical batch, sample order, optimizer-step count, and all scientific
settings unchanged. Gradient checkpointing was disabled, also matching that
established A6000 setting. Both conditions use this amended setting for all
three seeds; the earlier local smoke checkpoints are excluded from results.

Execution optimizations may be adopted only after a pre-endpoint parity gate.
Training candidates must preserve the configured logical batch and match all
trainable adapter tensors after a short fixed trajectory within the existing
numerical parity tolerance. Generation candidates must exactly match every raw
beam, canonical candidate, ranking, and exact flag against sequential batch-1
decoding. Known non-equivalent candidates (larger physical batches, batched
VJPs, and reduced endpoint paths) remain excluded even if faster.

Before any completed 256-reaction endpoint was inspected, the user directed a
second execution amendment to remove low-priority experiment sections and cut
runtime substantially. Seeds `2027` and `4099`, broad retrieval re-audits,
repeated standalone gradient audits, and exploratory ablations were removed.
The three paired seeds, all six official five-view generation endpoints,
paired uncertainty, native token diagnostics, trajectory interaction logs,
and fixed-256 representation diagnostics remain. Three-way concurrent
training is permitted only if the separate fixed short-trajectory benchmark
reproduces every adapter tensor bit-for-bit against the sequential reference.
This amendment was made without inspecting any complete fixed-256 outcome.

## Results

### Execution and provenance

The final implementation was tested with `84 passed, 1 skipped` and executed
on one NVIDIA RTX A6000 (48 GB), 6 vCPUs, PyTorch `2.3.0+cu121`, Transformers
`4.45.2`, and PEFT `0.13.2`. All six runs completed 320 optimizer updates from
paired initial trainable-state hashes. The official endpoint manifest hash was
`5b87bce1e75ed1ebf1a2a9091e0367aedaa8600a5621bc960352cf45b18e1865`.
The concurrency gate reproduced all 310 saved adapter tensors bit-for-bit in
three replicas (maximum error zero) and improved aggregate training throughput
3.28-fold. The four-worker beam evaluator reproduced every equivalence-panel
raw beam and gave a 2.13-fold endpoint speedup. From the first full trajectory
launch through the final diagnostic, the scientific execution took about 2 h
55 min; the resumed production driver took 2 h 29 min.

The reduced three-seed screen was also projected empirically on the local RTX
4050 from the implementation smoke timings: 9.3 s for an NTP update and an
expected 15.1 s per residual update at the locked 50% cadence, plus 190.6 s of
active beam evaluation per 24 reactions. Those measurements imply about 6.5 h
for six sequential trajectories, 3.4 h for six fixed-256 endpoints, and about
0.5 h for diagnostics and loading: approximately 10.5 h locally. The A6000
execution therefore cut the actual screen to under three hours without changing
the logical batches or gradients.

Raw predictions, teacher rows, final adapters, trajectory curves, representation
statistics, environment metadata, and parity benchmarks are preserved under
`runs/pair_residual/a6000/`. Optimizer-resume state and short benchmark adapter
copies were removed after hash validation; all six final epoch-4 adapters remain.

### Primary generated exact top-1

Each cell is `native -> residual (residual - native)` in percentage points.

| Seed | Five-view aggregate | View 1 | View 2 | View 3 | View 4 | View 5 |
|---:|---:|---:|---:|---:|---:|---:|
| 533 | 2.73 -> 3.12 (+0.39) | 2.34 -> 4.30 (+1.95) | 2.73 -> 2.73 (0.00) | 1.95 -> 3.12 (+1.17) | 1.56 -> 1.95 (+0.39) | 3.12 -> 2.34 (-0.78) |
| 917 | 2.34 -> 1.95 (-0.39) | 4.30 -> 2.73 (-1.56) | 4.30 -> 2.73 (-1.56) | 1.95 -> 1.95 (0.00) | 1.17 -> 1.56 (+0.39) | 1.95 -> 3.91 (+1.95) |
| 1301 | 5.08 -> 1.95 (-3.12) | 3.52 -> 2.73 (-0.78) | 3.12 -> 2.73 (-0.39) | 3.12 -> 2.34 (-0.78) | 2.73 -> 2.73 (0.00) | 3.52 -> 2.73 (-0.78) |
| Seed mean | 3.39 -> 2.34 (-1.04) | 3.39 -> 3.26 (-0.13) | 3.39 -> 2.73 (-0.65) | 2.34 -> 2.47 (+0.13) | 1.82 -> 2.08 (+0.26) | 2.86 -> 2.99 (+0.13) |

Native produced 26/768 exact aggregated top-1 predictions and residual-JEPA
18/768. Seed effects were `+0.39, -0.39, -3.12` pp: one positive and two
negative. The mean was `-1.04` pp, seed SD `1.85` pp, seed-level t 95% interval
`[-5.63, +3.54]` pp, and crossed seed-by-reaction bootstrap 95% interval
`[-3.52, +1.17]` pp. Thus the three-seed population uncertainty includes zero;
no across-seed significance is claimed. Seed 1301 alone had paired bootstrap
`[-5.86, -0.39]` pp and exact McNemar `p=0.0386`; the other two seed intervals
included zero. Every individual-view crossed interval also included zero. The
effect did not survive consistently across views.

### Mechanism diagnostics

The persistent residual did not retain the earlier favorable NTP effect.
Five-view token-weighted CE worsened in all seeds by `+0.01332`, `+0.00659`,
and `+0.00675`. With reactions as clusters, the mean CE change was `+0.01075`
with crossed 95% interval `[+0.00586, +0.01646]`. Mean correct-token margin
changed by `-0.0294` (`[-0.0741, +0.0385]`), and correct-token rate by
`-0.0423` pp (`[-0.190, +0.117]` pp). There is no favorable-token-metric/
generation disconnect here: both native token prediction and exact generation
were worse on average.

The trajectory logs directly locate the instability. Averaged across seeds,
the residual/NTP cosine moved from approximately `-0.004` in epoch 1 to
`-0.103` in epoch 4. The applied residual/NTP norm ratio grew from `0.34` to
`1.79`; the residual's counterfactual AdamW update effect grew from `0.13` to
`0.37` of the native update. Meanwhile true/shuffled endpoint-gradient cosine
fell from `0.55` to `0.39`, and residual/true-gradient norm grew from `0.98` to
`1.54`. AdamW preconditioning amplification fell from `0.56` to `0.26`, so the
optimizer damped rather than amplified the raw residual. One of 489 active
updates had an undefined adaptive-update cosine because of a zero norm; it is
retained as a non-finite count and excluded only from that descriptive mean.

Representation diagnostics confirm that the intervention was active and
strongly pairing-specific. Mean correct-minus-matched-shuffle cosine increased
from `0.0035` to `0.4321`, retrieval top-1 from `42.7%` to `80.9%`, and
necessary-component replacement sensitivity from `0.0048` to `0.3201`.
At the same time, mean source effective rank collapsed from `36.5` to `6.74`.
Thus the residual repeatedly created reaction-pair-sensitive, low-rank endpoint
geometry, but this substantial representational effect translated into worse
native token decisions and no generated exact top-1 benefit.

### Verdict

**FAIL.** The preregistered failure rule is met: mean generated exact top-1 is
nonpositive and only one of three seeds improves. This falsifies the hypothesis
that repeatedly applying the exact historical pair-specific residual—at the
locked coefficient, cadence, LoRA scope, data, and optimizer budget—improves
ChemFM forward-reaction generation. More specifically, it shows that the
previously favorable starting-point first-order direction does not integrate
into a useful optimizer trajectory: its relative magnitude grows and its NTP
alignment becomes negative as training progresses.

This result does not falsify the previously measured local one-step effect, the
existence of reaction-specific information, or the ability of the residual to
reshape representations; the present experiment confirms the latter two. It
also does not make claims about untested coefficients, optimizers, capacity, or
architectures. Those are outside this falsification and were not tuned after the
negative result.
