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
| Seeds | `533, 917, 1301, 2027, 4099` |
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

Each seed receives a reaction-paired bootstrap interval and exact McNemar
comparison. Across seeds, report every seed, the mean paired effect and sample
SD, a seed-level t interval, and a crossed seed-by-reaction bootstrap interval.
The latter resamples both seeds and reaction identities and is descriptive
with only five seeds; pooled seed-reaction McNemar significance will not be
claimed.

### Verdict rule

- **PASS:** positive aggregated exact top-1 effect in at least four of five
  seeds, positive mean effect, and the crossed paired 95% interval excludes
  zero.
- **FAIL:** the mean exact top-1 effect is nonpositive and no more than two
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
five seeds; the earlier local smoke checkpoints are excluded from results.

Execution optimizations may be adopted only after a pre-endpoint parity gate.
Training candidates must preserve the configured logical batch and match all
trainable adapter tensors after a short fixed trajectory within the existing
numerical parity tolerance. Generation candidates must exactly match every raw
beam, canonical candidate, ranking, and exact flag against sequential batch-1
decoding. Known non-equivalent candidates (larger physical batches, batched
VJPs, and reduced endpoint paths) remain excluded even if faster.

## Results

Pending execution.
