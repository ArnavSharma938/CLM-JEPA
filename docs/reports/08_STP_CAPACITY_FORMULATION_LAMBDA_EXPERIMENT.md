# STP capacity, formulation, and coefficient experiment

## Scope and provenance

This report completes the frozen protocol in
`08_STP_CAPACITY_FORMULATION_LAMBDA_PREREGISTRATION.md`. The upstream executable
reference remains `galilai-group/llm-jepa` commit
`ea0017c654ad917066ff32afc88276bea8ca5f7e`; released STP means its validated
`--linear=random_span` patch-versus-complement objective. Paper STP is the
separately named literal three-index objective
`1 - cos(h_r - h_s, h_t - h_r)`, with the same framing-excluded ChemFM semantic
path, one sample per example, final representation layer, FP32 cosine, and
symmetric gradients.

The frozen 512-reaction endpoint is the first 512 rows of the existing 1,280
endpoint and has SHA-256
`a2e6202a4abaf9a70f4700e04299a09964d38c10fce004022dc43e759aa6057d`.
Its first 256 reactions are byte-for-byte the earlier official-STP endpoint.
All new trajectories use seeds 533 and 917, four epochs/320 optimizer updates,
the frozen 1,280-row training set, effective batch size 16, fused AdamW and the
unchanged native ChemFM NTP task. The endpoint is five R-SMILES views, beam 10,
ten returned candidates per view, canonicalization, and reciprocal-rank
aggregation.

## Implementation and performance audit

LoRA rank and alpha are now explicit configuration values. The two capacity
conditions are rank/alpha 8/8 and 128/128, with unchanged dropout, target
modules, and `modules_to_save`. Adapter metadata records all reconstruction
fields; evaluation reads them automatically and falls back to 8/8 for legacy
checkpoints without the new run metadata.

Released STP remains a separately named condition. Its sampled spans,
transition construction, loss, gradients, and parameter updates are covered by
upstream parity and reference-versus-optimized tests. Paper STP has independent
definition tests and a non-degeneracy test proving the two objectives differ
away from a straight trajectory.

The completed local verification suite passes 99 tests with one expected skip.
It covers rank/alpha propagation, legacy rank-8 reconstruction, released-STP
numerical invariance, the literal paper equation, framing-excluded path
construction, and objective non-equivalence.

The material new training bottleneck was retaining every transformer layer via
`output_hidden_states=True` although both STP objectives use only the final
post-RMSNorm representation. A forward hook now captures that tensor directly.
On the A6000 rank-8 equivalence batch (746 tokens), reference and optimized
sampled spans, total/native/STP losses, and all trainable gradients were exactly
equal. Mean step time fell from 0.655873 to 0.580892 seconds (1.129x), while
peak allocated CUDA memory fell from 4,818,235,904 to 4,757,803,520 bytes and
throughput rose from 1,137.4 to 1,284.2 tokens/second. At rank 128, the same
values were exactly equal; time fell from 0.781460 to 0.766025 seconds (1.020x),
peak allocation fell from 5,245,267,456 to 5,185,033,728 bytes, and throughput
rose from 954.6 to 973.9 tokens/second. These are end-to-end forward/loss/
backward update benchmarks, not isolated cosine kernels.

Final-checkpoint-only writing is used for new runs because intermediate
checkpoints are not scientific endpoints. A four-step rank-128 preflight took
21.13 seconds wall/10.25 seconds measured training time and peaked at
6.47 GB allocated VRAM. One checkpoint write took 4.48 seconds and occupied
about 1.2 GB (778 MB optimizer/training state plus 389 MB adapter), so omitting
three unused writes saves about 13.45 seconds and 3.6 GB per trajectory without
changing an update.

The production rank-128 native seed pair then completed concurrently in
554.58 seconds wall, with 84.45% mean sampled GPU utilization, 17,603 MiB peak
device occupancy, and 20.11% mean host CPU utilization. The individual
trajectories took 509.30 and 527.84 seconds and processed 1,228.9 and 1,185.8
effective tokens/second respectively.

The already optimized evaluator was profiled rather than rewritten. Four
workers sustain roughly 86--95% mean GPU utilization and 100% peaks. In the
first uncached rank-128 production endpoint, model load was only 8.4--8.7
seconds per worker versus 1,853--1,962 seconds of generation, and overall mean
GPU utilization was 95.15%. A persistent-base/adapter-swap design therefore has
less than roughly 0.5%
available end-to-end benefit while adding synchronization and adapter-state
risk, and was rejected. Five/six worker variants had already been rejected by
the repository's measured worker sweep. Rank-128 generic-versus-optimized
evaluation is separately gated on identical ordered predictions for 24
reactions before its production result is accepted.

Other candidate rewrites were rejected after inspection/profiling. Native NTP
requires logits for the same backward pass, so suppressing logits is not an
exact option. Immutable panel construction and completed prediction rows are
already cached and resumed, leaving tokenization/canonicalization far below
generation time. The released per-example sampler is small relative to the
transformer update; replacing it with a batched sampler would jeopardize exact
RNG consumption without a material wall-time opportunity. Static worker tail
imbalance was only about 4--6%, insufficient to justify dynamic scheduling
that would alter execution order. Thus final-state capture and final-only
checkpointing are the only newly retained training changes; no new evaluation
change cleared the benefit/equivalence bar.

## Stage A: expanded rank-8 released-STP endpoint

No model was retrained. The six existing checkpoints were extended from 256 to
512 reactions by resuming their exact worker partitions; completed prediction
rows were reused verbatim. The mean treatment effect remains positive but is
not seed-consistent.

| Seed | Native exact top-1 | Released STP exact top-1 | Difference | Paired bootstrap 95% CI | Exact McNemar p |
|---:|---:|---:|---:|---:|---:|
| 533 | 13/512 (2.54%) | 24/512 (4.69%) | +2.15 pp | [+0.78,+3.71] pp | 0.0074 |
| 917 | 11/512 (2.15%) | 16/512 (3.12%) | +0.98 pp | [-0.20,+2.34] pp | 0.2266 |
| 1301 | 20/512 (3.91%) | 14/512 (2.73%) | -1.17 pp | [-2.34,-0.20] pp | 0.0703 |
| Seed mean | 2.86% | 3.52% | +0.65 pp | [-1.17,+2.54] pp (two-way) | -- |

The five view-specific treatment effects (views 1--5) were respectively:

| Seed | View 1 | View 2 | View 3 | View 4 | View 5 |
|---:|---:|---:|---:|---:|---:|
| 533 | +0.39 pp | +0.98 pp | +1.17 pp | +1.95 pp | +1.56 pp |
| 917 | +0.20 pp | +0.59 pp | +1.95 pp | +0.98 pp | +1.17 pp |
| 1301 | +0.39 pp | 0.00 pp | -0.78 pp | 0.00 pp | +0.59 pp |

Teacher-forced token-weighted CE improved in every seed by -0.00383, -0.00255,
and -0.00474. Correct-token rate changed by +0.147, +0.134, and +0.056
percentage points. Mean correct-token margin changed by +0.2290, +0.3317, and
-0.0085. Seed 1301 therefore again supplies the important disconnect: its CE
and correct-token rate improve while generated exact top-1 worsens. Expanding
the endpoint from 256 to 512 does not turn the released rank-8 result into a
seed-consistent generation benefit.

The failure is specifically top-1 ordering, not complete loss of the gold
product from all returned candidates. Aggregate exact top-10 changes by +1.37,
+4.49, and +0.59 pp in the three seeds even though seed 1301 top-1 falls.
Per-view final gold-beam-survival changes are mixed, and the native/STP raw
top-1 sequence first differs at median generated token 1 in every seed (31.2%,
43.6%, and 30.7% of reaction-view pairs change). These are final-beam
comparisons rather than online pruning traces, but they localize much of the
trajectory sensitivity to very early decoding and downstream ranking.

## Stage B: rank-128 capacity test

Two paired native/released-STP trajectories were trained at rank/alpha 128/128.
The within-rank treatment effect was smaller than at rank 8 and was not
seed-consistent.

| Seed | Native exact top-1 | Released STP exact top-1 | Difference | Paired bootstrap 95% CI | Exact McNemar p |
|---:|---:|---:|---:|---:|---:|
| 533 | 46/512 (8.98%) | 44/512 (8.59%) | -0.39 pp | [-2.34,+1.56] pp | 0.8450 |
| 917 | 45/512 (8.79%) | 52/512 (10.16%) | +1.37 pp | [-0.78,+3.52] pp | 0.2810 |
| Seed mean | 8.89% | 9.38% | +0.49 pp | [-1.76,+2.73] pp (two-way) | -- |

For the same two seeds, the existing rank-8 released-STP effect was +2.15 and
+0.98 pp, mean +1.56 pp with two-way 95% CI [+0.20,+3.12] pp. The rank-128
minus rank-8 difference in mean treatment effect is therefore -1.07 pp. Rank
128 does make both native and STP models stronger in absolute terms, but that
is not the question: it does not increase the released-STP treatment effect.
The frozen selector required a mean improvement of at least 0.5 pp and two
nonnegative rank-128 effects, so it selected rank 8.

The view-specific rank-128 treatment effects were:

| Seed | View 1 | View 2 | View 3 | View 4 | View 5 |
|---:|---:|---:|---:|---:|---:|
| 533 | -0.20 pp | +0.20 pp | -0.98 pp | -0.20 pp | -0.20 pp |
| 917 | +0.98 pp | +1.76 pp | +0.20 pp | +1.37 pp | +1.56 pp |

Teacher-forced CE still improved in both seeds (-0.00641 and -0.00353), but
correct-token rate changed by -0.020 and +0.106 pp and mean margin fell by
0.206 and 0.177. Exact top-10 changed by +0.59 and -1.76 pp. Thus the
rank-128 generation inconsistency is accompanied by weaker token-decision
evidence than at rank 8; extra LoRA capacity did not rescue STP.

The rank-128 generic-versus-fast-path gate compared 24 reactions and found
identical reaction identities, exact flags, raw candidates per view,
canonical candidates per view, and ordered aggregate candidates. The first
full endpoint took 1,975.84 seconds wall (1,961.84 active), sustained 95.15%
mean GPU utilization, and peaked at 17,972 MiB device occupancy.

## Stage C: released versus paper-equation STP

The frozen rank choice was 8. Before training, 32 frozen training reactions
were evaluated in batches of four with all trainable LoRA and
`modules_to_save` gradients. The same nominal coefficient does **not** imply
the same auxiliary pressure:

| Objective | STP loss | Raw STP/native norm ratio | lambda-weighted ratio | STP/native gradient cosine | Span fraction/loss Pearson r |
|---|---:|---:|---:|---:|---:|
| Released patch/complement | 1.4735 | 2.252 | 0.0450 | -0.397 | -0.119 |
| Paper three-index | 1.4136 | 0.388 | 0.00776 | +0.384 | -0.065 |

These are means across eight frozen batches. The paper objective exerts about
5.8 times less lambda-weighted gradient pressure at lambda=0.02 and is more
NTP-aligned on these batches. Per the preregistration, it was nevertheless
trained at the same published coefficient rather than post hoc norm-matched.

| Seed | Native exact top-1 | Paper STP exact top-1 | Difference | Paired bootstrap 95% CI | Exact McNemar p |
|---:|---:|---:|---:|---:|---:|
| 533 | 13/512 (2.54%) | 20/512 (3.91%) | +1.37 pp | [+0.20,+2.73] pp | 0.0654 |
| 917 | 11/512 (2.15%) | 18/512 (3.52%) | +1.37 pp | [+0.39,+2.54] pp | 0.0391 |
| Seed mean | 2.34% | 3.71% | +1.37 pp | [+0.39,+2.54] pp (two-way) | -- |

The corresponding view-specific paper-STP effects were:

| Seed | View 1 | View 2 | View 3 | View 4 | View 5 |
|---:|---:|---:|---:|---:|---:|
| 533 | +1.37 pp | +0.98 pp | +1.37 pp | +1.56 pp | +1.56 pp |
| 917 | -0.20 pp | -0.20 pp | +0.78 pp | +1.76 pp | +0.78 pp |

Paper STP improved teacher-forced CE by -0.00168 and -0.00574, correct-token
rate by +0.076 and +0.234 pp, and mean margin by +0.201 and +0.285. Exact
top-10 changed by -0.20 and +5.86 pp.

This is encouraging but is only a two-seed screen. Directly against released
STP, paper STP changed top-1 by -0.78 pp in seed 533 and +0.39 pp in seed 917;
the mean direct difference was -0.20 pp with two-way 95% CI
[-1.76,+1.37] pp. The frozen selector required paper STP to exceed released
STP by at least 0.5 pp in mean treatment effect with both effects nonnegative.
It therefore retained released STP. This decision does not establish that the
released equation is universally better; it only applies the stated screening
rule to this endpoint.

## Stage D: limited coefficient screen

The selected formulation/rank was released STP at rank 8. Existing lambda=0.02
checkpoints were reused; only lambda=0.005 and 0.08 were newly trained.

| Lambda | Seed 533 top-1 change | Seed 917 top-1 change | Mean change | Two-way 95% CI |
|---:|---:|---:|---:|---:|
| 0.005 | 13->23, +1.95 pp | 11->14, +0.59 pp | +1.27 pp | [-0.20,+2.93] pp |
| 0.02 | 13->24, +2.15 pp | 11->16, +0.98 pp | +1.56 pp | [+0.20,+3.12] pp |
| 0.08 | 13->17, +0.78 pp | 11->17, +1.17 pp | +0.98 pp | [-0.20,+2.15] pp |

Five-view treatment effects:

| Lambda / seed | View 1 | View 2 | View 3 | View 4 | View 5 |
|---|---:|---:|---:|---:|---:|
| 0.005 / 533 | +0.39 pp | +0.59 pp | +1.37 pp | +1.95 pp | +0.59 pp |
| 0.005 / 917 | +0.20 pp | +0.39 pp | +1.37 pp | +1.37 pp | +0.98 pp |
| 0.02 / 533 | +0.39 pp | +0.98 pp | +1.17 pp | +1.95 pp | +1.56 pp |
| 0.02 / 917 | +0.20 pp | +0.59 pp | +1.95 pp | +0.98 pp | +1.17 pp |
| 0.08 / 533 | +0.39 pp | +0.78 pp | +0.78 pp | +0.98 pp | +1.17 pp |
| 0.08 / 917 | -0.20 pp | +0.59 pp | +1.17 pp | +0.98 pp | +0.98 pp |

Lambda=0.02 has the largest mean exact top-1 effect. Neither edge improves it
by the preregistered one-percentage-point materiality threshold, so no outside
coefficient or dense sweep was run. Teacher-forced CE changes were
-0.00338/-0.00389 at 0.005, -0.00383/-0.00255 at 0.02, and
+0.00362/-0.00291 at 0.08. The CE harm in seed 533 at 0.08 is direct evidence
that stronger auxiliary pressure can begin to trade off against the native
objective. The final cross-rank check was not triggered because no materially
better coefficient was selected.

## Training and artifact accounting

All new runs completed 320 optimizer updates and wrote only the final
checkpoint. Paths below are relative to `runs/stp_matrix/a6000`; each row has
`seed_533` and `seed_917` subdirectories containing `training/result.json`, the
adapter checkpoint, full evaluation predictions, summary, and teacher-forced
rows.

| Condition path | Per-seed training wall time (s) | Per-seed peak allocated VRAM (GB) | Paired wall / sampled GPU / device peak |
|---|---:|---:|---:|
| `stage_b/native_r128` | 509.30 / 527.84 | 8.267 / 8.281 | 554.58 s / 84.45% / 17,603 MiB |
| `stage_b/released_r128_l0.02` | 685.38 / 673.06 | 8.267 / 8.278 | 704.25 s / 66.57% / 17,603 MiB |
| `stage_c/paper_r8_l0.02` | 638.54 / 644.99 | 6.612 / 6.621 | 657.96 s / 72.42% / 14,561 MiB |
| `stage_d/released_r8_l0.005` | 653.80 / 654.78 | 6.612 / 6.621 | 670.15 s / 69.11% / 14,559 MiB |
| `stage_d/released_r8_l0.08` | 643.12 / 633.35 | 6.612 / 6.621 | 657.22 s / 75.65% / 14,559 MiB |

The new rank-8 evaluations took 1,748.58--1,875.03 seconds per 512-reaction
checkpoint; rank-128 took 1,975.84--2,144.53 seconds. Rank-8 evaluation mean
GPU utilization was 85.38--93.49% and rank-128 was 90.04--95.15%. No new
evaluation rewrite was retained, so the new-evaluator speedup is 1.000x; this
is an explicit negative optimization result, not an omitted benchmark.

The execution source commit was `acba159` (the implementation sequence begins
at `52b3e80`); Python 3.10.12, PyTorch 2.3.0/CUDA 12.1, Transformers 4.45.2,
PEFT 0.13.2, one RTX A6000, and six CPUs were recorded in
`environment.json`. The exact endpoint and training-manifest hashes are
recorded there and in every result. `analysis.json` preserves all per-reaction
paired vectors and uncertainty; each `predictions.jsonl` preserves raw and
canonical ordered beam candidates for all five views.

## Mechanism diagnostics

Fixed spans were evaluated on the same 16 reactions for all six existing
rank-8 lambda=0.02 checkpoints. Released-STP training lowered fixed-span STP
loss from 1.446 to 1.327 (seed 533), 1.464 to 1.376 (917), and 1.428 to 1.345
(1301). It therefore learned its stated geometric objective in all seeds.
After weighting by lambda=0.02, mean STP/native gradient-norm ratios at the
trained STP checkpoints were 4.04%, 2.86%, and 2.54%. Mean gradient cosines
were +0.094, +0.175, and -0.010. The two generation-improving seeds have mildly
positive held-out interaction while the harmful seed is neutral, but three
points cannot establish this as a causal explanation.

Across all conditions, first native/treatment top-1 beam divergence is early:
the median is token 1 at rank 8 and token 4 at rank 128. This explains why
small teacher-forced changes can be amplified into different sequences, but
does not prove which early decision causes final exact correctness. The
teacher-forced/generation disconnect remains concrete: Stage A seed 1301 and
Stage B seed 533 improve CE yet reduce exact generated top-1.

## Final interpretation

**Verdict: INCONCLUSIVE for a robust released-STP improvement in ChemFM
forward-reaction generation.** On the full three-seed 512-reaction Stage A
evidence, the mean effect is +0.65 pp, but the two-way interval
[-1.17,+2.54] pp crosses zero and one seed is harmed. The experiment therefore
does not establish a seed-robust benefit. It also does not support the capacity
escape hatch: rank 128 reduces the mean treatment effect and remains
seed-inconsistent.

The result is not a global failure of STP. In the locked two-seed screening
subset, released STP at rank 8/lambda=0.02 improves both seeds, and literal
paper STP independently improves both seeds by +1.37 pp. Those results remain
exploratory because they reuse only two favorable members of the existing
three-seed set for model selection, and direct paper-versus-released
uncertainty includes zero. The coefficient screen identifies no better setting
than 0.02 and supplies no justification for further tuning.

What is established is narrower and useful: released STP learns the intended
geometry; nominally similar paper/released losses generate substantially
different gradient vectors and effective pressure; favorable CE changes do
not guarantee exact generation; and more LoRA capacity does not materially
increase the STP treatment effect under this protocol.

The single most justified continuation is a locked confirmatory run of
rank-8 released STP at lambda=0.02 on additional independent training seeds,
with the same 512 panel and no further rank/formulation/coefficient selection.
That directly resolves the remaining seed uncertainty without turning this
screen into an adaptive search.
