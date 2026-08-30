# Official Semantic Tube Prediction on ChemFM

## Preregistered protocol

This section was fixed on 2026-08-30 after the complete paper and upstream-code
audit, but before any trained STP ChemFM endpoint was generated.

### Authoritative method and paper/code discrepancy

The paper is arXiv:2602.22617v1, *Semantic Tube Prediction: Beating LLM Data
Efficiency with JEPA*. The executable reference is
`https://github.com/galilai-group/llm-jepa` at commit
`ea0017c654ad917066ff32afc88276bea8ca5f7e`. The downloaded PDF SHA-256 is
`bed646a5d7ab80c391a83d75535215bf85f9396e35506a24425bc3f126e773bc`;
the arXiv source-tar SHA-256 is
`1f1218e0cee4afee1b40d257d03fb563917d9a1feaca538326687fc89ede6216`.

The prose/equation and executable are not identical. The paper states that
three random indices `s < r < t` are used for
`1 - cos(h_t - h_r, h_r - h_s)`. The released experimental command is
`--linear=random_span`. Its implementation instead:

1. concatenates the query and answer *content* regions while skipping their
   intervening serialization tokens;
2. samples one non-full patch `[a,b)` by drawing `a` uniformly, drawing `b`
   uniformly conditional on `a`, and rejecting only the full content span;
3. constructs a transition for the patch and the sum of the transitions before
   and after the patch; and
4. minimizes `1 - cosine(patch, before + after)`.

That released behavior has been unchanged since the first STP source commit
`313a7a7`. No issue, tag, branch, or later source revision supplies a direct
three-index implementation for the published runs. In accordance with the
request to port working official code, the executable random-span objective is
the intervention tested here. The distinction is covered by an executable
parity test and will remain explicit in the interpretation.

### ChemFM-compatible port

The only architectural adaptation is boundary discovery. Official query
content maps to reactant-SMILES tokens and official answer content maps to
product-SMILES tokens in the unchanged ChemFM serialization
`<rstart>reactants<eos><prostart>product<eos>`. The four framing tokens are
excluded from content spans, just as upstream chat framing is excluded. The
ordinary ChemFM product-token NTP labels are unchanged.

Everything else follows the released base configuration:

- one ordinary causal forward pass with hidden states returned;
- final-layer per-token hidden states;
- one released-default random span per reaction on every microbatch;
- FP32 transition buffers and PyTorch `F.cosine_similarity` defaults;
- symmetric gradients through both compared transitions;
- no predictor, projection head, masking, EMA, stop-gradient, SIGReg, extra
  view, endpoint JEPA, or auxiliary dropout;
- total loss `L_NTP + 0.02 L_STP`, where `0.02` is the official released
  Llama-3.2-1B/SYNTH coefficient.

A dedicated device generator is seeded from the fine-tuning seed exactly as in
the upstream rank-zero experiment. Its state is checkpointed. It does not
advance the global model/dropout RNG stream.

### Matched ChemFM experiment

The fixed comparison is native ChemFM NTP versus NTP plus official executable
STP. The established native trajectories from the completed pair-residual
study are reused rather than recomputed, subject to exact configuration,
manifest, model, starting-trainable-state, and checkpoint validation. They are
valid controls because their native path is unchanged by this extension.

| Setting | Locked value |
|---|---|
| Dataset/task | USPTO-MIT forward reaction prediction |
| Model | pinned ChemFM-1B with maintained rank-8 LoRA |
| Train manifest | frozen 1,280-row pilot |
| Seeds | `533, 917, 1301` |
| Epochs / updates | 4 / 320; fixed epoch-4 endpoint |
| Physical / accumulation / effective batch | 4 / 4 / 16 |
| Optimizer | fused AdamW, LR `1e-4`, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `0.01` |
| Scheduler | cosine, 5% warmup, minimum LR `1e-5` |
| Precision / attention | BF16 / SDPA, no gradient checkpointing |
| Hardware | one authorized Thunder RTX A6000, 6 vCPU, 100 GB, no template |

The initialized trainable SHA-256 must match native and STP within every seed.
All other training inputs, ordering, serialization, stochastic model stream,
optimizer budget, and evaluation are fixed. No lambda tuning or STP ablation is
permitted after inspecting outcomes.

### Endpoint, mechanism checks, and verdict

The primary endpoint is the frozen 256-reaction manifest
`data/clm_jepa_uspto_mit_official_endpoint/prespecified_stage1_256.jsonl`, using
all five official R-SMILES views, beam width 10, ten candidates per view,
official canonicalization, and reciprocal-rank aggregation. Report aggregate
and per-view generated exact top-1 for every seed.

Mechanism diagnostics are limited to the STP loss trajectory, sampled span
fractions, and fixed-panel teacher-forced target-token CE, correct-token margin,
and correct-token rate. These explain activity and token-level consequences
without replacing generation as the endpoint.

Use reaction-paired bootstrap intervals and exact McNemar comparisons within
seed. Across seeds report each paired effect, mean, sample SD, a seed-level
t interval, and crossed seed-by-reaction bootstrap uncertainty. With only three
seeds, seed-level uncertainty is emphasized and pooled seed-reaction
significance is not claimed.

- **PASS:** all three aggregate exact top-1 effects are positive, their mean is
  positive, and the crossed paired 95% interval excludes zero.
- **FAIL:** the mean effect is nonpositive and no more than one seed improves.
- **INCONCLUSIVE:** every other outcome.

The verdict tests this pinned released STP objective, coefficient, ChemFM-LoRA
trajectory, and budget. A negative result does not falsify other coefficients,
the paper's non-executable three-index equation, or the authors' reported
natural-language datasets.

## Results

### Execution and provenance

The preregistered implementation was committed as
`aaf6894a29c60a409c53b3d5390f66307757a9fe` before a full STP endpoint existed.
It passed `94` tests with `1` skip locally and on Thunder, including exact
released-executable parity for the sampled spans, loss, hidden-state
transitions, and every model-parameter gradient. The executable upstream
reference remained `galilai-group/llm-jepa` commit
`ea0017c654ad917066ff32afc88276bea8ca5f7e`.

Execution used one RTX A6000, 6 vCPUs, and 100 GB storage with PyTorch
`2.3.0+cu121`, Transformers `4.45.2`, and PEFT `0.13.2`. The ChemFM model was
revision `f99dc2e89726539bb9cf31b2e2b43606650bac6a8`, model SHA-256
`24686705d779db6876acc09c81d64d432262ef8b5dbfccc3852112587079ce419`.
The canonical deployment archive SHA-256 was
`b14a3cf53bde2b9d4b1bacf7f8748628d01483952e75a7dceff5bebe953fa8cb`.
The frozen endpoint manifest matched the native evaluation byte-for-byte at
SHA-256 `5b87bce1e75ed1ebf1a2a9091e0367aedaa8600a5621bc960352cf45b18e1865`.

Every STP trajectory completed 320 optimizer updates and selected epoch 4.
Within each seed, the initialized trainable SHA-256 exactly matched its native
control. Three-way concurrent training finished in about 26 minutes; individual
saved wall times were `1523.7`, `1510.8`, and `1509.0` seconds. Each
four-worker five-view generation endpoint took `955.5`, `974.7`, and `958.4`
seconds including model load. The complete production driver ran for about 77
minutes. After validation, only disposable optimizer-resume states and the
preflight adapter were removed. The three final adapters and all scientific
outputs remain under `runs/stp/a6000/`. The retrieved compact archive SHA-256
was `e0a7ca0fe3ebbdebf9517343b6243b710b3e8bd8b7baa1db0d86b12edb3b9a65`.
The Thunder instance was deleted after local hash and analysis reproduction.

### Primary generated exact top-1

Each cell is `native -> STP (STP - native)` in percentage points. Every value
uses exactly 256 reactions.

| Seed | Five-view aggregate | View 1 | View 2 | View 3 | View 4 | View 5 |
|---:|---:|---:|---:|---:|---:|---:|
| 533 | 2.73 -> 5.08 (+2.34) | 2.34 -> 3.52 (+1.17) | 2.73 -> 4.30 (+1.56) | 1.95 -> 2.73 (+0.78) | 1.56 -> 3.91 (+2.34) | 3.12 -> 4.30 (+1.17) |
| 917 | 2.34 -> 3.12 (+0.78) | 4.30 -> 3.91 (-0.39) | 4.30 -> 4.69 (+0.39) | 1.95 -> 3.91 (+1.95) | 1.17 -> 2.73 (+1.56) | 1.95 -> 3.12 (+1.17) |
| 1301 | 5.08 -> 3.52 (-1.56) | 3.52 -> 3.91 (+0.39) | 3.12 -> 3.12 (0.00) | 3.12 -> 1.56 (-1.56) | 2.73 -> 1.95 (-0.78) | 3.52 -> 4.69 (+1.17) |
| Seed mean | 3.39 -> 3.91 (+0.52) | 3.39 -> 3.78 (+0.39) | 3.39 -> 4.04 (+0.65) | 2.34 -> 2.73 (+0.39) | 1.82 -> 2.86 (+1.04) | 2.86 -> 4.04 (+1.17) |

Native produced `26/768` exact aggregate top-1 predictions and STP produced
`30/768`. Aggregate seed effects were `+2.34`, `+0.78`, and `-1.56` pp. Their
mean was `+0.52` pp, sample SD `1.97` pp, seed-level t 95% interval
`[-4.36,+5.40]` pp, and crossed seed-by-reaction bootstrap 95% interval
`[-1.95,+3.12]` pp.

Within-seed paired results were:

| Seed | STP - native | Paired bootstrap 95% CI | Exact McNemar p |
|---:|---:|---:|---:|
| 533 | +2.34 pp | [0.00,+4.69] pp | 0.1094 |
| 917 | +0.78 pp | [-1.17,+3.12] pp | 0.7266 |
| 1301 | -1.56 pp | [-3.52,0.00] pp | 0.2188 |

The five view-level mean effects were `+0.39`, `+0.65`, `+0.39`, `+1.04`,
and `+1.17` pp. View 5 was `+1.17` pp in all three seeds, but its crossed
seed-by-reaction interval was `[-0.26,+2.86]` pp. The crossed intervals for
all five views included zero; consequently, no view-level significance or
general five-view survival claim is made.

### Mechanism diagnostics

The auxiliary objective was active and learned along every trajectory. Mean
STP loss changed from epoch 1 to epoch 4 as follows:

| Seed | Epoch-1 STP loss | Epoch-4 STP loss | Change | Epoch-4 sampled span fraction |
|---:|---:|---:|---:|---:|
| 533 | 1.4131 | 1.2735 | -0.1396 | 0.2461 |
| 917 | 1.4578 | 1.3075 | -0.1502 | 0.2524 |
| 1301 | 1.4148 | 1.2570 | -0.1578 | 0.2650 |

Teacher-forced native token decisions improved more consistently than
generation. Token-weighted CE and correct-token rate were:

| Seed | Native CE -> STP CE | CE change | Native correct rate -> STP correct rate | Margin change |
|---:|---:|---:|---:|---:|
| 533 | 0.22948 -> 0.22647 | -0.00301 | 93.071% -> 93.201% | +0.2307 |
| 917 | 0.23631 -> 0.23337 | -0.00294 | 92.923% -> 92.986% | +0.3414 |
| 1301 | 0.23349 -> 0.22908 | -0.00440 | 93.032% -> 93.073% | -0.0137 |

On reaction-paired five-view averages, CE improved in all three seeds with mean
change `-0.00420`, seed-level t 95% interval `[-0.00570,-0.00269]`, and
crossed bootstrap interval `[-0.00630,-0.00205]`. Correct-token rate also
improved in all three seeds, but its crossed interval included zero. Margins
improved in two seeds and were essentially flat/slightly negative in seed
1301.

The key disconnect is seed 1301: it had the largest token-weighted CE
improvement and a higher correct-token rate, yet aggregate generated exact
top-1 fell by `1.56` pp. Thus better teacher-forced native decisions did not
reliably integrate into better sequence-level generation.

## Verdict and scope

**INCONCLUSIVE.** The primary mean favored STP, but only two of three seeds
improved and both across-seed uncertainty intervals included zero. This fails
the preregistered PASS rule and does not establish that persistent official
executable STP improves ChemFM forward-reaction generation. It also does not
meet the preregistered FAIL rule because the mean was positive and two seeds
improved.

The result directly rules out a deterministic or seed-uniform improvement at
this pinned coefficient, data budget, ChemFM-LoRA setup, and optimizer budget:
one fully matched seed worsened despite better CE. It does not falsify a small
positive expected effect, because the three-seed interval is wide; nor does it
test the paper's non-executable three-index equation, other coefficients, or
the authors' natural-language results. No post-outcome tuning or alternate STP
variant was run.

The consolidated analysis is `runs/stp/a6000/analysis.json`. Per-seed raw
predictions, teacher-forced rows, paired summaries, training curves, and final
adapters are in `runs/stp/a6000/results/`; hardware and software provenance is
in `runs/stp/a6000/environment.json`.
