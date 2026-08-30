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
