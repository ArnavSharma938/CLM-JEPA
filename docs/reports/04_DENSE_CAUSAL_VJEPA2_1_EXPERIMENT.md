# Dense causal V-JEPA 2.1-style experiment

## Scope

The implementation translates V-JEPA 2.1 dense masked/context prediction and
deep self-supervision to the causal ChemFM product-token field. The evaluation
records causal token states, global source-product measurements, NTP, and beam
generation.

This experiment is a mechanism change, not a gradient filter. Historical
endpoint MSE/SIGReg, PCSF, projection heads, LLM-JEPA predictor tokens, PCGrad,
CAGrad, and auxiliary-similarity gating are absent from the new objective.

## Repository chronology and model context

This report first appeared on 2026-08-25 in commit `8e059c2`, one repository
day after the final endpoint generation-pathway audit and two days after the
gradient-interaction implementation record. It remains separate because it is
not another endpoint intervention: it adds a dense predictor, four-depth
supervision, suffix/context masks, and an EMA ChemFM target. It uses the same
ChemFM-1B base and seed-533 USPTO-MIT pilot manifest for comparison, but its
training state and objective family are distinct from all parts of the endpoint
intervention report.

## Primary-source lock

The implementation was frozen only after reading the complete 37-page
[V-JEPA 2.1 paper, arXiv:2603.14482v3](https://arxiv.org/abs/2603.14482) and
the official `facebookresearch/vjepa2` implementation at commit
[`204698b45b3712590f06245fbfba32d3be539812`](https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812/app/vjepa_2_1).
The executable references were `train.py`, `models/predictor.py`,
`models/vision_transformer.py`, `models/utils/masks_dist.py`, and the released
ViT-G/16 configuration.

The load-bearing reference facts are:

- dense masked and visible/context targets use elementwise L1;
- visible targets receive `1/sqrt(d_min)` proximity weights;
- the released context coefficient is 0.5 and is progressively enabled;
- four approximately equally spaced encoder levels are learned-normalized,
  concatenated, and fused through `4D -> D -> 384`;
- the predictor is a 24-block, width-384, 12-head noncausal transformer with
  separate masked and context output projections;
- the target applies an additional parameter-free LayerNorm per depth slice;
- all four equal-width target slices receive equal weight;
- the released target EMA coefficient is fixed at 0.99925;
- the released mask mixture has eight local 0.15-scale masks and two global
  0.70-scale masks.

The official code averages the proximity-weighted context error over visible
tokens/elements. It does not divide by the sum of the weights. The maintained
implementation follows that executable behavior. The exact translation table
and deviations are in `docs/VJEPA2_1_CHEMFM_MAPPING.md`.

## Causal ChemFM formulation

For each teacher-forced reaction, the target `<prostart>` marker remains
visible and one product-suffix boundary is sampled. Ten resumable mask
identities preserve the official 8:2 local/global mixture: eight use
`ceil(0.15 n)` future tokens and two use `ceil(0.70 n)`. The predictor receives
the source and true product prefix. Every future student feature is replaced by
the chosen train-only zero-initialized mask latent.

The online ChemFM forward is the ordinary complete teacher-forced NTP forward.
Only states before the sampled boundary enter JEPA as visible content. Because
ChemFM attention is causal, these prefix states are exactly identical to a
physically truncated `C_t` forward; the real-Llama test verifies bit-exact
equality after future tokens are changed. Reusing this forward avoids a third
ChemFM call without changing the objective.

The EMA target performs a second causal forward over the complete true
reaction and returns detached targets at depths 6, 11, 17, and 22. Frozen
ChemFM-1B parameters are shared; only EMA copies of trainable encoder state and
the four target norms are stored. The target is never in the optimizer and is
updated after the student optimizer step.

At each depth, the predictor emits a masked and a context slice. Masked L1 is
mean-reduced over true future positions, latent elements, and then levels.
Context L1 is multiplied by the continuous absolute-coordinate weight
`1/sqrt(boundary-position)` before the same reduction. The context coefficient
is 0, linearly ramps, and then stays at 0.5. The paper's 15k--30k interval is
mapped to 1/9--2/9 of the planned ChemFM update budget; using 15k literally
would disable the defining context mechanism throughout this 320-step pilot.

The full objective is ordinary label-shifted ChemFM target CE plus coefficient
one times dense JEPA. The tokenizer, LM-head width, NTP labels, decoder, beam
search, and generation-time model are unchanged.

## Maintained implementation and state

- `src/vjepa2_1.py`: suffix sampler, continuous weights, layer capture,
  hierarchical norms/fusion, rotary dense predictor, EMA functional target,
  loss, schedule, component VJPs, and checkpoint state.
- `src/train.py`: `clm_jepa_vjepa2_1` condition, shared optimizer, post-step
  EMA, logging, validation, and exact resume.
- `scripts/audit_vjepa2_1_feasibility.py`: frozen local target-token CE/CKA and
  representation displacement against validated native/direct endpoints.
- `scripts/run_vjepa2_1_a6000.sh`: sequential exact-environment setup,
  hash-verified model acquisition, super-mini, and pilot.
- `scripts/run_vjepa2_1_evaluation_a6000.sh`: process-triggered, sequential
  global and local frozen evaluation.

Each checkpoint contains the student PEFT adapter, predictor, online/target
level norms, EMA encoder tensors, EMA update count, optimizer/scheduler, RNG
states, progressive step budget, and exact suffix-sampler call index.

## Verification

The final local suite passed 102 tests with one intentional skip. The same
environment-pinned A6000 suite passed 101 tests with one skip before the BF16
integration correction; the focused corrected A6000 suite then passed 19/19.
Coverage includes causal non-leakage in a real HF Llama, product alignment,
mask scales, continuous source-to-prefix distance weights, four-depth block
semantics, predictor invariance to replaced future features, EMA-only Polyak
updates, stop-gradient targets, exact JEPA-disabled NTP parity, real PEFT
functional EMA execution, BF16 module compatibility, component-VJP integrity,
and predictor/EMA/sampler/schedule resume.

The model snapshot was downloaded at revision
`f99dc2e89726539bb9cf31b2e2b4360650bac6a8` and verified as 3,881,171,344
bytes with SHA-256
`24686705d779db6876acc09c81d64d432262ef8b5dbfccc385212587079ce419`.

## A6000 super-mini feasibility

One RTX A6000 48 GB / six-vCPU / 200-GB prototyping instance used pinned
PyTorch 2.3.0+cu121. Sixteen training reactions, eight validation reactions,
batch 2, and eight optimizer updates completed. Training took 27.56 seconds;
the complete process including model load, CE, and beam-10 validation took
111.3 seconds.

| Diagnostic | First | Final |
|---|---:|---:|
| NTP loss (training batch) | 1.5712 | 1.5701 |
| Masked dense L1 | 0.7813 | 0.7461 |
| Context dense L1 | 0.1167 | 0.1250 |
| Context coefficient | 0.0 | 0.5 |
| Student scale, depths 6/11/17/22 | .9962/.9971/1.0000/1.0000 | .9989/.9997/.9999/1.0000 |
| EMA scale, depths 6/11/17/22 | .9964/.9981/1.0000/1.0000 | .9990/.9997/.9999/1.0000 |

At the first fully active context step, exact gradient contributions to shared
ChemFM trainables were finite:

| Depth | Masked norm | Active context norm | Predictor masked/context norm |
|---:|---:|---:|---:|
| 6 | 21.253 | 1.808 | .2451/.0092 |
| 11 | 7.086 | 1.192 | .2453/.0095 |
| 17 | 9.819 | 3.665 | .2125/.0083 |
| 22 | 7.215 | 3.033 | .2192/.0080 |

The first aggregate ChemFM gradient included the saved-embedding first-step
transient also measured in the prior native/direct runs. It was
clipped by the unchanged canonical trainer; dense component VJPs themselves
were finite. Validation CE was 0.93766. Exact top-1 was 0/8 and is not treated
as an efficacy estimate.

## Controlled pilot and frozen evaluation

The controlled seed-533 pilot used the same 1,280 training reactions, frozen
256-reaction validation manifest, physical/accumulation/logical batch
`4/4/16`, four epochs, and 320 optimizer updates as the prior controlled
conditions. Dense JEPA was active on every logical update at coefficient one.
The context coefficient ramped from zero over steps 36--71 and remained 0.5.

Training took 2,315.43 seconds (38.59 minutes). The full process, including
model loading and one-view beam-10 generation over 256 reactions, took
5,155.74 seconds (85.93 minutes). Peak allocated CUDA memory was 7.77 GB and
effective throughput was 242.8 tokens/s. The ordinary training-batch NTP loss
fell from 1.5598 to 0.1340. Masked L1 was 0.7773 initially, 0.5415 at the first
fully active context step, and 0.6348 on the final sampled batch; context L1
was 0.1333, 0.0995, and 0.1016 respectively. The nonmonotonic masked values
are expected from the deliberately mixed short/long horizons and are not a
same-example learning curve.

At step 72, the exact per-component gradients into trainable ChemFM state were
small relative to the ordinary combined gradient norm of 14.89:

| Depth | Masked ChemFM norm | Context ChemFM norm | Masked predictor norm | Context predictor norm |
|---:|---:|---:|---:|---:|
| 6 | .00911 | .00196 | .06231 | .00459 |
| 11 | .01533 | .00169 | .07257 | .00509 |
| 17 | .02701 | .00227 | .08287 | .00547 |
| 22 | .02691 | .00324 | .08475 | .00589 |

At step 72, the predictor/ChemFM norm ratios for masked components at depths
6/11/17/22 were `6.84x`, `4.73x`, `3.07x`, and `3.15x`. Context-component
ratios were `2.34x`, `3.01x`, `2.41x`, and `1.82x`.

### One-view autoregressive behavior

All rows use the identical ordered 256-reaction panel and beam-10 evaluation.
Native and direct MSE+SIGReg are the validated epoch-4 references recorded in
Part III of [the trained-intervention report](03_ENDPOINT_INTERVENTION_EXPERIMENTS.md); dense
V-JEPA is the new epoch-4 checkpoint.

| Condition | Top-1 | Top-3 | Top-5 | Top-10 | Valid candidates |
|---|---:|---:|---:|---:|---:|
| Native | 6/256 (2.34%) | 26/256 (10.16%) | 40/256 (15.63%) | 52/256 (20.31%) | 78.20% |
| Direct MSE+SIGReg | 6/256 (2.34%) | 24/256 (9.38%) | 34/256 (13.28%) | 49/256 (19.14%) | 86.60% |
| Dense causal V-JEPA 2.1 | 6/256 (2.34%) | 21/256 (8.20%) | 29/256 (11.33%) | 39/256 (15.23%) | 83.20% |

Dense supervision had the same exact top-1 count and lower top-3/5/10 counts
than both references. This is a descriptive 256-reaction pilot. The
dense run's aggregate token-weighted validation loss was 0.239423, but that
reduction convention is not identical to the historical per-reaction CE
reported for native/direct. The matched frozen token analysis below supplies
the fair CE comparison.

### Exact target-token states used by the LM head

The frozen local panel contains 2,748 aligned product targets from the first 64
reactions. It compares `h[:, :-1]` exactly where `labels[:, 1:]` is a product
token, rather than an isolated JEPA endpoint.

| Condition | Target-token CE | Teacher-forced top-1 |
|---|---:|---:|
| Native | .243832 | 92.394% |
| Direct MSE+SIGReg | .253330 (+.009498) | 92.431% |
| Dense causal V-JEPA 2.1 | .253547 (+.009714) | 92.249% |

Dense target-token CE was 3.98% above native and `0.000217` above the direct
endpoint condition. The dense tokenizer retains ChemFM's native 392
entries; the legacy endpoint checkpoints contain ten unused historical
predictor tokens (402 entries). Shared chemical token IDs are identical. Since
the ten extra logits appear only in the legacy endpoints, this tokenizer-size
difference does not increase the dense condition's vocabulary denominator.

The causal-token comparisons were:

| Depth | Direct CKA / relative RMS displacement | Dense CKA / relative RMS displacement |
|---:|---:|---:|
| 6 | .9814 / .2481 | .9877 / .1947 |
| 11 | .9817 / .2690 | .9891 / .2097 |
| 17 | .9762 / .3028 | .9854 / .2351 |
| 22 | .9897 / .2268 | .9940 / .1444 |
| Final ChemFM norm | .9866 / .1880 | .9936 / .1233 |

Dense CKA was higher and relative RMS displacement lower than direct at every
reported depth. Dense target-token CE was `.253547`, teacher-forced top-1 was
`92.249%`, and beam top-10 was `39/256`.

### Global source-product structure

The standard k=0 frozen evaluator used all 256 reactions and pinned the
validated native/direct epoch-4 checkpoints explicitly.

| Condition | Source/target variance | Source/target effective rank | Pair-center spread | Raw retrieval | Residual retrieval | Ridge explained variance |
|---|---:|---:|---:|---:|---:|---:|
| Native | .001432/.002317 | 41.02/22.70 | .03447 | 45.31% | 78.52% | .1366 |
| Direct MSE+SIGReg | .001293/.001282 | 38.34/34.01 | .03125 | 83.98% | 93.75% | .3162 |
| Dense causal V-JEPA 2.1 | .001255/.001988 | 40.68/27.64 | .03182 | 33.59% | 77.34% | .0819 |

Dense raw retrieval, residual retrieval, and ridge explained variance were
`33.59%`, `77.34%`, and `.0819`; native values were `45.31%`, `78.52%`, and
`.1366`; direct values were `83.98%`, `93.75%`, and `.3162`.

## Artifact preservation

The local machine retains the complete machine-readable pilot, super-mini,
global representation, and local target-token JSONs under
`runs/vjepa2_1/a6000/`. The selected epoch-4 checkpoint is also local and is
512,537,041 bytes across seven files. Every file SHA-256 was checked against
the remote copy before the A6000 instance was deleted. In particular,
`training_state.pt` is
`1cf356e73e61ae8efd2b7d30449ce541741d759505df2429c21cae1a9d1b03e4`
and the adapter is
`d594b64b71f2e6adae9a7d0859f783de13e7f76ed775720cc7cd97e16cfa9b74`.

## Recorded comparisons

- Causal non-leakage, target alignment, four-depth extraction, EMA-only update,
  detached targets, disabled-JEPA parity, and checkpoint/resume tests passed.
- The eight-step super-mini completed with finite per-depth component gradients.
- The four-epoch pilot completed 320 updates. Target-token CE was `.253547`,
  versus `.243832` native and `.253330` direct endpoint MSE+SIGReg.
- Exact top-1 was `6/256` for all three conditions; dense top-3/5/10 counts were
  `21/29/39`, versus native `26/40/52` and direct `24/34/49`.
- Dense causal-token CKA was higher than direct at every reported depth. Dense
  global raw retrieval was `33.59%`, versus native `45.31%` and direct
  `83.98%`.
- At step 72, predictor/ChemFM component-gradient norm ratios ranged from
  `3.07x` to `6.84x` for masked loss and `1.82x` to `3.01x` for context loss.
