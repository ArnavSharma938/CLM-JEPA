# Projection-space MSE+SIGReg experiment

## Measured summary

The projected space had source/target variance `0.243582 / 0.263016`, effective
rank `3.22 / 3.29`, and pair retrieval `76.56%`. Raw LM space had variance
`0.004884 / 0.012639`, effective rank `12.58 / 5.22`, and pair retrieval
`85.55%`. On the matched
256-reaction teacher-forced panel, projected cLM-JEPA CE was `0.256497`, worse
than direct MSE+SIGReg (`0.248779`) and native (`0.240683`). On the first 512
reactions of the frozen official five-view manifest, exact top-1 was 4/512,
versus 15/512 for direct MSE+SIGReg and 18/512 for native.

## Primary-source basis

The architecture was fixed before training from the requested primary sources.

- [LeJEPA](https://arxiv.org/pdf/2511.08544) reports its best tested projector
  dimension at 64 in Table 1d and generally favors a three-layer projector in
  its depth ablation.
- The pinned official LeJEPA
  [`MINIMAL.md`](https://raw.githubusercontent.com/galilai-group/lejepa/c293d291ca87cd4fddee9d3fffe4e914c7272052/MINIMAL.md)
  separates backbone `emb` from shared-MLP `proj`, applies invariance and
  SIGReg to `proj`, and evaluates `emb`. Its MLP has two wide hidden layers with
  BatchNorm.
- [SimCLR Section 4.2](https://arxiv.org/pdf/2002.05709) reports its nonlinear,
  linear, and no-projection comparisons and evaluates the representation before
  the head for downstream use.
- The official SimCLR
  [`model.py`](https://github.com/google-research/simclr/blob/master/model.py)
  and
  [`model_util.py`](https://github.com/google-research/simclr/blob/master/model_util.py)
  pass projected hidden states to the SSL loss, use hidden BN/nonlinear layers,
  and omit a final ReLU.

The frozen shared head was therefore
`2048 -> 2048 -> 2048 -> 64`: hidden Linear-BatchNorm-ReLU blocks and a final
Linear layer, with no final nonlinearity and no L2 normalization. It has
8,532,032 trainable parameters.

## Objective and implementation

For the unchanged k=0 source and target EOS states `h_s,h_t`, training used one
shared projector `P` and only the following auxiliary:

`z_s=P(h_s), z_t=P(h_t)`

`L_aux = MSE(z_s,z_t) + [4 * 0.01 / 0.99] SIGReg({z_s,z_t})`

`L_total = L_NTP + active * 2 * L_aux`

`active` retained the established 50% logical-batch cadence. The physical batch
was 4, accumulation/logical JEPA batch was 16, and the projector received the
concatenated 32 source-plus-target rows in one call. Both BatchNorm layers thus
normalized the logical JEPA batch rather than individual microbatches.

The active update uses a no-grad endpoint pass over the four physical chunks,
one exact logical-batch projector/SIGReg backward pass, and RNG-replayed ChemFM
passes that inject the endpoint VJP alongside unchanged NTP. Gradients flow
`z -> P -> h -> ChemFM`; no endpoint is detached. A compact native-logit path
was enabled after executable state/loss/gradient parity tests to avoid
projecting unused token logits during active replay.

There was no raw-endpoint MSE or SIGReg in this condition. The projection
condition was subsequently removed from the maintained trainer. Historical
scripts, checkpoints, and evidence remain under `scripts/` and `runs/`.

## Frozen training protocol and execution

The run matched the prior direct MSE+SIGReg condition: ChemFM-1B, USPTO-MIT
forward prediction, seed 533, the identical 1,280-row training manifest, LoRA,
LR `1e-4`, four epochs, 80 optimizer updates per epoch, BF16, SDPA, fused AdamW,
no gradient checkpointing, physical batch 4, accumulation 4, k=0, outer
coefficient 2, SIGReg tradeoff `0.01`, and the same auxiliary activity RNG.
Manifest hashes matched the control exactly:

- train: `b5900bc7e4f1a858ecf3fdf3732da63e08fc0f955f1cb9ccf90534e2273c8dba`
- validation: `41db8068aa3c4b6faf7149ce8ee5645e5ce3794f64d52c0e56552456721e5013`

The A6000 preflight exercised real optimizer state and exact logical BatchNorm.
The full run completed 320 updates, 172 active logical updates, in 32.33 minutes
with 14.74 GiB peak tensor allocation.

| Epoch | Mean NTP | Projected MSE | Projected SIGReg | Active updates |
|---:|---:|---:|---:|---:|
| 1 | 1.302790 | 0.151014 | 5.078280 | 47/80 |
| 2 | 0.422565 | 0.091792 | 4.835297 | 43/80 |
| 3 | 0.307564 | 0.082210 | 3.753002 | 42/80 |
| 4 | 0.253846 | 0.085376 | 3.128446 | 40/80 |

NTP and both projected losses remained finite. The fixed two-row selector CE at
epoch 4 was `0.172400`; the larger matched CE panel below is the meaningful
autoregressive diagnostic.

## Two-space geometry

The epoch-4 diagnostic used 256 unique validation identities. Projector
BatchNorm used saved running statistics at evaluation. SIGReg's raw statistic
scales with evaluation sample count, so the 256-row value is reported as a
geometry diagnostic and is not compared numerically with the logical-N=16
training curve.

| Metric | Raw LM space h | Projected JEPA space z |
|---|---:|---:|
| Dimensions | 2048 | 64 |
| Source / target variance | `0.004884 / 0.012639` | `0.243582 / 0.263016` |
| Pair-center spread | `0.080046` | `0.491247` |
| Source / target effective rank | `12.58 / 5.22` | `3.22 / 3.29` |
| Source / target mean-direction energy | `0.9391 / 0.8389` | `0.00147 / 0.00392` |
| Correct-minus-random cosine margin | `0.04971` | `0.81588` |
| Pair retrieval top-1 / MRR | `85.55% / 90.53%` | `76.56% / 86.52%` |
| MSE alignment | n/a | `0.048751` |
| SIGReg statistic, N=256 | n/a | `49.7792` |

Across the four logical audit batches, projected source effective rank fell from
`9.59` at epoch 1 to `3.23` at epoch 2 and `2.82` at epoch 4 while variance
rose from `0.0178` to `0.3096`.

The matched epoch-4 native source/target variances were `0.001431/0.002320`, effective ranks
`41.00/22.61`, and mean-direction energies `0.9818/0.9695`. Direct MSE+SIGReg
was `0.001293/0.001281`, ranks `38.36/34.06`, and energies `0.9709/0.9698`.
The projected condition's raw values were `0.004884/0.012639`, ranks
`12.58/5.22`, energies `0.9391/0.8389`, and retrieval `85.55%`; direct
MSE+SIGReg retrieval was `85.9%`.

## Held-out NTP gradient alignment

Four disjoint logical batches were audited at epochs 1, 2, and 4. Projected
MSE, SIGReg, and their full applied combination were mapped through the
projector into the 308 LoRA gradient tensors and compared with held-out NTP.

| Epoch | Held-out NTP | z MSE / SIGReg | MSE cosine / norm ratio | SIGReg cosine / norm ratio | Full cosine / norm ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.48309 | `0.10642 / 5.21062` | `-0.0057 / 2.18x` | `+0.0087 / 0.55x` | `-0.0025 / 2.02x` |
| 2 | 0.34753 | `0.08278 / 3.76448` | `+0.0447 / 2.89x` | `-0.0385 / 1.14x` | `+0.0330 / 2.88x` |
| 4 | 0.28381 | `0.06437 / 2.78183` | `-0.0243 / 2.45x` | `-0.0079 / 0.81x` | `-0.0300 / 2.36x` |

At epoch 4, the full projected auxiliary/NTP cosine was `-0.0300` and its norm
ratio was `2.36x`. The training implementation propagated its VJP through `h`
into ChemFM.

## NTP and generation

On the same frozen 256-reaction decoder-coupling panel used by prior reports:

| Condition | Aggregate target-token CE | Relative to native |
|---|---:|---:|
| Native | 0.240683 | reference |
| Direct MSE+SIGReg | 0.248779 | +3.36% |
| **Projected MSE+SIGReg** | **0.256497** | **+6.57%** |

Projected CE was 3.10% above direct MSE+SIGReg.

Generation used exact ChemFM five-view, beam-10 scoring. The projected model's
longer decodes made all 1,280 reactions incompatible with the authorized cloud
budget, so evaluation was fixed during execution—before inspecting outcomes—to
the first 512 reactions of the already frozen sequential manifest. The same
prefix was extracted from existing native and direct prediction artifacts. It
is a matched, budget-bounded descriptive panel, not the original 1,280-reaction
confirmatory endpoint.

| Endpoint on matched 512 | Native | Direct MSE+SIGReg | Projected MSE+SIGReg |
|---|---:|---:|---:|
| Exact top-1 | 18 (3.52%) | 15 (2.93%) | **4 (0.78%)** |
| Exact top-3 | 88 (17.19%) | 83 (16.21%) | **33 (6.45%)** |
| Exact top-5 | 130 (25.39%) | 119 (23.24%) | **70 (13.67%)** |
| Exact top-10 | 187 (36.52%) | 176 (34.38%) | **135 (26.37%)** |
| Official view-candidate validity | 99.09% | 97.45% | **96.68%** |
| Mean unique valid per view | 7.24 | 8.59 | **8.31** |

Against direct MSE+SIGReg, projected top-1 changed by `-2.148` percentage
points, paired bootstrap 95% CI `[-3.516,-0.781]`, exact McNemar `p=0.00342`
(3 both correct, 12 direct-only, 1 projected-only). Against native, the change
was `-2.734` points, CI `[-4.297,-1.367]`, `p=0.000122`. Direct versus native
on this prefix was not significant (`p=0.629`).

## Recorded comparisons

- Projected `z` effective rank was `3.22 / 3.29`; raw `h` effective rank was
  `12.58 / 5.22`; native raw rank was `41.00 / 22.61`.
- Raw `h` variance was `3.41x / 5.45x` native source/target variance.
- Epoch-4 projected full-auxiliary/NTP cosine and norm ratio were `-0.0300` and
  `2.36x`.
- Target-token CE was `0.256497`, versus `0.248779` direct and `0.240683`
  native. Exact top-1 on the matched 512 was `4`, versus `15` direct and `18`
  native.

## Evidence

- Training result and curves: `runs/projected_mse_sigreg/training/result.json`
- Epoch 1-4 adapters, projectors, and optimizer states:
  `runs/projected_mse_sigreg/training/checkpoints/`
- Epoch-4 projector SHA-256:
  `6c96e4660ee6225f15119d7fdcebcfe9bfe15e33ba0b35c6e6dd662d8762aedd`
- Two-space 256-row geometry:
  `runs/projected_mse_sigreg/evaluation/two_space_epoch4.json`
- Epoch 1/2/4 gradient audit:
  `runs/projected_mse_sigreg/evaluation/gradient_audit.json`
- Matched 256-row CE/interventions:
  `runs/projected_mse_sigreg/evaluation/projected_decoder_coupling_256.json`
- Ordered 512-row manifest, predictions, and paired summaries:
  `runs/projected_mse_sigreg/evaluation/panel_512/`
- Projected 512 prediction SHA-256:
  `2cd5dcb5b6a7cb3eb144e656ad293abd9c31f1489337317437927961034629fc`

The real ChemFM smoke test reached all projector parameters and 154/308 LoRA
gradient tensors on the local RTX 4050. The full local suite and the controlled
A6000 environment each passed 77 tests with one skip before training; final
post-report verification is recorded in the repository handoff.
