# Contraction and held-out NTP directional audit

## Measured summary

- On the same fixed batches, native pair-center spread fell from `0.05229` at epoch 1 to `0.03609` at epoch 4 (`-31.0%`). MSE-only was already `15.3%` below time-matched native at epoch 1 and `47.4%` below it at epoch 2.
- The active MSE step contracted pair-center spread in every checkpoint/batch measurement (`33/33`). In contrast, the correctly paired residual `2(g_true-g_shuffle)` increased spread in `33/33` measurements.
- When active, PCSF increased spread in every measurement. At MSE+SIGReg epochs 2 and 4, applied SIGReg more than reversed MSE contraction in all six measurements.
- At MSE+SIGReg epochs 2 and 4, the pair-specific MSE residual reduced held-out NTP loss in `6/6` measurements, while total MSE increased it in `6/6`.
- The applied SIGReg step increased held-out loss in `5/6` epoch-2/4 measurements, including `3/3` at epoch 4. The PCSF step reduced held-out loss in `9/9` low-spread measurements.

## Method

The assay used the existing fixed USPTO-MIT serialization and k=0 source/target EOS readout. Three seed-533 training batches of 16 reactions supplied auxiliary gradients. Each was paired with a different, disjoint batch of 16 reactions from the frozen validation panel for the NTP evaluation gradient. The same batches were used at every state.

| Condition | Frozen epochs |
|---|---|
| Matched native | 1, 2, 4 |
| MSE-only | 1, 2 |
| PCSF | 1, 2, 4 |
| MSE+SIGReg-16 | 1, 2, 4 |

All reported parameter-space directions use every trainable LoRA A/B tensor, matching the established mechanistic audit. Large `modules_to_save` token-I/O tensors were excluded from this focused audit because materializing all objectives for them caused host-memory paging; prior work treated token-I/O separately. The model used non-reentrant checkpointing, exact native+source+target padded-row endpoint computation, dropout disabled, and two independently seeded SIGReg projection draws averaged per measurement. The source/target-only BF16 candidate had `11.9%` LoRA-gradient relative error and was not used.

No optimizer was constructed. Trainable-state fingerprints were unchanged before and after all 11 checkpoints. The run completed 33 checkpoint-batch measurements in `2,020.9 s` on the local RTX 4050.

For an objective gradient `g`, the spread velocity is

```text
v(g) = - grad(sigma_PC) dot g
```

Positive values increase pair-center spread under an infinitesimal descent step. For held-out NTP,

```text
Delta L_eval(g) = - grad(L_NTP,heldout) dot g
```

Negative values improve held-out NTP; positive values worsen it. These are local first-order changes per unit learning rate, not finite-step loss predictions.

Actual active-step coefficients were retained:

```text
MSE-only:       2 MSE
PCSF:           2 MSE + 8.4 PCSF
MSE+SIGReg:     2 MSE + 0.08080808 SIGReg
```

The 50% auxiliary dropout halves their expectation without changing direction. Raw and applied values are retained in the machine artifact.

## Time-matched spread trajectory

`r(t)` is the mean batchwise current spread divided by native spread at the same epoch.

| State | Pair-center sigma | `r(t)` |
|---|---:|---:|
| Native e1 | 0.05229 | 1.000 |
| Native e2 | 0.04213 | 1.000 |
| Native e4 | 0.03609 | 1.000 |
| MSE e1 | 0.04471 | 0.847 |
| MSE e2 | 0.02219 | 0.526 |
| PCSF e1 | 0.04604 | 0.873 |
| PCSF e2 | 0.02583 | 0.611 |
| PCSF e4 | 0.01963 | 0.543 |
| MSE+SIGReg e1 | 0.05651 | 1.078 |
| MSE+SIGReg e2 | 0.03584 | 0.855 |
| MSE+SIGReg e4 | 0.03329 | 0.925 |

The instantaneous native NTP spread direction was mixed at epoch 1 and positive in `3/3` batches at epochs 2 and 4. The MSE direction was negative in every batch.

## Spread-direction measurements

The table reports the mean actual active-step spread velocity. Values are multiplied by `1e3`; signs in parentheses give batches with positive/negative velocity. `g_pair=2(g_true-g_shuffle)` is the applied pair-specific residual. `g_reg` and `g_full` use the regularizer and full objective appropriate to that condition.

| State | `v_NTP` | `v_MSE` | `v_pair` | `v_reg` | `v_full` |
|---|---:|---:|---:|---:|---:|
| Native e1 | -0.641 (1/2) | -22.45 (0/3) | +6.566 (3/0) | - | -22.45 |
| Native e2 | +2.490 (3/0) | -37.48 (0/3) | +5.143 (3/0) | - | -37.48 |
| Native e4 | +3.716 (3/0) | -30.51 (0/3) | +3.995 (3/0) | - | -30.51 |
| MSE e1 | -1.094 (0/3) | -14.94 (0/3) | +0.964 (3/0) | - | -14.94 |
| MSE e2 | +0.041 (2/1) | -1.863 (0/3) | +0.374 (3/0) | - | -1.863 |
| PCSF e1 | +0.150 (1/2) | -5.332 (0/3) | +1.065 (3/0) | 0; floor inactive | -5.332 |
| PCSF e2 | +0.002 (2/1) | -4.385 (0/3) | +0.516 (3/0) | +0.411 (3/0) | -3.973 (0/3) |
| PCSF e4 | +0.512 (3/0) | -1.028 (0/3) | +0.312 (3/0) | +0.732 (3/0) | -0.296 (1/2) |
| MSE+SIGReg e1 | -6.664 (0/3) | -7.730 (0/3) | +4.294 (3/0) | +6.566 (3/0) | -1.164 (1/2) |
| MSE+SIGReg e2 | +0.595 (2/1) | -5.078 (0/3) | +7.282 (3/0) | +8.495 (3/0) | +3.417 (3/0) |
| MSE+SIGReg e4 | +0.123 (1/2) | -4.498 (0/3) | +7.484 (3/0) | +7.482 (3/0) | +2.984 (3/0) |

The raw MSE gradient cosine with `grad(sigma_PC)` ranged from `+0.728` to `+0.907` across trained states. The pair-specific residual cosine ranged from `-0.417` to `-0.970`, with positive spread velocity in all 33 measurements.

PCSF is anti-aligned with the spread gradient while its hinge is active. It offset about `9%` of MSE's negative spread velocity at PCSF epoch 2 and `71%` at epoch 4. At the MSE-only epoch-2 state, the calibrated PCSF term offset about `42%`.

At MSE+SIGReg epochs 2 and 4, the applied SIGReg positive spread velocity was `1.67x` and `1.66x` the magnitude of the MSE negative spread velocity. The full auxiliary spread velocity was positive in `6/6` measurements.

## Held-out NTP directional measurements

The table reports mean held-out `Delta L_NTP` per unit learning rate, multiplied by `1e3`, for the most informative trained states. Negative is favorable. Parentheses again give positive/negative batch counts, so `(3/0)` is consistently harmful and `(0/3)` consistently helpful.

| State | MSE | Pair-specific residual | Regularizer | Full auxiliary |
|---|---:|---:|---:|---:|
| MSE e2 | +0.144 (2/1) | -0.115 (0/3) | - | +0.144 (2/1) |
| PCSF e2 | +0.108 (2/1) | -0.084 (0/3) | -0.038 (0/3) | +0.071 (2/1) |
| PCSF e4 | -0.019 (2/1) | -0.016 (2/1) | -0.046 (0/3) | -0.065 (1/2) |
| MSE+SIGReg e2 | +0.469 (3/0) | -0.527 (0/3) | +0.409 (2/1) | +0.877 (2/1) |
| MSE+SIGReg e4 | +0.209 (3/0) | -0.474 (0/3) | +1.068 (3/0) | +1.277 (3/0) |

The measured signs across all states were:

1. At MSE-only epochs 1/2, the pair-specific residual improved held-out NTP in `6/6` measurements. It also improved held-out NTP in `6/6` MSE+SIGReg epoch-2/4 measurements.
2. Total MSE increased held-out loss in `6/6` MSE+SIGReg epoch-2/4 measurements and had mixed signs elsewhere.
3. PCSF's applied step reduced held-out loss in every low-spread measurement where its hinge was active (`9/9`: MSE e2 plus PCSF e2/e4).
4. The MSE+SIGReg full objective increased held-out NTP loss in `5/6` epoch-2/4 measurements and in `3/3` at epoch 4.

## Recorded directional comparisons

- Native spread decreased `31.0%` from epoch 1 to epoch 4. MSE spread was `15.3%` and `47.4%` below time-matched native at epochs 1 and 2.
- MSE spread velocity was negative in `33/33` measurements; pair-residual velocity was positive in `33/33`.
- PCSF and SIGReg spread velocities were positive wherever reported active. The combined PCSF objective retained negative mean velocity at epochs 2 and 4; the combined SIGReg objective had positive velocity in `6/6` epoch-2/4 measurements.
- At MSE+SIGReg epochs 2 and 4, pair-residual held-out NTP change was negative in `6/6`, total MSE change was positive in `6/6`, and full-objective change was positive in `5/6`.

## Limits and artifacts

- This is an exact local first-order assay at frozen states, not a reconstruction of the full historical optimizer trajectory.
- Three independent train/held-out batch pairs establish sign consistency but do not estimate a population effect precisely.
- The result concerns LoRA A/B update space. It does not supersede the prior token-I/O swap evidence.

Artifacts:

- Machine results: `runs/diagnostics/contraction_ntp_directional_audit/audit.json`
- Compact extraction: `runs/diagnostics/contraction_ntp_directional_audit/primary_tables.txt`
- Execution logs: `runs/diagnostics/contraction_ntp_directional_audit/run.stdout.log`, `run.stderr.log`
- Implementation: `scripts/audit_contraction_ntp_direction.py`
- Focused tests: `tests/test_contraction_ntp_direction_audit.py`
