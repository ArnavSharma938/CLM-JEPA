# Contraction and held-out NTP directional audit

## Result

Representation contraction is partly a normal native-fine-tuning effect, but the much larger cLM-JEPA contraction is imposed by the **pair-blind component of raw endpoint MSE**, not by the reaction-pair-specific component.

- On the same fixed batches, native pair-center spread fell from `0.05229` at epoch 1 to `0.03609` at epoch 4 (`-31.0%`). MSE-only was already `15.3%` below time-matched native at epoch 1 and `47.4%` below it at epoch 2.
- The active MSE step contracted pair-center spread in every checkpoint/batch measurement (`33/33`). In contrast, the correctly paired residual `2(g_true-g_shuffle)` increased spread in `33/33` measurements.
- PCSF and SIGReg act on the relevant direction rather than merely having large norms. When active, PCSF increased spread in every measurement, but its applied force did not consistently reverse MSE. At MSE+SIGReg epochs 2 and 4, applied SIGReg more than reversed MSE contraction in all six measurements.
- The pair-specific MSE residual had a small but consistent favorable held-out NTP effect at the important trained states. At MSE+SIGReg epochs 2 and 4 it reduced held-out NTP loss in `6/6` measurements, while total MSE increased it in `6/6`.
- SIGReg's geometry repair was adverse to held-out NTP at the trained MSE+SIGReg endpoint: its applied step increased held-out loss in `5/6` epoch-2/4 measurements, including `3/3` at epoch 4. PCSF itself was favorable in `9/9` low-spread measurements, but MSE+PCSF remained approximately neutral because the non-pair-specific MSE component persisted.

The result does not support treating contraction as the sole bottleneck. Correct-pair learning is neither the source of contraction nor completely decoupled from autoregressive prediction. The measured failure is more specific: raw MSE is dominated by a shared alignment direction that contracts the representation and is neutral or adverse to held-out NTP; SIGReg fixes its scale while adding another pair-blind, later adverse update.

## Method

The assay used the existing fixed USPTO-MIT serialization and k=0 source/target EOS readout. Three seed-533 training batches of 16 reactions supplied auxiliary gradients. Each was paired with a different, disjoint batch of 16 reactions from the frozen validation panel for the NTP evaluation gradient. The same batches were used at every state.

| Condition | Frozen epochs |
|---|---|
| Matched native | 1, 2, 4 |
| MSE-only | 1, 2 |
| PCSF | 1, 2, 4 |
| MSE+SIGReg-16 | 1, 2, 4 |

All reported parameter-space directions use every trainable LoRA A/B tensor, matching the established mechanistic audit. Large `modules_to_save` token-I/O tensors were excluded from this focused audit because materializing all objectives for them caused host-memory paging; prior work treated token-I/O separately. The model used non-reentrant checkpointing, exact native+source+target padded-row endpoint computation, dropout disabled, and two independently seeded SIGReg projection draws averaged per measurement. The faster source/target-only BF16 path was measured and rejected for this GPU because its LoRA-gradient relative error was `11.9%`.

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

Native NTP therefore produces a meaningful ordinary scale change, but it does not account for the additional MSE contraction. The instantaneous native NTP spread direction was near zero/mixed at epoch 1 and **expansive in 3/3 batches** at epochs 2 and 4, whereas MSE remained contractive in every batch.

## Who causes contraction?

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

The raw MSE gradient was strongly aligned with `grad(sigma_PC)`: mean cosine ranged from `+0.728` to `+0.907` across the trained states. Gradient descent on MSE therefore contracts spread. The pair-specific residual had the opposite direction, with spread-gradient cosine from `-0.417` to `-0.970` and positive velocity in all 33 measurements. Absolute endpoint alignment, not correct-pair specificity, creates the contraction shortcut.

PCSF is exactly anti-aligned with the spread gradient while its hinge is active, as its definition implies. The relevant finding is its applied magnitude: it offset about `9%` of MSE's contractive velocity at PCSF epoch 2 and `71%` at epoch 4, leaving the mean full direction contractive. At the failed MSE-only epoch-2 state, the calibrated PCSF term offset about `42%`. This explains why the floor was directionally correct yet not enforced.

SIGReg also opposed contraction. At its own epochs 2 and 4 its applied expansive velocity was about `1.67x` and `1.66x` the MSE contraction, respectively, making the full auxiliary direction expansive in `6/6` measurements. Its geometry repair is therefore a genuine directional effect.

## Is the JEPA update useful to held-out NTP?

The table reports mean held-out `Delta L_NTP` per unit learning rate, multiplied by `1e3`, for the most informative trained states. Negative is favorable. Parentheses again give positive/negative batch counts, so `(3/0)` is consistently harmful and `(0/3)` consistently helpful.

| State | MSE | Pair-specific residual | Regularizer | Full auxiliary |
|---|---:|---:|---:|---:|
| MSE e2 | +0.144 (2/1) | -0.115 (0/3) | - | +0.144 (2/1) |
| PCSF e2 | +0.108 (2/1) | -0.084 (0/3) | -0.038 (0/3) | +0.071 (2/1) |
| PCSF e4 | -0.019 (2/1) | -0.016 (2/1) | -0.046 (0/3) | -0.065 (1/2) |
| MSE+SIGReg e2 | +0.469 (3/0) | -0.527 (0/3) | +0.409 (2/1) | +0.877 (2/1) |
| MSE+SIGReg e4 | +0.209 (3/0) | -0.474 (0/3) | +1.068 (3/0) | +1.277 (3/0) |

These effects are small and mostly near-orthogonal in cosine terms, so magnitude should not be overstated. Their sign structure is nevertheless informative:

1. At MSE-only epochs 1/2, the pair-specific residual improved held-out NTP in `6/6` measurements. It also improved held-out NTP in `6/6` MSE+SIGReg epoch-2/4 measurements.
2. Total MSE did not preserve that benefit. It was adverse in `6/6` MSE+SIGReg epoch-2/4 measurements and predominantly adverse or mixed elsewhere.
3. PCSF's applied restorative step improved held-out NTP in every low-spread measurement where its hinge was active (`9/9`: failed MSE e2 plus PCSF e2/e4). It was not the source of task conflict, but its full objective remained near zero because MSE retained its non-pair-specific component.
4. SIGReg repaired spread but was adverse at the later trained state. The MSE+SIGReg full objective increased held-out NTP loss in `5/6` epoch-2/4 measurements and in `3/3` at epoch 4.

## Causal answer

### 1. Who causes contraction?

Native fine-tuning establishes a moderate contracting trajectory, but the extreme additional contraction comes from raw MSE's pair-insensitive alignment direction. Correct-pair-specific MSE pressure moves in the opposite direction and preserves/increases reaction spread. PCSF and SIGReg both counteract the actual contractive direction; PCSF was too weak under the tested dynamics, whereas SIGReg reversed it.

### 2. Is the JEPA gradient useful to autoregressive prediction?

The total endpoint-MSE update is not reliably useful to held-out NTP. However, its pair-specific residual is modestly useful at the trained states. The evidence therefore rejects both simple extremes:

- contraction is not merely ChemFM containing a wholly useless pair-specific objective;
- repairing contraction alone is not sufficient, because SIGReg restores spread while worsening the task-facing direction.

The bottleneck is the composition of the endpoint objective: a useful but small pair-specific component is diluted by a larger common alignment component, and SIGReg adds a pair-blind spreading component that is adverse to later held-out NTP. Further coefficient or anti-collapse optimization alone is not justified. The next mechanistic work should isolate how to retain the measured pair-specific MSE residual while removing the shared endpoint-alignment force; it should not begin by strengthening PCSF/SIGReg or by assuming the JEPA signal has no autoregressive coupling.

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
