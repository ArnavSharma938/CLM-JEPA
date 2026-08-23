# Gradient-interaction execution checkpoint

Status captured 2026-08-23 22:00 UTC. This is a recovery handoff, not the
scientific conclusion; the controlled matrix is still running sequentially on
the sole retained RTX A6000.

## Saved and complete

- Active production code is direct raw-endpoint `MSE + SIGReg`; PCSF and the
  projection head are absent from `src/` and remain historical only.
- Weighted-sum controls at effective JEPA weights 0.25, 0.5, 1.0, and 2.0 have
  complete four-epoch results and checkpoints. Each has 320 updates and the
  identical 172 active JEPA updates.
- Mean active-update `(cosine, conflict fraction, raw JEPA/NTP norm ratio,
  applied JEPA/NTP norm ratio)` is respectively:
  - lambda 0.25: `(0.00352, 0.5407, 0.09829, 0.04914)`;
  - lambda 0.5: `(0.00466, 0.6453, 0.09914, 0.09914)`;
  - lambda 1.0: `(-0.02637, 0.8081, 0.08985, 0.17970)`;
  - lambda 2.0: `(-0.03769, 0.8372, 0.06889, 0.27556)`.
- Exact A6000 evaluation optimization is complete and parity checked:
  representation diagnostics are 3.94x faster; warmed one-view generation is
  1.93x faster; five-view steady throughput is 1.82x faster at 0.28181
  reactions/s.
- The full local suite passes: 84 passed, one intentional skip.
- Local implementation checkpoint commit: `07fac75`.

## Running and incomplete

- PCGrad epoch 1 is saved; later PCGrad epochs are running through the
  restartable matrix runner.
- CAGrad and Du auxiliary-gradient similarity have not yet run.
- Cross-checkpoint held-out gradient audits, seven-condition representation
  diagnostics, one-view generation/CE, and official five-view generation are
  pending.
- The autoregressive verdict must not be inferred from the early two-row
  training validation or from geometry. It remains pending the frozen panels.
- Final work: validate all condition counts, build `summary.json`, replace the
  in-progress Results section in report 09, copy all artifacts locally, rerun
  tests, commit the final report/results, and delete the cloud instance.

The remote runner advances by completion files rather than active polling:
training -> diagnostics -> generation -> `EXPERIMENT_COMPLETE`.
