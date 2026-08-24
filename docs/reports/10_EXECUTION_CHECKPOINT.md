# Gradient-interaction execution checkpoint

Final status updated 2026-08-24. The controlled matrix and revised 256-reaction
endpoint are complete. This recovery handoff is superseded by the measured
results and verdict in [report 09](09_GRADIENT_INTERACTION_EXPERIMENT.md).

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

## Final completion

- All seven training conditions, cross-checkpoint gradient audits, and
  seven-condition representation diagnostics are complete.
- At the user's explicit post-training scope revision, official behavioral
  evaluation was limited to the first 256 rows of the frozen manifest. Further
  lambda-weight generation was dropped. Native, historical direct MSE+SIGReg,
  and lambda 0.25 are identity-aligned slices of already completed predictions;
  PCGrad, CAGrad, and Du auxiliary similarity are the only new endpoints.
- Direct versus native on the revised official panel is already rescored:
  9/256 versus 11/256 top-1, difference -0.781 pp, 95% paired-bootstrap CI
  [-3.125,+1.562] pp, exact McNemar p=0.7539.
- Lambda 0.25 versus native is 5/256 versus 11/256 top-1, difference -2.344 pp,
  CI [-4.688,-0.391] pp, exact McNemar p=0.0703.
- PCGrad, CAGrad, and Du auxiliary similarity each completed one-view and
  official five-view evaluation on exactly 256 reactions.
- Every retained prediction file has 256 unique identities in exact frozen
  manifest order. All paired native/direct summaries were produced.
- The consolidated local artifact is
  `runs/gradient_interaction/a6000/endpoint_256/summary.json`.
- The autoregressive verdict is negative: PCGrad and CAGrad worsen target-token
  CE; Du adaptation largely suppresses JEPA and approaches but does not improve
  on native CE or official generation. See report 09 for full statistics.
