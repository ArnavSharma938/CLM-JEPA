# Maintained execution and analysis scripts

Scientific model implementations live in `src/`; this directory contains the
reusable experiment, evaluation, and analysis entrypoints retained by the two
consolidated reports.

| Workflow | Scripts |
|---|---|
| Official endpoint | `design_uspto_mit_endpoint.py`, `run_uspto_mit_official_endpoint.sh`, `eval_teacher_forced_five_view.py` |
| Dense V-JEPA | `audit_vjepa2_1_feasibility.py`, `run_vjepa2_1_a6000.sh`, `run_vjepa2_1_evaluation_a6000.sh` |
| Pair residual | `run_pair_residual_a6000.sh`, `run_pair_residual_local.py`, `analyze_pair_residual_results.py` |
| STP execution | `run_stp_a6000.sh`, `run_stp_local.py`, `run_stp_matrix.py`, `run_stp_completion.py` |
| STP endpoint/mechanism analysis | `analyze_stp_results.py`, `analyze_stp_matrix.py`, `analyze_stp_completion.py`, `analyze_stp_beams.py`, `diagnose_stp_checkpoint.py`, `diagnose_stp_objectives.py`, `eval_teacher_forced_token_deltas.py` |
| Frozen representation study | `analyze_stp_representations.py` |
| Geodesic Mechanism Audit | `run_geodesic_audit.py`, `summarize_geodesic_audit.py`, `validate_geodesic_capture.py`, `finalize_geodesic_audit.py` |
| Geodesic derived controls | `compare_candidate_geometry.py`, `analyze_candidate_length_controls.py`, `analyze_signal_uncertainty.py` |
| Frozen oracle next-token transition | `run_oracle_transition.py` |

The geodesic remote shell wrappers reproduce the completed Thunder sequence,
but the hardware-neutral scientific entrypoint is `run_geodesic_audit.py`.
Historical one-off source remains recoverable from Git history. Current
protocols, result boundaries, and artifact hashes are recorded in
`docs/reports/README.md` and the linked consolidated reports.

The oracle runner is scoped to Native r8 seeds 533/917, product final
post-RMSNorm states, and common `t-3...t+2` rows on the frozen audit split.
Its stages are `smoke`, `extract`, `probe-smoke`, `ridge`, `mlp`, and
`summarize`, in that order. It saves compact product-state caches, actual
checkpoint embedding/LM-head matrices, train-only PCA and standardizers,
fitted probes, reaction-level metrics, and paired uncertainty under
`runs/oracle_transition/`. The scientific report is saved to
`docs/reports/03_NATIVE_ORACLE_TRANSITION_DECOMPOSITION.md`.
Completed artifacts are reused; this path
never trains ChemFM or generates tokens. The saved caches are local ignored
run artifacts and must be retained separately from Git.

For the Report 03 amendment, `amend_oracle_transition.py compute` uses only
saved oracle caches and probes: it measures the test PCA ceiling, fits the
six missing nonlinear shuffled-token probes, and scores the native vocabulary.
`amend_oracle_transition.py summarize` regenerates the amended report and
compact audit summaries. Original full-head metrics remain robustness-only;
neither amendment stage calls ChemFM or generation.
