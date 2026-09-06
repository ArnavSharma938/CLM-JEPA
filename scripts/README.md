# Script map

| Workflow | Entrypoints |
|---|---|
| Oracle transition | run_oracle_transition.py |
| Saved-artifact Report 03 amendment | amend_oracle_transition.py |
| Frozen latent audit | run_latent_audit_pipeline.py, run_latent_predictability_audit.py, analyze_latent_predictability_audit.py |
| STP execution/confirmation | run_stp_matrix.py, run_stp_completion.py, design_stp_confirmation.py, run_stp_confirmation.py |
| STP results/diagnostics | analyze_stp_results.py, analyze_stp_matrix.py, analyze_stp_completion.py, analyze_stp_beams.py, diagnose_stp_checkpoint.py, diagnose_stp_objectives.py |
| Teacher-forced evaluation | eval_teacher_forced_five_view.py, eval_teacher_forced_token_deltas.py |
| Frozen representations | analyze_stp_representations.py |
| Geodesic audit | run_geodesic_audit.py, summarize_geodesic_audit.py, validate_geodesic_capture.py, finalize_geodesic_audit.py |
| Saved candidate analyses | compare_candidate_geometry.py, analyze_candidate_length_controls.py, analyze_signal_uncertainty.py |
| Model download | download_chemfm_model.py |

Oracle artifacts: runs/oracle_transition/. Reports: docs/reports/.
The oracle runner retains extraction for reproducibility; amendments reuse saved
caches and probes. amend_oracle_transition.py compute fits only missing shuffled
probes and rescores saved states; summarize produces compact summaries and Report
03. Neither amendment stage calls ChemFM or generation.

Execution-time source hashes correspond to commit 3b9b9b6, before cleanup.
Retired endpoint, dense V-JEPA, pair-residual, and one-off hardware scripts are
recoverable from that commit.
