# Maintained execution scripts

Maintained scientific implementations live in `src/`. This directory now contains only reusable setup/evaluation entrypoints for the current endpoint and dense V-JEPA workflows.

| File | Purpose |
|---|---|
| `audit_vjepa2_1_feasibility.py` | Frozen local target-token CE/CKA and component-gradient summary for native, direct endpoint, and dense V-JEPA checkpoints |
| `design_uspto_mit_endpoint.py` | Freeze and evaluate the prespecified sequential USPTO-MIT endpoint design |
| `download_chemfm_model.py` | Download and SHA-256-check the pinned ChemFM-1B snapshot |
| `run_uspto_mit_official_endpoint.sh` | Run the exact five-view native/direct endpoint comparison and stopping rule |
| `run_vjepa2_1_a6000.sh` | Set up, smoke-test, and run the dense V-JEPA super-mini or controlled pilot |
| `run_vjepa2_1_evaluation_a6000.sh` | Run the matched global and local evaluations after the dense pilot |

The removed scripts were one-off historical audits, completed intervention launchers, profilers, or artifact mergers. Their methods and outputs are preserved in the consolidated reports and compact run summaries; their last full source is recoverable from commit `61fbc74`. They are not dependencies of `src/train.py`.
