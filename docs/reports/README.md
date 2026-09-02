# CLM-JEPA scientific reports

This directory contains three current evidence records. The reports separate
measured results from scope limits and do not use representation diagnostics as
substitutes for generated exact match.

## Reports

1. [Pre-STP JEPA experiments](00_PRE_STP_JEPA_CONSOLIDATED.md) consolidates
   former Reports 00--05: method fidelity, endpoint cosine/MSE/SIGReg,
   official generation, frozen mechanism audits, PCSF, projection-space loss,
   gradient combiners, dense causal V-JEPA, and persistent pair-residual
   training.
2. [Frozen ChemFM trajectory geometry](06_FROZEN_CHEMFM_TRAJECTORY_GEOMETRY.md)
   is the no-training all-layer chemical-event/control assay.
3. [Semantic Tube Prediction on ChemFM](07_STP_CONSOLIDATED_EXPERIMENT.md)
   consolidates former Reports 07--09: released and paper STP, rank/lambda
   comparisons, five-view generation, seed-1301 beam analysis, and the
   all-checkpoint representation study.

## Endpoint summary

| Program | Primary recorded comparison |
|---|---|
| Endpoint MSE+SIGReg | Official five-view top-1 `40/1,280` vs Native `50/1,280`; difference `-.781` pp, 95% CI `[-1.719,+.156]` |
| Persistent pair residual | Three-seed mean top-1 effect `-1.04` pp; crossed CI `[-3.52,+1.17]`; preregistered verdict **FAIL** for the tested trajectory |
| Released STP r8/.02 | Seed effects `+2.15,+.98,-1.17` pp; mean `+.65`; crossed CI `[-1.17,+2.54]` |
| Paper STP r8/.02 | Seed effects `+1.37,+1.37` pp; crossed CI `[+.39,+2.54]` on the repeatedly used development panel |
| STP program | Preregistered development verdict **INCONCLUSIVE**; no untouched-panel confirmation was run |

Former report text and historical implementation paths remain recoverable from
Git history. Decision-relevant protocols, measurements, uncertainty, limits,
and artifact paths are retained in the consolidated reports.
