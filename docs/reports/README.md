# CLM-JEPA scientific reports

Historical protocols are consolidated in [PROTOCOL_ARCHIVE.md](PROTOCOL_ARCHIVE.md).
Retired implementations and pre-cleanup analysis source are recoverable at
commit `3b9b9b6`; see [cleanup boundaries](../CLEANUP.md).

This directory contains four current evidence records. The reports separate
measured results from scope limits and do not use representation diagnostics as
substitutes for generated exact match.

## Reports

1. [Pre-STP JEPA experiments](00_PRE_STP_JEPA_CONSOLIDATED.md) consolidates
   former Reports 00--05: method fidelity, endpoint cosine/MSE/SIGReg,
   official generation, frozen mechanism audits, PCSF, projection-space loss,
   gradient combiners, dense causal V-JEPA, and persistent pair-residual
   training.
2. [STP and trajectory geometry](01_STP_AND_TRAJECTORY_GEOMETRY.md) combines
   the frozen base-ChemFM chemical-event/control assay with the complete
   released/paper STP program: rank/lambda comparisons, five-view generation,
   seed-1301 beam analysis, the all-checkpoint representation study, and the
   frozen Geodesic Mechanism Audit of tube scale, intrinsic and decoder-Fisher
   geometry, predictive perpendicular motion, beam trajectories, and inference
   cones.
3. [Untouched STP confirmation and latent predictability](02_STP_CONFIRMATION_AND_LATENT_PREDICTABILITY.md)
   records the independent 640-reaction/four-seed confirmation and the frozen
   future-state predictability, decoder-coupling, chemical-view invariance,
   candidate replay, and joint predictability-by-invariance audit.
4. [Native oracle transition decomposition](03_NATIVE_ORACLE_TRANSITION_DECOMPOSITION.md)
   records the frozen final-layer k=1 oracle-token and short-history probes
   for Native seeds 533/917, their decoder preservation, and paired uncertainty.

## Endpoint summary

| Program | Primary recorded comparison |
|---|---|
| Endpoint MSE+SIGReg | Official five-view top-1 `40/1,280` vs Native `50/1,280`; difference `-.781` pp, 95% CI `[-1.719,+.156]` |
| Persistent pair residual | Three-seed mean top-1 effect `-1.04` pp; crossed CI `[-3.52,+1.17]`; preregistered verdict **FAIL** for the tested trajectory |
| Released STP r8/.02 | Seed effects `+2.15,+.98,-1.17` pp; mean `+.65`; crossed CI `[-1.17,+2.54]` |
| Paper STP r8/.02 | Seed effects `+1.37,+1.37` pp; crossed CI `[+.39,+2.54]` on the repeatedly used development panel |
| STP development program | Report 01's repeatedly used development panel was **INCONCLUSIVE**; confirmation status is reported separately below |
| Literal final-layer Euclidean geodesic mechanism | Not supported: no small-radius/positive-persistence local tube, no consistent STP straightening, and perpendicular motion remains predictively active |
| Released STP untouched confirmation | Four new seed effects `+.469,+.625,+.625,+.469` pp; mean `+.547` pp; crossed CI `[-.195,+1.328]`; Holm p `.1025`; **INCONCLUSIVE** |
| Paper STP untouched confirmation | Four new seed effects `+.625,+.781,-.938,0` pp; mean `+.117` pp; crossed CI `[-.898,+1.133]`; not supported on the confirmation panel |
| Frozen latent mechanism | Native final-product `R2=.641,.432,.161,.008` at k=1/2/4/8; neither STP increases predictable fraction, decoder preservation, or final cross-view invariance consistently |

Former report text and historical implementation paths remain recoverable from
Git history. Decision-relevant protocols, measurements, uncertainty, limits,
and artifact paths are retained in the consolidated reports.
