# Cleanup and recovery — 2026-09-06

Report 03's completed amendment and its execution-time source are committed at
`3b9b9b6`. The subsequent cleanup does not rerun extraction, generation, or
ChemFM training, and does not change scientific measurements.

Retired endpoint JEPA, gradient-combiner, dense V-JEPA, and persistent
pair-residual implementations, their dedicated tests, and obsolete hardware
wrappers were removed. Their source remains recoverable from `3b9b9b6`.
The shared trainer now supports Native and released/paper STP only. Vocabulary
and adapter-loading compatibility helpers live in `src/chemfm.py`; historical
logging field names are retained for existing STP consumers.

Native/STP, frozen geometry/latent/oracle infrastructure, report analyses,
checkpoint evaluation, and their relevant tests remain. Historical research
plans/context maps were retired; README is the current map. The four original
preregistration documents are consolidated, with original hashes and text, in
[the protocol archive](reports/PROTOCOL_ARCHIVE.md).

137 untracked transfer bundles, routine logs/PID files, and temporary maintenance
scripts (1,214,916,114 bytes) were moved out of `runs/`, not destroyed. Every move
was hash-verified. [The manifest](cleanup_manifest.json) records original
run-relative paths, byte counts, hashes, and the local backup root:

`C:/Users/arnav.DHEERAJACER/CLM-JEPA-cleanup-backups/20260906`

To recover a file, copy its manifest-relative path from that backup to the same
relative path under `runs/`, without overwriting an existing file, and verify its
SHA-256. The backup is local, not in Git; preserve it separately if moving machines.

All standalone checkpoints, state caches, PCA/probe artifacts, frozen splits,
raw scientific metrics/beams, unique compact result archives, and all oracle
artifacts remain in place. Historical run evidence is intentionally retained
even where the old implementation is retired. Model/data/reference directories
were not cleaned. Source hashes in the oracle manifest describe the execution
version at `3b9b9b6`, not later import-only cleanup changes.
