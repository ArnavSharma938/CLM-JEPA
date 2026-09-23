# ESM-2 Gate-1 handoff

Updated: 2026-09-22. This is the resume ledger; do not reconstruct this work from chat or rerun completed stages.

## Final state

- Experiment: ESM-2-650M adaptation to Dayhoff BackboneRef, 10k and nested 50k loads.
- Decision: **No meaningful LoRA gap through 50k. Stop this ESM-2 setting before H1-H4 or scale expansion.**
- Queue completed successfully (`protein/runs/esm2_gate1/queue.exit` = 0).
- FT at 50k was correctly skipped because 10k FT and DTFT were equivalent.
- The borderline 10k LoRA-r8 standard result was replicated with seed 23 and remained below the registered 10% threshold.
- Thunder instance `r9xiha5y` was deleted; `tnr status --no-wait` returned no instances. Do not provision another instance for this completed screen.

## Canonical artifacts

- Scientific report: `protein/docs/reports/ESM2_650M_BACKBONEREF_GATE1.md`
- Machine-readable result: `protein/runs/esm2_gate1/gate1_summary.json`
- Transfer hashes: `protein/runs/esm2_gate1/transfer_inventory.sha256`
- Profiling: `protein/runs/esm2_gate1/execution_profile.json`
- Backend benchmark: `protein/runs/esm2_gate1/backend_benchmark.json`
- HPO selections: `protein/runs/esm2_gate1/hpo/10k/<method>/selection.json`
- Compact trainable checkpoints: `protein/runs/esm2_gate1/evidence/<load>/<method>/seed_<seed>/best.trainable.pt`
- External backup: `C:\Users\arnav.DHEERAJACER\ThunderBackups\r9xiha5y_20260922`
  - results archive SHA-256: `3e8fd1da91c1b8abce8c322c2188decef1f58b6fc625cfbea8ef3542c667d147`
  - compact-checkpoint archive SHA-256: `19c9a561047bd69f0f2cec76a19490556a5067bf28bbed9bfe4ccc6905e2b731`

The report contains full NLLs, bootstrap intervals, manifests, hashes, retention, runtime, and limitations. Query the JSON/report for exact values; do not dump whole files into chat.

## Pinned provenance

- Dayhoff: `microsoft/Dayhoff@7f6771db1bfad58011fcadf72f86eacbef39a482`
- ESM-2: `facebook/esm2_t33_650M_UR50D@08e4846e537177426273712802403f7ba8261b6c`
- Model weight SHA-256: `a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0`
- Structure ZIP SHA-256: `5c75396fe6e0229f4e4a8dbeab7b88b95cd8e89c47137cf83f9e33d3fb40270a`
- Run source commit/hash: `d09b5fb28cae23363d7cc1527194bd55bba80c3a` / `c076a70023fe07fec5aaa4760d85f80d42fe9e391b8bbb1bee18f7e375b4ced3`
- Corpus counts: BRn 138,044; BRq 127,633; intersection 54,324; 58 preferred-intersection low-complexity removals; no BRn-only fallback.
- Structure identity uses accession/PDB stem. Never join the released structure Arrow rows to sequences by row position.

## Selected execution settings

- Attention: SDPA; BF16.
- Train: 12,288 microbatch tokens, 2 workers, no gradient checkpointing, no `torch.compile`.
- Evaluation: 16,384 tokens, 2 workers, no `torch.compile`; reuse one loaded model across standard, OOD, and natural sets.
- Effective non-padding residue budget was held constant across methods.
- Learning rates: FT/DTFT `1e-5`; LoRA-r8/r64 `1e-4`.
- Evidence seed 11; seed 7 was HPO only; seed 23 was used only for the triggered 10k DTFT/LoRA-r8 replication.

## Decisive results

- Base NLL: standard 2.087370; structural OOD 2.165268; natural 1.939585.
- At 10k, `F_gap`: r8 standard 0.081, r8 OOD 0.062; r64 standard -0.056, r64 OOD -0.053.
- Seed-23 10k r8 replication: standard 0.077, OOD 0.068; not a replicated candidate gap.
- At 50k, `F_gap`: r8 standard 0.053, r8 OOD 0.053; r64 standard -0.007, r64 OOD 0.002.
- FT-vs-DTFT loss of adaptation gain at 10k: 0.8% standard and 1.0% OOD; target coverage is not a material confound.
- Remote OOD contains 111 backbones because the fixed TM-score threshold, not desired sample count, controlled membership.

## Resume and token-conservation rules

1. Read this ledger, then query only the canonical artifact needed for the current question.
2. Do not rerun data preparation, distance search, profiling, HPO, training, evaluation, bootstrap, or reporting unless the user explicitly reopens the experiment.
3. Never provision compute merely to inspect results; all required outputs and compact checkpoints are local.
4. Extract JSON fields programmatically. Use bounded log tails and exact process fields; never print full inventories, logs, or tool catalogs.
5. For future remote runs, maintain one status JSON containing stage, PID/session, log, checkpoint, progress, and exit code. Reuse one non-interactive connection and obey the requested check interval.
6. Before any new benchmark, record its hypothesis, decisive metric, and stop condition. Start with one baseline and one candidate.
7. Extract archives to staging, compare hashes/file lists, and selectively copy artifacts. Never unpack remote source over the working tree.
8. Keep updates to outcome, current stage, failure if any, and evidence-based ETA; do not repeat the experimental background.

## Working-tree caveat

The direct remote archive extraction replaced the local Gate-1 test file with the tracked remote copy. The post-transfer suite passed 40 tests with one existing NumPy warning, whereas an earlier local suite had 46 tests. Six uncommitted tests may therefore have been lost. Do not claim that the current tree contains those six tests; recover them from any available editor/local-history backup before further Gate-1 code changes. Current intentional uncommitted items include `AGENTS.md` and `protein/runs/esm2_gate1/transfer_inventory.sha256`.
