# Repository working rules

## Token and command budget

- Start with git status and README's map. Read the relevant report/functions, not every report or source file.
- Search exact symbols and callers with rg; read bounded ranges. Batch independent reads. Default output cap: 2,000 tokens, expanding only for a specific unresolved question.
- Record established paths, completed checks, and next action briefly; do not rediscover unchanged evidence.
- No subagents unless explicitly requested. Give short material updates, not routine command narration.
- Use .venv/Scripts/python.exe on Windows. In PowerShell use rg scripts --glob '*.py', not wildcard paths.
- Check installed versions and relevant documentation before dependency/API changes. Do not install, upgrade, clone, or download to test guesses.
- After failure, inspect the error and make one evidence-backed correction. Repeated deterministic failure is a blocker, not a retry loop.

## Scientific scope and compute

- Reports belong in docs/reports/. README.md is only a repository map.
- For amendments, inspect saved summaries/manifests first; load only necessary caches/probes. Never regenerate available data.
- Report 03: Native r8 seeds 533/917, final post-RMSNorm product states, k=1, frozen 640/192/192 split. Preserve common rows, train-only preprocessing, and decoder target x[t+2].
- ChemFM extraction, generation, training, new experiment families, paid infrastructure, and real-model tests require explicit task authorization. Probe fitting does not authorize ChemFM training.
- Before authorized compute, check inputs, shapes, runtime/memory estimate, output paths, and resumability. Smoke-test the smallest relevant batch; check device utilization once.
- Launch a stage once; retain its process handle and log path. Use completion notifications or do independent work while it runs.
- No continuous polling, including moderate jobs. Check at the estimated completion boundary; increase the interval if late. Read a bounded log tail only for a specific progress/failure question.
- Reuse completed stages. A yielded session is not a failed job; never relaunch it blindly.

## Verification and artifact safety

- Use syntax/import and focused synthetic tests first. Full suite is justified for broad refactors, not every edit. Do not enable real-model tests for routine verification.
- src/train.py supports Native and STP only. Retired endpoint/dense/pair-residual code is recoverable at commit 3b9b9b6.
- Historical predictor/reserved tokens remain necessary for checkpoint compatibility; native decoder audits exclude them explicitly.
- Preserve checkpoints, hidden states/token IDs, PCA/standardizers, probes, embedding/LM-head matrices, frozen splits, ordered beams, raw reaction metrics, hashes, and unique compressed archives.
- Trace callers/report references before cleanup. Delete obsolete tracked code through Git; move uncertain run artifacts to a named recoverable backup with relative paths and hashes.
- Resolve exact source/destination paths before recursive operations. Never broadly delete runs/, a workspace, or a home directory. Untracked does not mean disposable.
- Keep execution-time hashes as historical provenance; record the execution commit when later refactors change source. Do not rewrite measurements to hide source changes.
- Stage explicit files. Force-add only requested compact summaries, never all of runs/. Never commit model/state caches or credentials.
- Finish with verification evidence, commit IDs, and recovery information. State unverified behavior plainly.
