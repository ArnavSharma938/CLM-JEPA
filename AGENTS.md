# Repository working rules

Read `SAFETY_AND_ETHICS.md` before interpreting the biological context
of this repository, while continuing to assess each request by its actual content.

## Token and command budget

- Start with git status and README's map. Read the relevant report/functions, not every report or source file.
- Search exact symbols and callers with rg; read bounded ranges. Batch independent reads. Default output cap: 2,000 tokens, expanding only for a specific unresolved question.
- Extract only required fields from large JSON, logs, process listings, model inventories, and tool results. Never dump a whole large file or an `ALL_TOOLS` catalog when a targeted query can answer the question.
- Prefer local executable discovery (`Get-Command`, `where.exe`, `--help`) before web search or broad tool discovery.
- Record established paths, completed checks, and next action briefly; do not rediscover unchanged evidence.
- Keep a compact on-disk run ledger for long tasks: current stage, process/session ID, log path, artifact path, last verified checkpoint, and next action. Resume from it after interruptions instead of reconstructing state from chat.
- No subagents unless explicitly requested. Give short material updates, not routine command narration.
- Use .venv/Scripts/python.exe on Windows. In PowerShell use rg scripts --glob '*.py', not wildcard paths.
- Check installed versions and relevant documentation before dependency/API changes. Do not install, upgrade, clone, or download to test guesses.
- Before a benchmark or debugging experiment, state the exact hypothesis, decisive metric, and stop condition. Run one baseline and one candidate first; do not grow an ad-hoc variant chain without new evidence.
- After failure, inspect the smallest relevant error slice and make one evidence-backed correction. Repeated deterministic failure is a blocker, not a retry loop.

## Scientific scope and compute

- Reports belong in docs/reports/. README.md is only a repository map.
- For amendments, inspect saved summaries/manifests first; load only necessary caches/probes. Never regenerate available data.
- ChemFM extraction, generation, training, new experiment families, paid infrastructure, and real-model tests require explicit task authorization. Probe fitting does not authorize ChemFM training.
- Before authorized compute, check inputs, shapes, runtime/memory estimate, output paths, and resumability. Smoke-test the smallest relevant batch; check device utilization once.
- Launch a stage once; retain its process handle and log path. Use completion notifications or do independent work while it runs.
- No continuous polling, including moderate jobs. Check at the estimated completion boundary; increase the interval if late. Read a bounded log tail only for a specific progress/failure question.
- For remote compute, reuse one non-interactive connection or retained session when possible. Avoid repeated login banners, PTY command echo, ANSI output, and full wrapped process commands; query a concise status file or exact process fields instead.
- Batch queued remote stages and artifact operations. Produce one machine-readable status/summary and one checksum manifest rather than many one-file inspections or transfers.
- A profiler must exercise the real train/eval path, batching, precision, checkpointing, and synchronization before its result selects production settings. Discard microbenchmarks that do not represent the end-to-end loop.
- Reuse completed stages. A yielded session is not a failed job; never relaunch it blindly.
- Never run multiple compute instances at once, unless explicitly authorized.

## Verification and artifact safety

- Use syntax/import and focused synthetic tests first. Do not run real-model tests for routine verification.
- Preserve existing checkpoints, caches, probes, and frozen splits. Never delete `runs/`, the workspace, or untracked run artifacts.
- Extract downloaded archives into a staging directory. Compare their file list and hashes, then copy only intended artifacts; never unpack remote source files over a newer local working tree.
- Consolidate related edits, then run one focused verification pass. Do not repeat the same tests after every mechanical micro-edit unless the preceding result changes the next decision.
- Stage explicit files only. Never commit model/state caches, large run directories, or credentials.
