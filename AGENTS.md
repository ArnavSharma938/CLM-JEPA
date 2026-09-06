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
- ChemFM extraction, generation, training, new experiment families, paid infrastructure, and real-model tests require explicit task authorization. Probe fitting does not authorize ChemFM training.
- Before authorized compute, check inputs, shapes, runtime/memory estimate, output paths, and resumability. Smoke-test the smallest relevant batch; check device utilization once.
- Launch a stage once; retain its process handle and log path. Use completion notifications or do independent work while it runs.
- No continuous polling, including moderate jobs. Check at the estimated completion boundary; increase the interval if late. Read a bounded log tail only for a specific progress/failure question.
- Reuse completed stages. A yielded session is not a failed job; never relaunch it blindly.
- Never run multiple compute instances at once, unless explicitly authorized.

## Verification and artifact safety

- Use syntax/import and focused synthetic tests first. Do not run real-model tests for routine verification.
- Preserve existing checkpoints, caches, probes, and frozen splits. Never delete `runs/`, the workspace, or untracked run artifacts.
- Stage explicit files only. Never commit model/state caches, large run directories, or credentials.
