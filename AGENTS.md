# Token-Conserving Execution

Optimize for the fewest model turns, file reads, tool calls, and command retries needed to complete the task correctly. Do not save tokens by weakening reasoning, skipping relevant documentation, omitting necessary source inspection, or avoiding verification.

## Scope first

- Identify the exact requested outcome, constraints, relevant paths, and completion test before acting.
- Use an existing plan or specification as the source of truth. Do not create a second plan that restates it.
- For large plans, extract only a compact requirement checklist and load the relevant section for the current step.
- Do not broaden the task, refactor unrelated code, or investigate optional improvements unless required.

## Read selectively

- Read applicable `AGENTS.md`, the governing plan, and high-signal repository docs before implementation.
- Map before scanning: inspect a shallow directory listing, manifests, entry points, tests, and exact referenced files.
- Search for exact symbols, filenames, error text, imports, and call sites before opening broad files.
- Read the smallest useful line ranges. Do not recursively read the whole repository by default.
- Do not reread unchanged files or repeat repository discovery. Maintain a terse working map of facts already established.
- Ignore generated outputs, dependencies, caches, checkpoints, datasets, binaries, build directories, and large lockfiles unless directly relevant.
- Treat source code and current configuration as authoritative; compact notes are indexes, not substitutes for checking the relevant implementation.

## Documentation before guessing

- Before unfamiliar or version-sensitive setup, installation, Git, API, SDK, framework, cloud, or CLI actions, inspect local documentation and current official documentation.
- Verify the installed version, operating system, shell, working directory, required tools, and exact command syntax before execution.
- Never invent flags, package names, paths, environment variables, configuration keys, or API behavior.
- Do not install or upgrade dependencies merely to test a guess.

## Command discipline

- Every command must answer a specific question or directly advance the task.
- Prefer one small diagnostic or targeted operation over a broad command.
- Never rerun an unchanged failed command.
- After failure, inspect the complete relevant error and classify it: syntax, path, missing dependency, authentication, permission, network/service, unsupported environment, repository/data, or code/test failure.
- A retry must be materially different and supported by new evidence.
- Permit at most two attempts toward the same command objective. Stop sooner if the same deterministic error appears twice.
- Do not respond to failures by randomly reinstalling tools, clearing caches, changing shells, changing package managers, or trying unrelated command variants.
- Use timeouts for commands that may hang. If progress stops, terminate once, inspect state, and diagnose before continuing.
- Redirect verbose output to a file and inspect targeted matches or the final relevant lines. Do not repeatedly print full logs.

## Git, cloning, and Git LFS

- Before cloning, establish why repository contents are needed.
- Prefer, in order: existing workspace files, local documentation, one raw file or release artifact, an archive download, sparse/shallow checkout, then a full clone only when necessary.
- Do not clone an entire repository merely to read documentation or one source file.
- Before Git operations, inspect repository status and preserve user changes. Never use destructive reset, clean, checkout-over, or rebase as generic troubleshooting.
- For Git LFS, first inspect `.gitattributes`, confirm whether Git LFS is installed, check endpoint/authentication, and determine whether the actual LFS objects are required.
- Never repeat `git clone`, `git pull`, or `git lfs pull` without new evidence.
- If LFS objects are unnecessary, use a documented skip-smudge or selective-download path. If required objects are inaccessible, report the blocker instead of looping.

## Testing and expensive work

- Use this verification ladder: static/syntax check -> import or focused unit test -> one real-input smoke test -> relevant test subset -> full suite only when justified.
- Do not rerun an unchanged broad test after a deterministic failure; run the smallest test that distinguishes the current hypothesis.
- Before remote, paid, GPU, or full-scale experiments, validate one real batch, shapes, dtypes, devices, data paths, loss/metrics, and checkpoint behavior locally or at minimal scale.
- Estimate runtime and memory before launching expensive work. Check early for actual GPU use and CPU/data-loader bottlenecks.
- Do not launch paid or full-scale compute without explicit authorization.

## Communication and context

- Do not narrate routine reads and commands. Report only material decisions, blockers, irreversible actions, and final evidence.
- Avoid repeating the prompt, plan, prior findings, command output, or unchanged status.
- Keep progress updates and final summaries compact; include exact paths, commands, errors, and test results only when useful.
- Use one session for one coherent objective. When context becomes stale or continuation would require repeated reconstruction, write a concise `HANDOFF.md` containing only goal, decisions, changed files, current failure, verification state, and next action.
- Do not spawn subagents, parallel attempts, or duplicate reviewers unless explicitly requested or clearly necessary for independent work. More agents are not a token-saving strategy.

## Completion

- Do not claim success without fresh evidence tied to the original requirement.
- If verification is unavailable, state exactly what remains unverified.
- When blocked, stop and report the minimum user action or external dependency needed to continue; do not consume turns exploring unsupported workarounds.
