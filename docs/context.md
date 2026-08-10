# cLM-JEPA memoryless agent context

This file hands the repository to an agent with no conversation memory. Work
only inside `C:\Users\arnav.DHEERAJACER\CLM-JEPA`. Read `AGENTS.md` and then
`CLM_JEPA_Plan.md` completely before changing experiments. The plan is the
source of truth; this file records current state and user expectations.

## Non-negotiable operating rules

1. Execute gates in numerical order and stop at every permission or stop-work
   condition. A smoke test is not a later-gate pass.
2. Use official LLM-JEPA `references/llm-jepa/finetune.py` as the controlling
   JEPA reference and compare tensor operations line by line. Preserve the
   official ChemFM tokenizer, task serialization, LoRA, generation, and
   evaluation behavior wherever applicable.
3. Do not make arbitrary speed or memory changes. Diagnose first; if a fair
   run does not fit locally after plan-authorized measures, request Thunder
   Compute. The currently authorized continuation shape is one RTX A6000,
   6 vCPUs, 200 GB primary disk, no template, using the current v2 CLI's
   mode-less prototyping path.
4. W&B belongs in the actual fine-tuning loop. Never store API keys, Hugging
   Face tokens, passwords, or other credentials in files, commands, logs, or
   commits. Treat credentials pasted in conversation as exposed and rotate
   them.
5. Preserve tests and reusable experiment code. Keep compact gate decisions
   and evidence; remove reproducible logs, dumps, caches, and completed
   one-off smoke runners.
6. `CLM_JEPA_Plan.md` has user modifications. Do not overwrite or stage it
   unless the user explicitly requests that exact change.

## Provenance and environment

- Public remote: `https://github.com/ArnavSharma938/CLM-JEPA.git`, branch
  `main`; last observed committed revision before this cleanup: `b373bb1`.
- LLM-JEPA reference revision:
  `ea0017c654ad917066ff32afc88276bea8ca5f7e`.
- ChemFM reference revision:
  `ee35b23d03de1a8e97b8e04dcdfb1d579de70f02`.
- ChemFM-1B model revision:
  `f99dc2e89726539bb9cf31b2e2b4360650bac6a8`.
- Local `.venv` uses Python 3.10.20. The ignored model checkpoint is about
  3.7 GB; `.venv` is about 5.5 GB. Neither belongs in Git.
- Local GPU is an RTX 4050 Laptop GPU with 6,141 MiB. Frozen inference fitting
  locally does not establish that LoRA fine-tuning fits.
- Reference repositories are vendored as ordinary source files; their nested
  `.git` metadata was removed so the audited code is visible to Git and users.

## Current repository layout

- `src/chemfm.py`: ChemFM tokenizer, three task directions, collation, LoRA
  loading, generation, and canonicalization.
- `src/jepa.py`: predictor-token setup, native and auxiliary rows, state
  selection, shuffling, loss dropout, and cosine JEPA loss.
- `src/metrics.py`: candidate scoring, stored-prediction scoring, effective
  rank, retrieval, ridge, and relationship diagnostics. Former small
  evaluation/scoring/assay modules were consolidated here.
- `src/train.py`: real Gate 4/5 fine-tuning path and integrated W&B tracker.
- `src/experiments.py`: Gate 4/5 launcher; `src/download_model.py`: explicit
  checkpoint bootstrap.
- `gates/gate1/results/`, `gates/gate2/results/`: compact historical evidence.
- `gates/gate3/`: preparation, frozen assay, summary, report, and one
  consolidated machine-readable result.
- `data/gate3/`: seven versioned 1,024-row samples. Full datasets are stored
  under ignored, clearly named dataset directories in `data/`.
- `references/`: pinned official source. `tests/`: behavioral and exact parity
  verification. Most `runs/` output is ignored; the selected native and
  cLM-JEPA seed-533 epoch-3 checkpoints and the frozen decoder-coupling
  continuation artifacts are intentionally tracked with Git LFS where needed.

The removed `assets/chemfm_reaction_tokenizer` was only 142 KB, not a hidden
large dependency: all three files were byte-identical to the tokenizer under
`references/chemfm/finetuning/reaction_prediction/tokenizer`. The old
`artifacts/data`, `artifacts/gpu`, prediction dumps, console logs, JUnit XML,
failed-FP16 output, and standalone W&B/Thunder helpers were reproducible or
redundant and were removed. Gate 3's former summary, aggregate CSV, and eight
dataset JSONs were consolidated into `gates/gate3/results.json`.

## Dataset state and leakage

The Gate 3 samples cover three forward datasets (`uspto_mit_synthesis`,
`orderly_forward`, `non_uspto_forward`), MetaTrans
metabolism, and three retrosynthesis datasets (`uspto_50k_retro`,
`uspto_480k_template_heldout`, `non_uspto_retro`). Each has 1,024 deterministic
unique-target examples after exact source-target component-overlap
removal.

Full local data is normalized to one CSV per real split under clearly named
dataset directories. Canonical columns are `example_id`, `split`, `source`,
and `target`; old OpenNMT parallel text, Parquet, raw-release, and duplicate
CSV copies were removed after count/hash verification. Exact paths and counts
are in Plan Section 3.4 and `data/dataset_manifest.json`. The MetaTrans
251-row released validation is reference data, not the required derived
parent-grouped scaffold validation; that scaffold split remains a pre–Gate 4
requirement.

Recorded released-split overlaps were: ORDerly forward 7 reaction pairs;
MetaTrans train/validation 3 parents; USPTO-50K 16;
USPTO-480K 0. These were reported, not silently removed. Gate 3 used the
declared full-dataset source for each frozen diagnostic, including external
test-only sets where no training split exists. NL-RX-SYNTH/Llama code, data,
and results were removed at the user's
explicit request and must not be restored without a new request.

## Gate status

### Gates 0 and 1 — passed historically

Gate 1 overfit results: 32 examples reached 32/32 valid and exact; 128 examples
reached 128/128 valid and 121/128 exact. Evidence is under
`gates/gate1/results/`. Obsolete Gate 1 runner code is intentionally gone.

### Gate 2 — JEPA core passed

Compact real-model evidence is under `gates/gate2/results/`. Tests cover exact
lambda-zero native equivalence, one model call, causal and row isolation,
source sensitivity, target EOS selection, shared-backbone gradients,
monitor-only gradients, and reproducible unequal-target shuffling. The golden
reference test independently reconstructs the pinned `finetune.py` default
path and compares states, losses, and every parameter gradient exactly.

### Gate 3 — passed after correction

Pinned ChemFM-1B, no fine-tuning, seed 533, batch size 16, seven retained
datasets and 7,168 examples. Tested `k in {-1,0,1,2,3}`; project `k=-1` means the source's
second-to-last active token. Every position retained pair-specific signal on
all datasets. The prespecified equal-weight cross-dataset rank selected `k=0`
and `k=1`. Canonical evidence is `gates/gate3/README.md` and
`gates/gate3/results.json`.

The original retrospective assays incorrectly used forward markers. In
addition, USPTO-50K had been sampled before direction correction, invalidating
target-keyed deterministic selection. All three retrosynthesis assays were
rerun correctly; USPTO-50K was regenerated from the complete verified
160,012-row file and rerun. No other Gate 3 correction was required.

### Gate 4 — passed for the reduced USPTO-MIT pilot

The backend fidelity gaps were corrected and the user replaced broad HPO with
a fixed reliable configuration: learning rate 1e-4, k=1, effective JEPA weight
1.0, dropout 0.5, seeds 533 and 917. The selected seed-533 checkpoint is epoch
3. Canonical evidence is `gates/gate4/results.json`.

### Gate 5 — failed on the frozen primary rule

The selected seed-533 native and cLM-JEPA endpoints tied at exact top-1,
2/32 = 0.0625. Because the plan required cLM-JEPA to strictly beat native, the
full control matrix was stopped. Do not claim a generative improvement. The
selected checkpoints are:

- `runs/gate5/checkpoints/native-s533/epoch_3`
- `runs/gate4_v2/reliable/clm_jepa-s533-checkpoints/epoch_3`

The follow-up 1,024-identity representation assay found extreme
common-direction concentration induced by cLM-JEPA but strong pair-specific
chemistry in the centered/top-PC-removed residual. This is not dimensional or
constant-vector collapse. See `docs/USPTO_MIT_GEOMETRY_DIAGNOSIS.md`.

The active frozen-checkpoint decoder-coupling diagnostic uses 1,024 unique
canonical validation identities. Both endpoint CE/representation/intervention
artifacts are complete. Aggregate target-token CE is 0.2338166 for native and
0.2355382 for cLM-JEPA, so the earlier small-sample CE advantage reverses to a
0.74% disadvantage. Native beam-10 generation is resumable from 36/1,024 rows.
See `runs/diagnostics/decoder_coupling/README.md`. Do not retrain, tune, change
the objective, or proceed to MetaTrans/retrosynthesis before this analysis is
complete.
## LLM-JEPA fidelity claim

The default non-additive cosine JEPA core in `src/jepa.py` is functionally
equivalent to the corresponding pinned `finetune.py` path after required
ChemFM serialization and EOS adaptations. The files are not textually
identical. Optional upstream additive-mask, L2, MSE, and InfoNCE branches are
not ported. This claim rests on a whole-file responsibility audit, an
operation-level mapping, exact executable tensor/gradient comparison, causal
invariants, and real frozen ChemFM-1B execution—not unit tests alone. Read
`docs/LLM_JEPA_FIDELITY.md` for exact scope and exclusions.

## Current continuation

Local generation was stopped because two protocol-faithful RTX 4050 beam-10
passes projected to roughly 12 hours. Continue on the authorized Thunder RTX
A6000 instance, install dependencies with `uv`, verify the pinned ChemFM model,
resume native generation, run cLM-JEPA generation, and monitor `nvidia-smi`.
Keep `--generation-batch-size 1`: a batch-size-2 smoke test changed lower-beam
ordering and is excluded from evidence.

After both passes, run the frozen summarizer, update
`docs/USPTO_MIT_GEOMETRY_DIAGNOSIS.md` with paired/correlation uncertainty,
and report only the selected native and fully correct cLM-JEPA endpoints. Per
the user's latest instruction, omit the “both correct” and “neither correct”
outcome categories; retain native-only and cLM-JEPA-only differences and all
prespecified aggregate top-k metrics. Exact top-1 remains primary.
Useful checks:

```powershell
Set-Location C:\Users\arnav.DHEERAJACER\CLM-JEPA
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe gates\gate3\summarize.py
git status --short
git diff --check
```
