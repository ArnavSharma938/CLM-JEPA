# CLM-JEPA

ChemFM-1B pilot implementation for the gated cLM-JEPA study specified in
`CLM_JEPA_Plan.md`. The repository intentionally contains source code, tests,
immutable reduced-data manifests, the official reaction tokenizer, and compact
gate evidence only. Reproducible environments, upstream repository mirrors,
papers, model checkpoints, and full-scale datasets are not versioned.

## Thunder Compute: Gate 3

Use an Ubuntu prototype with one A100 80 GB, 8 vCPUs, and 200 GB storage:

```bash
git clone https://github.com/ArnavSharma938/CLM-JEPA.git
cd CLM-JEPA
bash scripts/run_thunder.sh
```

The bootstrap downloads the pinned `ChemFM/ChemFM-1B` revision and verifies
the checkpoint SHA-256 before running the unchanged frozen relationship assay.
Gate output is written to `artifacts/gate3/relationship_assay.json`; the full
console log is written to `artifacts/gate3/thunder_run.log`.

## Multi-dataset Gate 3

The controlled assay uses 1,024 deterministic unique-target examples per
dataset, seed 533, batch size 16, and `k in {-1,0,1,2,3}`. Here `k=-1` means
the source's second-to-last active token; every target remains its final EOT or
EOS token. Full datasets stay under `data/official/` and are ignored by Git.

```bash
python scripts/prepare_gate3_datasets.py
python scripts/run_gate3.py --manifest data/manifests/gate3_multi/uspto_mit_synthesis.csv --dataset uspto_mit_synthesis
python scripts/run_gate3_llama.py
python scripts/summarize_gate3.py
```

Compact results are in `artifacts/gate3/multi/`. The cross-chemistry selection
retains `k=0` and `k=1`; the NL-RX-SYNTH Llama assay retains only `k=0`.
Dataset provenance, released-split overlap checks, validity boundaries, and the
20-second GPU trace are preserved under `artifacts/data/` and `artifacts/gpu/`.

## Tests

```bash
python -m pytest -q
```

Gate 1 manifest regeneration requires the official USPTO-MIT-Synthesis
`train.csv` at `data/official/uspto_mit_synthesis/train.csv`. The checked-in
32- and 128-row immutable manifests are sufficient for Gates 2 and 3.

## Experiment tracking

Fine-tuning uses `clm_jepa.tracking.WandbTracker`. Set credentials only in the
process environment:

```bash
export WANDB_API_KEY=...
export WANDB_PROJECT=clm-jepa
export WANDB_ENTITY=arnavsharma-0914-cortexpd-labs
python scripts/smoke_wandb.py
```

The tracker logs the Section 13 loss, task, validity, gradient, learning-rate,
JEPA activity, token, timing, throughput, VRAM, seed, fraction, and resolved
configuration fields. It intentionally does not watch the model or log
parameter histograms and embedding dumps.
