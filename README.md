# CLM-JEPA

ChemFM-1B pilot implementation for the gated cLM-JEPA study specified in
`CLM_JEPA_Plan.md`. The repository intentionally contains source code, tests,
immutable reduced-data manifests, the official reaction tokenizer, and compact
gate evidence only. Reproducible environments, upstream repository mirrors,
papers, the 3.7 GB model checkpoint, and the 236 MB full training CSV are not
versioned.

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

## Tests

```bash
python -m pytest -q
```

Gate 1 manifest regeneration requires the official USPTO-MIT-Synthesis
`train.csv` at `data/official/uspto_mit_synthesis/train.csv`. The checked-in
32- and 128-row immutable manifests are sufficient for Gates 2 and 3.
