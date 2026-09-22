# ESM-2-650M BackboneRef Gate-1 runbook

This branch is an encoder-only MLM screen and does not modify the frozen ESM-IF1 plan. It reuses the earlier study's architecture-derived target checks, paired randomness, validation-selected checkpoints, resumability, evidence-seed separation, structured ledgers, and stop logic.

## Pinned sources

- Dayhoff dataset: `microsoft/Dayhoff@7f6771db1bfad58011fcadf72f86eacbef39a482`
- ESM-2 model: `facebook/esm2_t33_650M_UR50D@08e4846e537177426273712802403f7ba8261b6c`
- Backbone structure ZIP SHA-256: `5c75396fe6e0229f4e4a8dbeab7b88b95cd8e89c47137cf83f9e33d3fb40270a`

The released `backboneref_structures` Arrow configuration has 240,811 rows but no accession column. It must not be joined to sequences by implicit row position. The official structure ZIP retains the accession as each PDB stem; `distances.py` verifies its hash and complete BRn∩BRq accession coverage before extracting selected structures.

## Local correctness check

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest protein\tests -q
```

No model weights or full Dayhoff shards are required for these tests.

## A6000 execution order

Use one A6000, 6 vCPU, 100 GB, base template. Create it only after the local suite passes. On the instance, create a clean environment, install the pinned Python requirements, install `flash-attn==2.8.3.post1` with `--no-build-isolation` after PyTorch is available, and install current Foldseek and MMseqs2 binaries. Record their exact versions.

1. Materialize deterministic manifests:

   ```bash
   python -m protein.esm2_gate1.prepare_data --output-dir protein/data/esm2_gate1
   ```

2. Download the pinned official structure archive, then build distances and evaluation sets:

   ```bash
   python -m protein.esm2_gate1.distances \
     --manifest-root protein/data/esm2_gate1 \
     --structure-zip /path/to/backbone_export.zip
   ```

3. Run metadata, original-checkpoint parity, and attention-backend benchmarks:

   ```bash
   mkdir -p protein/runs/esm2_gate1
   python -m protein.esm2_gate1.preflight metadata --output protein/runs/esm2_gate1/preflight.json
   python -m protein.esm2_gate1.preflight parity --output protein/runs/esm2_gate1/fair_esm_parity.json
   python -m protein.esm2_gate1.preflight benchmark --output protein/runs/esm2_gate1/backend_benchmark.json
   ```

   An unexplained Fair-ESM mismatch is a hard stop. Select only a backend whose forward NLL and backward gradient checks pass.

4. Profile and optimize the complete training and evaluation paths on the actual manifests.
   This compares token budgets, worker counts, gradient checkpointing on/off, and (only after
   the stable eager/SDPA path passes) `torch.compile`, using FT as the worst-case
   training-memory condition. Measurements include collation, host-to-device transfer,
   forward, backward, clipping, and fused AdamW. Evaluation is profiled separately. Compiled
   execution must improve end-to-end throughput by at least 5%; otherwise it is rejected.
   The selected evaluation path reuses one loaded model/checkpoint across all three held-out
   sets, and identical best/last checkpoint states use hard links to avoid duplicate I/O:

   ```bash
   python -m protein.esm2_gate1.profile_execution \
     --manifest protein/data/esm2_gate1/train_10k.jsonl \
     --output protein/runs/esm2_gate1/execution_profile.json \
     --attention-backend sdpa
   ```

5. Execute the conditional screen with the selected backend and profiled token/worker values:

   ```bash
   python -m protein.esm2_gate1.run_screen \
     --manifest-root protein/data/esm2_gate1 \
     --output-root protein/runs/esm2_gate1 \
     --attention-backend sdpa \
     --microbatch-tokens 4096 \
     --num-workers 4
   ```

   The driver runs seed-7 short LR probes, seed-11 10k evidence, conditionally skips 50k FT when FT≈DTFT, runs seed-11 50k DTFT/LoRA, and launches seed 23 only for load/rank pairs crossing the registered 10% gap threshold. It never launches H1-H4.

6. Render the report:

   ```bash
   python -m protein.esm2_gate1.report \
     --manifest-root protein/data/esm2_gate1 \
     --run-root protein/runs/esm2_gate1 \
     --output protein/docs/reports/ESM2_650M_BACKBONEREF_GATE1.md
   ```

All training paths use deterministic exposure-specific corruptions, ordinary independent protein sequences, length-bucketed token-budget batches, a common effective non-padding residue target per optimizer step, BF16, and fixed-mask validation. `last.pt` contains optimizer, trainable, gradient, RNG, epoch, and data-position state for exact continuation, including a partially accumulated optimizer step.

For Thunder, `run_thunder_queue.sh` wraps all data, parity, backend, distance,
end-to-end profiling, LR, evidence, conditional replication, and report stages in one
fail-fast/resumable queue. Launch it in a detached `screen` session and use the explicit
`queue.exit` file as the terminal status; SSH disconnection must not own the process lifetime.
