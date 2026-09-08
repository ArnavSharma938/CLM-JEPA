# Actual second-640 frozen-checkpoint confirmation

This is the second-640 evaluation of the existing epoch-4 Native and decoder-projected checkpoints for seeds 2027/3163. No retraining was performed.

**The earlier Report-05 first-640 result is not the confirmation result.** Its attribution to the second 640 was incorrect. Only the newly verified second-640 results below enter the frozen decision rule.

## Panel and endpoint

`data/clm_jepa_uspto_mit_stp_confirmation/untouched_second_640.jsonl` is the byte-preserving `rows[640:1280]` slice of `untouched_1280.jsonl`.

SHA-256: `f454d1f12e8035dc63c3992ce59a21809ecf3745b049dc910aec4538ed3c36f8`. Verified 640 unique reactions, zero first-640 overlap, exact identity order and original panel indices 640–1279. All four prediction files match the panel identities, official group indices, source views, targets, and ordering.

The unchanged official evaluator uses five views, beam 10, ten candidates/view, canonical chemical identity, and reciprocal-rank aggregation. Aggregate exact top-1 is primary. The original optional exact execution paths and model configuration were preserved.

## Results

| Seed | Arm | Top-1 | Top-3 | Top-5 | Top-10 | View validity | Ranked validity |
|---|---|---|---|---|---|---|---|
| 2027 | native | 23/640 (3.59375%) | 100/640 (15.62500%) | 153/640 (23.90625%) | 204/640 (31.87500%) | 98.284375% | 100.000000% |
| 2027 | decoder_projected | 17/640 (2.65625%) | 97/640 (15.15625%) | 147/640 (22.96875%) | 204/640 (31.87500%) | 97.990625% | 100.000000% |
| 3163 | native | 21/640 (3.28125%) | 115/640 (17.96875%) | 159/640 (24.84375%) | 223/640 (34.84375%) | 98.487500% | 99.968750% |
| 3163 | decoder_projected | 21/640 (3.28125%) | 102/640 (15.93750%) | 155/640 (24.21875%) | 210/640 (32.81250%) | 97.909375% | 100.000000% |

| Seed | Effect (pp) | Projected-only | Native-only | Ties | Exact McNemar p | Paired bootstrap 95% CI (pp) |
|---|---|---|---|---|---|---|
| 2027 | -0.93750 | 3 | 9 | 628 | 0.14599609 | [-2.03125, 0.15625] |
| 3163 | 0.00000 | 6 | 6 | 628 | 1 | [-1.09375, 1.09375] |

Mean top-1 effect: **-0.468750 pp**. Pooled discordances (descriptive): 9 projected-only / 15 Native-only.

Crossed seed/reaction bootstrap 95% CI: **[-1.718750, 0.781250] pp**. Exact reaction-cluster sign-flip two-sided p: **0.32863379**.

Bootstrap intervals use 20,000 percentile resamples. Per-seed reaction bootstrap RNG seeds are 2027 and 3163. Crossed bootstrap RNG seed is 20273163: sample two seed indices and 640 reaction indices independently with replacement; apply the same sampled reaction indices across both sampled seeds. Sign flips use one common sign per reaction across seeds and exact integer convolution, comparing absolute summed differences. Pooled discordances are not treated as independent observations.

## Frozen decision

**STOP**

Rule: both effects >0 → PROCEED; both effects ≤0 → STOP; otherwise → SEED_4211_REQUIRED. No expansion seeds were launched.

## Provenance

All cells passed checkpoint and manifest hash verification before interpretation. Full file hashes are frozen in `runs/decoder_projected/confirmation_second640/frozen_inputs.json`; exact launch commands are saved in each evaluation directory. The following adapter hashes identify the active saved LoRA, embedding, and LM-head weights; the base-model/config/tokenizer hashes are also recorded in the frozen input manifest.

| Seed | Arm | Adapter SHA-256 | Predictions SHA-256 |
|---|---|---|---|
| 2027 | native | `efc31efcfe07d6066553d2a5b87a615efabb8ea2d28b192cf03f9df2108dff0c` | `7c6342162742c3306b80b9941e5ad92cd98acbbbe8b4f4e68a7d6bc0c0822e01` |
| 2027 | decoder_projected | `dbf5663bbba412daaba7e2ad5ed59683c99602dd9501a91c881a61249042ea01` | `7f04af70c07101679520be7d4451dbd3c736b15a84cf0086bc8350bf1a86cd70` |
| 3163 | native | `7c9247520d897e36f1b065b37f981b254b0233f3a73d279d184ac43d880e1f54` | `0154b4850ff84befe468eef3a8425a3f05c89cbebebbb2f9f1dadf3fc48d836b` |
| 3163 | decoder_projected | `7c47f4195e32af2149afa76b48c14bb5b2f675ffec2c11604746af3ff0c7b6dd` | `949eb825527eda85edb29737c4bd064e1e7514ad4c8c50d209e42c3aec4b3b65` |

Exact remote checkpoint paths and evaluator manifest paths:

- 2027 native: `/home/ubuntu/CLM-JEPA/runs/decoder_projected/confirmation/seed_2027/native/training/checkpoints/epoch_4`; manifest `/home/ubuntu/CLM-JEPA/data/clm_jepa_uspto_mit_stp_confirmation/untouched_second_640.jsonl`; 640 predictions.
- 2027 decoder_projected: `/home/ubuntu/CLM-JEPA/runs/decoder_projected/confirmation/seed_2027/decoder_projected/training/checkpoints/epoch_4`; manifest `/home/ubuntu/CLM-JEPA/data/clm_jepa_uspto_mit_stp_confirmation/untouched_second_640.jsonl`; 640 predictions.
- 3163 native: `/home/ubuntu/CLM-JEPA/runs/decoder_projected/confirmation/seed_3163/native/training/checkpoints/epoch_4`; manifest `/home/ubuntu/CLM-JEPA/data/clm_jepa_uspto_mit_stp_confirmation/untouched_second_640.jsonl`; 640 predictions.
- 3163 decoder_projected: `/home/ubuntu/CLM-JEPA/runs/decoder_projected/confirmation/seed_3163/decoder_projected/training/checkpoints/epoch_4`; manifest `/home/ubuntu/CLM-JEPA/data/clm_jepa_uspto_mit_stp_confirmation/untouched_second_640.jsonl`; 640 predictions.

Machine-readable endpoints: `runs/decoder_projected/confirmation_second640/confirmation_results.json`.

Scope ends at this confirmation. No NextLat, optimization, representation analysis, retraining, or expansion-seed work was performed.
