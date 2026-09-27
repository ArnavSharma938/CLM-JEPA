# ESM-2-650M BackboneRef Gate-1 results

## Provenance and corpus

- Dayhoff revision: `7f6771db1bfad58011fcadf72f86eacbef39a482`
- ESM-2 revision: `08e4846e537177426273712802403f7ba8261b6c`
- ESM-2 weight SHA-256: `a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0`
- Repository commit / Gate-1 source hash: `d09b5fb28cae23363d7cc1527194bd55bba80c3a` / `c076a70023fe07fec5aaa4760d85f80d42fe9e391b8bbb1bee18f7e375b4ced3`
- BRn / BRq / intersection backbones: 138,044 / 127,633 / 54,324
- Malformed / invalid-length / exact-duplicate / low-complexity removals: 0 / 0 / 0 / 354
- Low-complexity removals from the preferred BRn intersection: 58
- BRn-only fallback backbones: 0
- Structure ZIP SHA-256: `5c75396fe6e0229f4e4a8dbeab7b88b95cd8e89c47137cf83f9e33d3fb40270a`
- Structure accessions / missing intersection accessions: 240,811 / 0
- Accession grouping check (BRq rows/accession min-max): 60-79; BRn rows/accession distribution: `{"maximum": 73.0, "mean": 72.44067109037698, "median": 73.0, "minimum": 67.0, "q01": 70.0, "q25": 72.0, "q75": 73.0, "q99": 73.0}`
- Foldseek / MMseqs versions: `463739e0014a1549a527de589102cde98f802f37` / `d401e78c2d18a822cdb1527d7464a043f6035a15`

The representative sequence is the minimum stable SHA-256 over accession, generation description, and sequence. Backbone accession, not sequence row, is the independent unit.

| Manifest | Backbones | Sequences | Residues | SHA-256 |
|---|---:|---:|---:|---|
| heldout_candidates | 4,266 | 4,266 | 900,539 | `973be19dd232cb58e7ce174ac103ab9c5f89d1d28ed16fbac277b6fb4c7818fb` |
| natural_test | 2,000 | 2,000 | 420,689 | `c4c3deaec95ae29f01838ca04a07dc60af9d6205e8a4ae1bb2ea34d8eb036b01` |
| train_10k | 10,000 | 10,000 | 2,107,974 | `9e7012dd4f3cd757f8fe224c7755f891b11e03462831410042ab3f03002d790d` |
| train_50k | 50,000 | 50,000 | 10,542,729 | `9b2659623a2498ff93c5308d1c082381e3f1f5fbc90f42c03cf53e40d849d3b0` |
| remote_ood_test | 111 | 111 | 34,212 | `cad76067d55777fd35b5215016792d9a1907e7ba1537d83ab254104915db1d00` |
| standard_test | 2,000 | 2,000 | 411,224 | `1d5b73f6d7ef8cab9446ebb52e1a0f192a53c90935afaf0f919cb7e8ac14e531` |
| validation | 555 | 555 | 118,837 | `a28bc79104272db0c276db45b1608082906f1927774818dd9702ab33c472d772` |

Length and per-sequence entropy summaries (minimum / median / mean / maximum):

| Manifest | Length | Entropy (bits) |
|---|---|---|
| heldout_candidates | 53 / 201 / 211.1 / 509 | 2.641 / 3.588 / 3.548 / 3.985 |
| natural_test | 63 / 200 / 210.3 / 491 | 0.933 / 4.017 / 3.981 / 4.242 |
| train_10k | 51 / 200 / 210.8 / 512 | 2.474 / 3.587 / 3.544 / 3.932 |
| train_50k | 51 / 200 / 210.9 / 512 | 2.418 / 3.587 / 3.545 / 3.987 |
| remote_ood_test | 123 / 314 / 308.2 / 472 | 3.174 / 3.673 / 3.652 / 3.893 |
| standard_test | 53 / 196 / 205.6 / 504 | 2.641 / 3.585 / 3.542 / 3.925 |
| validation | 76 / 206 / 214.1 / 502 | 2.750 / 3.577 / 3.540 / 3.889 |

The 50k amino-acid frequencies are: A=0.1180, C=0.0051, D=0.0380, E=0.1630, F=0.0204, G=0.0352, H=0.0075, I=0.0694, K=0.0899, L=0.1390, M=0.0110, N=0.0188, P=0.0270, Q=0.0096, R=0.0557, S=0.0340, T=0.0505, V=0.0901, W=0.0031, Y=0.0147.
The preregistered natural-reference low-complexity rule and thresholds were: `{"entropy_bits_min": 2.407114448891748, "maximum_homopolymer_fraction_max": 0.10537199795605505, "maximum_residue_fraction_max": 0.4769859943977581, "rule": "exclude if entropy<q0.001 OR max-residue>q0.999 OR homopolymer>q0.999"}`.

### Natural-reference composition and complexity comparison

| Population | Sequences | Entropy median | Max-residue q99 | Homopolymer q99 | Unique-residue median |
|---|---:|---:|---:|---:|---:|
| natural_length_matched_uniref50 | 25,930 | 4.011 | 0.253 | 0.044 | 1.000 |
| synthetic_after_low_complexity_filter | 54,266 | 3.587 | 0.336 | 0.049 | 0.900 |
| synthetic_before_low_complexity_filter | 54,324 | 3.587 | 0.339 | 0.051 | 0.900 |

Natural versus prefilter-synthetic amino-acid frequencies: A=0.0878/0.1181, C=0.0146/0.0051, D=0.0534/0.0380, E=0.0612/0.1629, F=0.0399/0.0204, G=0.0695/0.0353, H=0.0224/0.0075, I=0.0564/0.0693, K=0.0508/0.0898, L=0.0977/0.1391, M=0.0229/0.0111, N=0.0386/0.0187, P=0.0503/0.0270, Q=0.0367/0.0096, R=0.0620/0.0557, S=0.0709/0.0340, T=0.0553/0.0505, V=0.0665/0.0901, W=0.0137/0.0031, Y=0.0294/0.0147 (natural/synthetic). No composition matching was applied.

## Model and execution validation

- Architecture: 33 layers, hidden size 1280, 20 heads, FFN size 5120
- Runtime versions: Transformers `5.17.0`, PyTorch `2.12.1+cu130`
- Loaded parameters: 651,043,254
- Dense target matrices/scalars: 198 / 648,806,400
- FT / DTFT / LoRA-r8 / LoRA-r64 trainables: 651,043,254 / 648,806,400 / 6,082,560 / 48,660,480
- DTFT/FT scalar coverage: 0.996564
- Fair-ESM parity max-logit/NLL differences: 0 / 0
- Selected attention backend after end-to-end profiling: `sdpa`
- torch.compile selected for training: `False`

| Dense target family (per layer) | Weight shape | Matrices |
|---|---|---:|
| attention.output.dense | 1280 x 1280 | 33 |
| attention.self.key | 1280 x 1280 | 33 |
| attention.self.query | 1280 x 1280 | 33 |
| attention.self.value | 1280 x 1280 | 33 |
| intermediate.dense | 5120 x 1280 | 33 |
| output.dense | 1280 x 5120 | 33 |

| Backend | Passed | tokens/s | Peak VRAM (GB) |
|---|:---:|---:|---:|
| eager | yes | 1087.7 | 2.63 |
| sdpa | yes | 1101.6 | 2.63 |
| flash_attention_2 | no | nan | 0.00 |

### End-to-end code optimization

The full data-loader → transfer → model → loss path was profiled on the A6000. Training additionally included backward, gradient clipping, and fused AdamW; FT was used as the worst-memory condition. All candidate measurements are preserved in `execution_profile.json`.

| Path | Token budget | Workers | Gradient checkpointing | torch.compile | Residues/s | Peak VRAM (GB) |
|---|---:|---:|:---:|:---:|---:|---:|
| training | 12,288 | 2 | False | False | 10137.9 | 39.13 |
| evaluation | 16,384 | 2 | n/a | False | 32054.7 | 3.70 |

Evaluation reuses one loaded model/checkpoint across standard, structural-OOD, and natural sets; identical best/last checkpoint states use hard links to avoid duplicate serialization and storage.

Training comparison maxima (residues/s; peak VRAM belongs to the maximizing row):

| Factor | Setting | Best residues/s | Peak VRAM (GB) |
|---|---|---:|---:|
| attention backend | eager | 8557.4 | 37.59 |
| attention backend | sdpa | 12638.4 | 28.28 |
| gradient checkpointing | False | 12638.4 | 28.28 |
| gradient checkpointing | True | 7886.9 | 11.57 |
| train token budget | 12288 | 12638.4 | 28.28 |
| train token budget | 16384 | 9807.9 | 49.12 |
| train token budget | 2048 | 5696.1 | 14.20 |
| train token budget | 4096 | 8618.5 | 19.17 |
| train token budget | 8192 | 10060.7 | 29.15 |
| workers | 2 | 12638.4 | 28.28 |
| workers | 4 | 10037.6 | 29.15 |
| torch.compile | False | 10137.9 | 39.13 |
| torch.compile | True | 12638.4 | 28.28 |

Evaluation token-budget/worker/compile candidates are retained in `execution_profile.json`; the selected compiled 16,384-token, four-worker path is shown above.

## Learning-rate screen

| Method | LR | Minimum validation NLL | Terminal validation NLL | Selected |
|---|---:|---:|---:|:---:|
| dtft | 1e-06 | 1.985376 | 1.985376 | no |
| dtft | 3e-06 | 1.934979 | 1.934979 | no |
| dtft | 1e-05 | 1.868865 | 1.868865 | yes |
| ft | 1e-06 | 1.985376 | 1.985376 | no |
| ft | 3e-06 | 1.934979 | 1.934979 | no |
| ft | 1e-05 | 1.868865 | 1.868865 | yes |
| lora64 | 1e-05 | 2.100148 | 2.100148 | no |
| lora64 | 3e-05 | 2.059113 | 2.059113 | no |
| lora64 | 0.0001 | 1.956905 | 1.956905 | yes |
| lora8 | 1e-05 | 2.100148 | 2.100148 | no |
| lora8 | 3e-05 | 2.059113 | 2.059113 | no |
| lora8 | 0.0001 | 1.956905 | 1.956905 | yes |

Candidate trajectories are retained under `hpo/10k/<method>/selection.json`; the registered NLL tie tolerance is 1e-4 and ties prefer the lower LR.

## Base distribution-pressure comparison

Base masked-token NLL was 2.087370 on standard synthetic, 2.165268 on structural-OOD synthetic, and 1.939585 on length-matched UniRef50.

## Raw seed-11 NLL

| Load | Set | Base | FT | DTFT | LoRA-r8 | LoRA-r64 |
|---|---|---:|---:|---:|---:|---:|
| 10k | standard_test | 2.087370 | 1.763581 | 1.766308 | 1.792171 | 1.748378 |
| 10k | remote_ood_test | 2.165268 | 1.889419 | 1.892068 | 1.909032 | 1.877691 |
| 50k | standard_test | 2.087370 | not run | 1.690044 | 1.710961 | 1.687427 |
| 50k | remote_ood_test | 2.165268 | not run | 1.829727 | 1.847632 | 1.830327 |

## Gate quantities (seed 11)

| Load | Rank | Set | A_DTFT | G_LoRA | F_gap | 95% CI A / G / F | Candidate | Replicate |
|---|---|---|---:|---:|---:|---|:---:|:---:|
| 10k | lora64 | remote_ood_test | 0.273200 | -0.014378 | -0.053 | [0.2567,0.2901] / [-0.0185,-0.0103] / [-0.0685,-0.0373] | no | no |
| 10k | lora64 | standard_test | 0.321061 | -0.017930 | -0.056 | [0.3165,0.3258] / [-0.0192,-0.0166] / [-0.0599,-0.0518] | no | no |
| 10k | lora8 | remote_ood_test | 0.273200 | 0.016963 | 0.062 | [0.2567,0.2901] / [0.0141,0.0199] / [0.0513,0.0734] | no | no |
| 10k | lora8 | standard_test | 0.321061 | 0.025863 | 0.081 | [0.3165,0.3258] / [0.0249,0.0269] / [0.0775,0.0836] | no | yes |
| 50k | lora64 | remote_ood_test | 0.335541 | 0.000600 | 0.002 | [0.3179,0.3535] / [-0.0037,0.0048] / [-0.0109,0.0142] | no | no |
| 50k | lora64 | standard_test | 0.397326 | -0.002617 | -0.007 | [0.3920,0.4027] / [-0.0039,-0.0013] / [-0.0098,-0.0034] | no | no |
| 50k | lora8 | remote_ood_test | 0.335541 | 0.017905 | 0.053 | [0.3179,0.3535] / [0.0143,0.0214] / [0.0427,0.0638] | no | no |
| 50k | lora8 | standard_test | 0.397326 | 0.020918 | 0.053 | [0.3920,0.4027] / [0.0198,0.0220] / [0.0499,0.0554] | no | no |

### Continuous structural-distance analysis

| Load | Rank | Set | Linear slope G(distance) | Spearman rho | p-value |
|---|---|---|---:|---:|---:|
| 10k | lora64 | remote_ood_test | 0.063416 | 0.1016 | 0.289 |
| 10k | lora64 | standard_test | 0.039816 | 0.0887 | 7.06e-05 |
| 10k | lora8 | remote_ood_test | 0.015850 | 0.0075 | 0.937 |
| 10k | lora8 | standard_test | -0.071998 | -0.1864 | 4.22e-17 |
| 50k | lora64 | remote_ood_test | 0.090565 | 0.1307 | 0.172 |
| 50k | lora64 | standard_test | 0.017052 | 0.0433 | 0.0527 |
| 50k | lora8 | remote_ood_test | -0.053691 | -0.0569 | 0.553 |
| 50k | lora8 | standard_test | -0.049663 | -0.1232 | 3.22e-08 |
| 10k | lora64 | standard+OOD | 0.033238 | 0.0879 | 5.22e-05 |
| 10k | lora8 | standard+OOD | -0.066536 | -0.2029 | 4.74e-21 |
| 50k | lora64 | standard+OOD | 0.016172 | 0.0459 | 0.0351 |
| 50k | lora8 | standard+OOD | -0.043979 | -0.1245 | 9.49e-09 |

Distance quintiles and all bootstrap intervals are retained in `gate1_summary.json`; the resampling unit is backbone/protein.
Structural distance is `1 - maximum query-normalized Foldseek TM-score` to the named training load. Sequence proximity is the best-bit MMseqs2 hit with identity and minimum query/target coverage retained.

### Registered screen-pattern flags

Load-dependence deltas (`F_gap(50k) - F_gap(10k)`): lora8/standard_test=-0.028, lora64/standard_test=0.049, lora8/remote_ood_test=-0.009, lora64/remote_ood_test=0.054.
Rank-sensitive candidate gaps (r8 >=10%, r64 <10%): none.
Material r64 candidate gaps: none.

### FT versus DTFT target-coverage check (10k)

| Set | Base NLL | FT NLL | DTFT NLL | (DTFT-FT)/(Base-FT) | Equivalent within 10% |
|---|---:|---:|---:|---:|:---:|
| remote_ood_test | 2.165268 | 1.889419 | 1.892068 | 0.010 | yes |
| standard_test | 2.087370 | 1.763581 | 1.766308 | 0.008 | yes |

### Triggered seed-23 replication

| Load/rank | Set | A_DTFT | G_LoRA | F_gap | 95% CI A / G / F | Candidate gap replicated |
|---|---|---:|---:|---:|---|:---:|
| 10k/lora8 | remote_ood_test | 0.273747 | 0.018584 | 0.068 | [0.2575,0.2906] / [0.0157,0.0216] / [0.0573,0.0793] | no |
| 10k/lora8 | standard_test | 0.321375 | 0.024882 | 0.077 | [0.3168,0.3261] / [0.0239,0.0258] / [0.0745,0.0804] | no |

## Natural-protein retention

| Load | Method | Base natural NLL | Method natural NLL | Delta retention |
|---|---|---:|---:|---:|
| 10k | dtft | 1.939585 | 2.003118 | 0.063533 |
| 10k | ft | 1.939585 | 2.011284 | 0.071699 |
| 10k | lora64 | 1.939585 | 2.143841 | 0.204256 |
| 10k | lora8 | 1.939585 | 2.045785 | 0.106200 |
| 50k | dtft | 1.939585 | 2.086217 | 0.146632 |
| 50k | lora64 | 1.939585 | 2.360188 | 0.420602 |
| 50k | lora8 | 1.939585 | 2.264451 | 0.324866 |

## Training trajectories and resources

| Load | Method | Seed | Status | Best validation NLL | Steps | Sequences | Residues | Effective residues/step | Residues/s | Padding | Runtime (h) | Peak VRAM (GB) | Best checkpoint |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10k | dtft | 11 | improving | 1.788161 | 231 | 40,000 | 8,431,896 | 36407.2 | 7391.7 | 0.074 | 0.32 | 41.83 | `/home/ubuntu/CLM-JEPA/protein/runs/esm2_gate1/evidence/10k/dtft/seed_11/best.pt` |
| 10k | dtft | 23 | improving | 1.787305 | 232 | 40,000 | 8,431,896 | 36294.4 | 6489.8 | 0.074 | 0.36 | 41.82 | `/home/ubuntu/CLM-JEPA/protein/runs/esm2_gate1/evidence/10k/dtft/seed_23/best.pt` |
| 10k | ft | 11 | improving | 1.785334 | 231 | 40,000 | 8,431,896 | 36407.2 | 7543.3 | 0.074 | 0.31 | 41.93 | `/home/ubuntu/CLM-JEPA/protein/runs/esm2_gate1/evidence/10k/ft/seed_11/best.pt` |
| 10k | lora64 | 11 | improving | 1.768296 | 231 | 40,000 | 8,431,896 | 36407.2 | 8736.4 | 0.074 | 0.27 | 35.22 | `/home/ubuntu/CLM-JEPA/protein/runs/esm2_gate1/evidence/10k/lora64/seed_11/best.pt` |
| 10k | lora8 | 11 | improving | 1.810811 | 231 | 40,000 | 8,431,896 | 36407.2 | 8943.0 | 0.074 | 0.26 | 34.14 | `/home/ubuntu/CLM-JEPA/protein/runs/esm2_gate1/evidence/10k/lora8/seed_11/best.pt` |
| 10k | lora8 | 23 | improving | 1.809332 | 232 | 40,000 | 8,431,896 | 36294.4 | 8961.1 | 0.074 | 0.26 | 34.13 | `/home/ubuntu/CLM-JEPA/protein/runs/esm2_gate1/evidence/10k/lora8/seed_23/best.pt` |
| 50k | dtft | 11 | improving | 1.711576 | 1,176 | 200,000 | 42,170,916 | 35849.8 | 8229.7 | 0.072 | 1.43 | 41.84 | `/home/ubuntu/CLM-JEPA/protein/runs/esm2_gate1/evidence/50k/dtft/seed_11/best.pt` |
| 50k | lora64 | 11 | improving | 1.714121 | 1,176 | 200,000 | 42,170,916 | 35849.8 | 8811.7 | 0.072 | 1.34 | 35.27 | `/home/ubuntu/CLM-JEPA/protein/runs/esm2_gate1/evidence/50k/lora64/seed_11/best.pt` |
| 50k | lora8 | 11 | improving | 1.732140 | 1,176 | 200,000 | 42,170,916 | 35849.8 | 8957.2 | 0.072 | 1.32 | 34.14 | `/home/ubuntu/CLM-JEPA/protein/runs/esm2_gate1/evidence/50k/lora8/seed_11/best.pt` |

Best and resumable terminal checkpoints are retained beside each evidence run. Execution JSON records sequences, residues, effective residue budgets, padding, throughput, runtime, and VRAM.

## Gate-1 decision

**No meaningful LoRA gap through 50k**

## Limitations

- Foldseek uses query-normalized TM-score with exact refinement. Nested-set pruning gives exact 50k maxima for possible OOD backbones; already-standard backbones retain their exact 10k maximum as a certified 50k lower bound because that is sufficient for the fixed 0.5 split.
- One deterministic ProteinMPNN generation represents each backbone in primary manifests.
- Bootstrap intervals quantify fixed-test protein sampling uncertainty, not training-seed uncertainty.
- This screen does not test or interpret H1-H4.
