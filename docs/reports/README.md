# Experiment reports

Reports are numbered by dependency order. Hardware appears in the filename only when it determines the execution path or scope. Every report begins with its result; detailed protocols, tables, uncertainty, and artifact paths follow.

| # | Report | Scope | Primary result |
|---:|---|---|---|
| 00 | [LLM-JEPA method fidelity](00_LLM_JEPA_METHOD_FIDELITY.md) | Static and executable parity | Default symmetric cosine JEPA is functionally equivalent after ChemFM serialization/EOS substitutions; optional upstream branches are outside the claim. |
| 01 | [Original cosine failure](01_USPTO_MIT_ORIGINAL_COSINE_FAILURE.md) | Gate 5, geometry, 512-reaction coupling | Initial top-1 tied 2/32. On 512 reactions native/cLM-JEPA were 24/17; cLM-JEPA target variance was 495× below native with strong residual retrieval but no measured decoder coupling. |
| 02 | [Target stop-gradient](02_USPTO_MIT_TARGET_STOP_GRADIENT.md) | 32- and 512-reaction panels | On 512 reactions native/stop-gradient were 24/26, McNemar p=0.8318. Variance increased 12.03× versus symmetric JEPA but remained 46.3× below native; CE was 7.69% worse than native. |
| 03 | [SIGReg batch-2 k ablation](03_USPTO_MIT_SIGREG_BATCH2_K_ABLATION.md) | k=0 versus k=1, epoch 2 | Native/k0/k1 top-1 were 7/2/3 of 256. SIGReg did not decrease and neither readout restored native-scale geometry. |
| 04 | [SIGReg batch 128](04_USPTO_MIT_SIGREG_BATCH128.md) | Exact streamed N=128 statistic | Geometry moved near native scale, but optimizer updates fell 16× and generation degraded to 0/256 top-1 with CE 1.080978. The training comparison is confounded by cadence. |
| 05 | [SIGReg batch-16 preflight](05_USPTO_MIT_SIGREG_BATCH16_PREFLIGHT_RTX4050.md) | RTX 4050 equivalence/calibration | Streamed/direct loss and parameter gradients matched exactly. Proposed weighting was 1.20% of NTP gradient norm; exact N=16 halves update cadence and requires a matched native control. |
| 06 | [Frozen SIGReg gradient response](06_USPTO_MIT_SIGREG_GRADIENT_RESPONSE_RTX4050.md) | RTX 4050, no optimizer steps | SIGReg endpoint norms stayed 0.0376–0.0427 across the contraction trajectory; gradients did not vanish under the measured common-direction contraction. |
| 07 | [Controlled SIGReg batch 16](07_USPTO_MIT_SIGREG_BATCH16_A6000.md) | A6000, matched four-epoch cadence | Epoch-4 native/SIGReg top-1 were 6/3 of 256. SIGReg remained above its epoch-1 mean and variance stayed 16.9×/14.8× below native. |
| 08 | [MSE and MSE+SIGReg](08_USPTO_MIT_MSE_SIGREG_A6000.md) | A6000, staged ablation | MSE+SIGReg restored source/target variance to 90.3%/55.2% of native at epoch 4; exact top-1 tied 6/256 while CE was 3.36% worse. |
| 09 | [Official five-view endpoint](09_USPTO_MIT_OFFICIAL_ENDPOINT_A6000.md) | A6000, 1,280 unique reactions | Native/cLM-JEPA top-1 were 3.906%/3.125%; difference -0.781 pp, 95% CI [-1.719,+0.156], McNemar p=0.1433. Prespecified futility stopped extension to 3,300. |
| 10 | [GSM8K LLM-JEPA reference](10_GSM8K_LLM_JEPA_REFERENCE_A6000.md) | A6000, DeepSeek-1.5B, two epochs | NTP/LLM-JEPA accuracy was 36/28 of 300, p=0.229. LLM-JEPA target variance was 1.45× below NTP rather than ChemFM's 495× contraction, but this reduced run was not a successful behavioral control. |

## Current endpoint conclusion

The selected epoch-4 MSE+SIGReg cLM-JEPA checkpoint did not improve official five-view USPTO-MIT generation over the cadence-matched native checkpoint. At the prespecified 1,280-reaction interim, the +1 percentage-point effect of interest was excluded by the frozen futility rule. This conclusion is limited to the compared checkpoints, seed, pilot-training exposure, and official test sample.
