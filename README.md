# Repository map

| Path | Contents |
|---|---|
| [AGENTS.md](AGENTS.md) | Workspace, token-budget, compute, and preservation rules |
| [src/](src/) | Native/STP training and frozen scientific analysis |
| [src/chemfm.py](src/chemfm.py) | Model loading, checkpoint compatibility, serialization, collation |
| [src/train.py](src/train.py), [src/stp.py](src/stp.py) | Native trainer and released/paper STP objectives |
| [src/latent_predictability.py](src/latent_predictability.py) | PCA, ridge, residual probes, latent metrics |
| [src/frozen_geometry.py](src/frozen_geometry.py), [src/geodesic_audit.py](src/geodesic_audit.py) | Geometry analysis infrastructure |
| [scripts/](scripts/README.md) | Experiment runners and saved-artifact analyses |
| [tests/](tests/) | Synthetic/unit tests and optional real-model integration |
| [docs/reports/](docs/reports/README.md) | Reports 00–03 and archived protocol provenance |
| [docs/CLEANUP.md](docs/CLEANUP.md) | Cleanup boundaries and recovery information |
| [data/](data/) | Local datasets, frozen panels, and split metadata |
| [models/](models/) | Local base model weights |
| [references/](references/) | Pinned upstream reference sources |
| [runs/](runs/) | Checkpoints, caches, probes, raw evidence, compact summaries |
| [requirements.txt](requirements.txt) | Python dependencies |
