from __future__ import annotations

from pathlib import Path


DATASET_ID = "Propedia v26"
DATASET_RELEASE = "v17"
DATASET_URL = "https://bioinfo.dcc.ufmg.br/propedia26/data/propedia26_v17.csv"
DATASET_FILE = "propedia26_v17.csv"
DATASET_SHA256 = "22ba847c52ca4f6e6a4aa111c45032d25cfbee376a6218237e19ca2516437d8d"
CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"
MODEL_ID = "facebook/esm2_t33_650M_UR50D"
MODEL_REVISION = "08e4846e537177426273712802403f7ba8261b6c"
MAX_MODEL_TOKENS = 1024
DIAMOND_VERSION = "2.2.8"
CLUSTER_IDENTITY = 30.0
CLUSTER_COVERAGE = 80.0
HPO_SEED = 7
EVIDENCE_SEED = 11


def default_data_root() -> Path:
    return Path("data/protein/propedia26")


def default_run_root() -> Path:
    return Path("runs/protein/propedia26_gate0")

