"""Pinned scientific constants for the protein diagnostic."""

CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"
CANONICAL_AA_SET = frozenset(CANONICAL_AA)

RITA_MODEL_ID = "lightonai/RITA_m"
RITA_REVISION = "d819157a3a96d278500232b2cbb2a02d9646bcf6"
RITA_TOKENIZER_REVISION = RITA_REVISION
RITA_CONTEXT = 1024
RITA_EOS_ID = 2
RITA_PAD_ID = 1
RITA_VOCAB_SIZE = 26

UNIREF50_ID = "alejoacelas/uniref50-2025-10"
UNIREF50_REVISION = "2b1724ac1b0adb362f6af3e80b87e675bf8f8933"

RITA_REPOSITORY = "https://github.com/lightonai/RITA"
RITA_FITNESS_SOURCE = (
    "https://github.com/lightonai/RITA/blob/master/compute_fitness.py"
)
PROTEINGYM_RITA_SOURCE = (
    "https://github.com/OATML-Markslab/ProteinGym/blob/main/"
    "proteingym/baselines/rita/compute_fitness.py"
)

NEXTLAT_REPOSITORY = "https://github.com/JaydenTeoh/NextLat"
NEXTLAT_COMMIT = "3770be6009cea2b3c455a9ce7f2ca88b504bb955"
STP_REPOSITORY = "https://github.com/galilai-group/llm-jepa"
STP_COMMIT = "ea0017c654ad917066ff32afc88276bea8ca5f7e"

