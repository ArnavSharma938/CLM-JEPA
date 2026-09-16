#!/usr/bin/env bash
set -euo pipefail

cd /workspace/CLM-JEPA
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y git mmseqs2 build-essential
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
print({"torch": torch.__version__, "cuda": torch.version.cuda,
       "gpu": torch.cuda.get_device_name(0)})
PY
.venv/bin/python -m pytest protein/tests -q
: "${CLM_JEPA_BUNDLE_SHA256:?set the uploaded bundle SHA256}"
CLM_JEPA_BASE_COMMIT=e2ba2dfe7495b12ca724a8eafa88fcc47c1c74bf \
  .venv/bin/python protein/scripts/prepare_causal_pilot.py all
