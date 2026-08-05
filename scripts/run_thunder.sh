#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/bootstrap_model.py
mkdir -p artifacts/gate3
python scripts/run_gate3.py 2>&1 | tee artifacts/gate3/thunder_run.log
