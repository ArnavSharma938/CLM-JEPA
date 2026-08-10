import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from experiments import (
    TRAIN_MANIFEST, VALIDATION_MANIFEST, prepare_pilot_manifests,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.real_model
@pytest.mark.skipif(
    os.environ.get("RUN_REAL_GATE45_TEST") != "1",
    reason="set RUN_REAL_GATE45_TEST=1 for the pinned ChemFM-1B CUDA integration test",
)
def test_one_real_batch_covers_training_tracking_validation_checkpoint_and_output(tmp_path):
    assert torch.cuda.is_available()
    prepare_pilot_manifests()
    output = tmp_path / "result.json"
    checkpoints = tmp_path / "checkpoints"
    wandb_dir = tmp_path / "wandb"
    wandb_dir.mkdir()
    command = [
        sys.executable,
        str(ROOT / "src" / "train.py"),
        "--gate", "4",
        "--dataset", "uspto_mit_synthesis",
        "--condition", "clm_jepa",
        "--seed", "533",
        "--learning-rate", "0.0001",
        "--k", "1",
        "--lambda-eff", "1.0",
        "--dropout", "0.0",
        "--epochs", "1",
        "--batch-size", "4",
        "--gradient-accumulation-steps", "2",
        "--data-fraction", "0.000001",
        "--train-manifest", str(TRAIN_MANIFEST),
        "--validation-manifest", str(VALIDATION_MANIFEST),
        "--checkpoint-dir", str(checkpoints),
        "--max-train-rows", "4",
        "--max-validation-rows", "15",
        "--output", str(output),
    ]
    environment = os.environ.copy()
    environment.update({
        "WANDB_MODE": "offline",
        "WANDB_PROJECT": "clm-jepa",
        "WANDB_DIR": str(wandb_dir),
    })
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["compute"]["optimizer_steps"] == 1
    assert result["compute"]["jepa_active_microbatches"] == 1
    assert result["validation_metrics"]
    assert result["diagnostics"]
    assert result["predictions"]
    selected = Path(result["selected_checkpoint"])
    assert (selected / "training_state.pt").exists()
    assert list(selected.rglob("adapter_model.safetensors"))
    assert list(wandb_dir.rglob("*.wandb"))
    assert list(wandb_dir.rglob("wandb-metadata.json"))
