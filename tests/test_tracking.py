import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train import TrackingContext, WandbTracker


class FakeRun:
    def __init__(self):
        self.logged = []
        self.summary = {}
        self.finished = False

    def log(self, payload, step):
        self.logged.append((step, payload))

    def finish(self):
        self.finished = True


class FakeWandb:
    def __init__(self):
        self.kwargs = None
        self.run = FakeRun()

    def init(self, **kwargs):
        self.kwargs = kwargs
        return self.run


def context():
    return TrackingContext("forward", "uspto_mit_synthesis", "clm-jepa", 533, 0.1, {"lr": 1e-4})


def test_tracker_requires_environment_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    with pytest.raises(EnvironmentError):
        WandbTracker(context(), run_name="missing", wandb_module=FakeWandb())


def test_tracker_allows_offline_runs_without_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("WANDB_MODE", "offline")
    fake = FakeWandb()
    tracker = WandbTracker(context(), run_name="offline", wandb_module=fake)
    assert fake.kwargs["dir"] == "runs/wandb"
    tracker.finish()


def test_tracker_logs_required_training_and_evaluation_fields(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "test-only")
    monkeypatch.setenv("WANDB_PROJECT", "clm-jepa")
    fake = FakeWandb()
    tracker = WandbTracker(context(), run_name="test", wandb_module=fake)
    tracker.log_training_step(
        step=1, native_loss=1.2, jepa_loss=0.3, total_loss=1.5,
        gradient_norm=2.0, learning_rate=1e-4, jepa_active=True,
        batch_tokens=100, model_calls=3, effective_tokens=240,
        max_gradient_parameter="base_model.layer.lora_A",
        max_parameter_gradient_norm=3.5,
        peak_vram_bytes=1024, estimated_flops=500.0,
    )
    tracker.log_evaluation(
        step=1, split="validation", task_metrics={"top1": 0.4},
        validity=0.9, native_loss=1.1,
    )
    train = fake.run.logged[0][1]
    assert {
        "train/native_loss", "train/jepa_loss", "train/total_loss",
        "train/gradient_norm", "train/max_gradient_parameter",
        "train/max_parameter_gradient_norm",
    } <= train.keys()
    assert {"compute/jepa_active_batches", "compute/total_tokens", "compute/tokens_per_second", "compute/peak_vram_bytes"} <= train.keys()
    assert fake.run.logged[1][1]["validation/top1"] == 0.4
    assert "WANDB_API_KEY" not in str(fake.kwargs)
    tracker.finish({"selected": True})
    assert fake.run.summary["selected"] is True
    assert fake.run.finished


def test_tracker_rejects_secret_like_config(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "test-only")
    bad = TrackingContext("forward", "x", "native", 1, 1.0, {"api_key": "bad"})
    with pytest.raises(ValueError):
        WandbTracker(bad, run_name="bad", wandb_module=FakeWandb())
