from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clm_jepa.tracking import TrackingContext, WandbTracker


def main() -> None:
    context = TrackingContext(
        task="infrastructure",
        dataset="none",
        condition="logging-smoke",
        seed=533,
        data_fraction=0.0,
        resolved_hyperparameters={"purpose": "schema-validation"},
    )
    tracker = WandbTracker(context, run_name="logging-smoke")
    tracker.log_training_step(
        step=0, native_loss=0.0, jepa_loss=0.0, total_loss=0.0,
        gradient_norm=0.0, learning_rate=0.0, jepa_active=False,
        batch_tokens=0, model_calls=0, effective_tokens=0,
        peak_vram_bytes=0, estimated_flops=0.0,
    )
    tracker.log_evaluation(
        step=0, split="validation", task_metrics={"schema_ok": 1.0},
        validity=1.0, native_loss=0.0,
    )
    tracker.finish({"smoke_passed": True})


if __name__ == "__main__":
    main()
