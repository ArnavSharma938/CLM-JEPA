from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


DATASET_ID = "microsoft/Dayhoff"
DATASET_REVISION = "7f6771db1bfad58011fcadf72f86eacbef39a482"
MODEL_ID = "facebook/esm2_t33_650M_UR50D"
MODEL_REVISION = "08e4846e537177426273712802403f7ba8261b6c"
CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"
COMMON_MASK_SEED = 20250917
HPO_SEED = 7
EVIDENCE_SEED = 11
REPLICATION_SEED = 23
LOADS = (10_000, 50_000)
MODES = ("ft", "dtft", "lora8", "lora64")

Mode = Literal["base", "ft", "dtft", "lora8", "lora64"]


@dataclass(frozen=True)
class TrainConfig:
    mode: Mode
    seed: int
    learning_rate: float
    train_manifest: Path
    validation_manifest: Path
    output_dir: Path
    model_revision: str = MODEL_REVISION
    common_mask_seed: int = COMMON_MASK_SEED
    microbatch_tokens: int = 4096
    effective_residues_per_step: int = 32768
    maximum_epochs: int = 10
    maximum_optimizer_steps: int | None = None
    validation_every_steps: int = 100
    early_stopping_checks: int = 6
    num_workers: int = 4
    attention_backend: str = "sdpa"
    gradient_checkpointing: bool = True
    compile_model: bool = False
    save_checkpoints: bool = True

    def validate(self) -> None:
        if self.seed == HPO_SEED and "evidence" in self.output_dir.parts:
            raise ValueError("seed 7 is reserved for LR probes")
        if self.mode not in ("base", *MODES):
            raise ValueError(f"unsupported mode: {self.mode}")
        if self.mode == "base" and self.learning_rate != 0:
            raise ValueError("Base is not trained")
        if self.microbatch_tokens <= 0 or self.effective_residues_per_step <= 0:
            raise ValueError("token budgets must be positive")
        if self.effective_residues_per_step < self.microbatch_tokens:
            raise ValueError("effective budget must be at least one microbatch budget")
        if self.attention_backend not in {"eager", "sdpa", "flash_attention_2"}:
            raise ValueError("unknown attention backend")
        if self.maximum_optimizer_steps is not None and self.maximum_optimizer_steps < 1:
            raise ValueError("maximum_optimizer_steps must be positive")

    def serializable(self) -> dict:
        result = asdict(self)
        for key in ("train_manifest", "validation_manifest", "output_dir"):
            result[key] = str(result[key])
        return result


# Intentionally small, literature-plausible screens; values are selected by validation NLL.
DENSE_HPO_LRS = (1e-6, 3e-6, 1e-5)
LORA_HPO_LRS = (1e-5, 3e-5, 1e-4)
HPO_EQUIVALENCE_NLL = 1e-4
