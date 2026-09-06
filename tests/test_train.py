"""Native/STP trainer contracts; retired objective tests live in Git history."""
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from train import (
    ADAM_BETAS, ADAM_EPSILON, MIN_LEARNING_RATE, WARMUP_RATIO, WEIGHT_DECAY,
    restore_training_checkpoint, validation_selector, condition_family, NativeObjective,
)

def test_supported_conditions_and_retired_objectives():
    assert condition_family("native") == "native"
    assert condition_family("stp_released") == "semantic_tube_prediction_released"
    assert condition_family("stp_paper") == "semantic_tube_prediction_paper"
    for retired in ("clm_jepa", "clm_jepa_mse_sigreg", "clm_jepa_pair_residual", "clm_jepa_vjepa2_1"):
        with pytest.raises(ValueError): condition_family(retired)

def test_native_objective_is_exactly_the_causal_model_loss():
    parameter=torch.tensor(2.,requires_grad=True)
    class Model:
        def __call__(self,**kwargs):
            assert set(kwargs)=={"input_ids","attention_mask","labels"}
            return SimpleNamespace(loss=parameter.square(),logits=torch.zeros(1,2,3))
    batch={key:torch.ones(1,2,dtype=torch.long) for key in ("input_ids","attention_mask","labels")}
    output=NativeObjective()(Model(),batch)
    output.loss.backward()
    assert parameter.grad.item()==4 and output.loss is output.native_loss
    assert not output.jepa_active and output.jepa_loss is None


def test_native_training_orchestration_without_chemfm(monkeypatch, tmp_path):
    """Exercise the refactored loop on one synthetic CPU parameter, never a model cache."""
    import train as training

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2.0]))
            self.device = torch.device('cpu')
            self.config = SimpleNamespace(_attn_implementation='synthetic')

        def cuda(self):
            return self

        def num_parameters(self, **kwargs):
            return 1

        def forward(self, **kwargs):
            return SimpleNamespace(loss=self.weight.square(), logits=torch.zeros(1, 2, 3))

    model = Model()
    monkeypatch.setattr(training, 'set_seed', lambda seed: None)
    monkeypatch.setattr(training.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(training.torch.cuda, 'is_current_stream_capturing', lambda: False)
    monkeypatch.setattr(training.torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(training.torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(training, 'add_predictor_tokens', lambda *a: [])
    class Tokenizer(list):
        eos_token_id = 2
    monkeypatch.setattr(training, 'load_reaction_tokenizer', lambda *a: Tokenizer([0, 1, 2]))
    monkeypatch.setattr(training, 'load_lora_model', lambda *a, **kw: model)
    monkeypatch.setattr(training, 'read_rows', lambda *a, **kw: [{}])
    monkeypatch.setattr(training, 'validate_serialization_endings', lambda *a: None)
    monkeypatch.setattr(training, 'file_sha256', lambda *a: 'synthetic')
    monkeypatch.setattr(training, 'ReactionCollator', lambda *a, **kw: lambda rows: {
        key: torch.ones(len(rows), 2, dtype=torch.long)
        for key in ('input_ids', 'attention_mask', 'labels')
    })
    monkeypatch.setattr(training, 'native_loss', lambda *a: 1.0)
    monkeypatch.setattr(training, 'beam_evaluate', lambda *a, **kw: (
        {'exact_top1': 1.0, 'valid_rate': 1.0}, ['synthetic']))
    saved = []
    monkeypatch.setattr(training, 'save_training_checkpoint', lambda path, *a, **kw: saved.append(path))
    monkeypatch.setattr(training, 'load_adapter_checkpoint', lambda *a: None)
    args = SimpleNamespace(
        seed=533, dataset='uspto_mit_synthesis', condition='native', gate=4,
        train_manifest=tmp_path/'train.csv', validation_manifest=tmp_path/'val.csv',
        max_train_rows=None, max_validation_rows=None, attention_implementation=None,
        lora_rank=8, lora_alpha=8, gradient_checkpointing=False, dataloader_workers=0,
        dataloader_prefetch_factor=2, batch_size=1, pin_memory=False,
        gradient_accumulation_steps=1, epochs=1, learning_rate=1e-4, fused_adamw=False,
        stp_lambda=.02, stop_after_epoch=1, k=0, lambda_eff=1., dropout=.5,
        eval_generation_batch_size=1, evaluation_epochs=[1], no_wandb=True,
        data_fraction=1., resume_from=None, checkpoint_dir=tmp_path/'checkpoints',
        final_checkpoint_only=False,
    )
    result = training.train(args)
    assert model.weight.item() < 2.0
    assert len(saved) == 1 and result['selected_epoch'] == 1
    assert result['compute']['optimizer_steps'] == result['compute']['model_calls'] == 1
    assert result['compute']['jepa_active_microbatches'] == 0
    assert result['diagnostics']['type'] == 'native_training_summary'

def test_chemfm_optimizer_and_scheduler_settings_are_fully_resolved():
    assert ADAM_BETAS == (0.9, 0.999)
    assert ADAM_EPSILON == 1e-8
    assert WEIGHT_DECAY == 0.01
    assert WARMUP_RATIO == 0.05
    assert MIN_LEARNING_RATE == 1e-5

def test_checkpoint_selector_uses_only_frozen_task_metric():
    assert validation_selector(
        {"exact_top1": 0.4, "valid_rate": 0.1}, "forward"
    ) == (0.4,)
    assert validation_selector(
        {"recall_at5": 0.3, "lower_bound_precision_at5": 0.2}, "metabolism"
    ) == (0.3, 0.2)

def test_checkpoint_restore_moves_loader_rng_state_back_to_cpu(monkeypatch, tmp_path):
    expected_state = torch.Generator().get_state()

    class RelocatedState:
        def cpu(self):
            return expected_state

    class Stateful:
        def load_state_dict(self, state):
            self.state = state

    checkpoint_state = {
        "planned_epochs": 4,
        "optimizer": {"optimizer": True},
        "scheduler": {"scheduler": True},
        "loader_generator_state": RelocatedState(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": [],
    }
    monkeypatch.setattr("train.load_adapter_checkpoint", lambda *args: None)
    monkeypatch.setattr("train.torch.load", lambda *args, **kwargs: checkpoint_state)
    monkeypatch.setattr("train.torch.cuda.set_rng_state_all", lambda states: None)
    generator = torch.Generator()
    restored = restore_training_checkpoint(
        tmp_path,
        type("Model", (), {"device": torch.device("cpu")})(),
        Stateful(), Stateful(), generator, 4,
    )
    assert restored is checkpoint_state
    assert torch.equal(generator.get_state(), expected_state)
