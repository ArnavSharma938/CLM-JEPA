from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from vjepa2_1 import (
    CausalLanguagePredictor,
    CausalSuffixMaskSampler,
    DenseVJEPA21,
    DenseVJEPA21Config,
    LayerCapture,
    ProgressiveContextSchedule,
    component_gradient_norms,
    context_distance_weights,
)


IGNORE_INDEX = -100


def tiny_config(**overrides):
    values = {
        "encoder_dim": 8,
        "encoder_depths": (1, 2, 3, 4),
        "predictor_dim": 8,
        "predictor_depth": 2,
        "predictor_heads": 2,
        "num_mask_tokens": 2,
        "short_mask_tokens": 1,
        "seed": 0,
    }
    values.update(overrides)
    return DenseVJEPA21Config(**values)


def reaction_batch():
    return {
        "input_ids": torch.tensor([
            [1, 2, 3, 4, 5, 6, 7],
            [2, 1, 4, 3, 7, 6, 0],
        ]),
        "attention_mask": torch.tensor([
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0],
        ]),
        # Position 2 is the target <prostart>; future sampling starts after it.
        "labels": torch.tensor([
            [-100, -100, 3, 4, 5, 6, 7],
            [-100, -100, 4, 3, 7, 6, -100],
        ]),
    }


class CausalBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.projection = nn.Linear(dim, dim)

    def forward(self, values):
        # Prefix means are strictly causal and make leakage tests sensitive.
        positions = torch.arange(1, values.size(1) + 1, device=values.device)
        causal = values.cumsum(dim=1) / positions[None, :, None]
        return (torch.tanh(self.projection(causal)),)


class TinyBackbone(nn.Module):
    def __init__(self, vocab=16, dim=8, depth=4):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList([CausalBlock(dim) for _ in range(depth)])

    def forward(self, input_ids, attention_mask, **_kwargs):
        values = self.embed_tokens(input_ids)
        for layer in self.layers:
            values = layer(values)[0]
        values = values * attention_mask.unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=values)


class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyBackbone()
        self.lm_head = nn.Linear(8, 16)

    def forward(self, input_ids, attention_mask, labels, **_kwargs):
        hidden = self.model(input_ids, attention_mask).last_hidden_state
        logits = self.lm_head(hidden)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        return SimpleNamespace(loss=loss, logits=logits)


def test_causal_suffix_sampler_aligns_product_positions_and_two_scales():
    sampler = CausalSuffixMaskSampler(tiny_config())
    batch = reaction_batch()
    short = sampler.sample(batch["labels"], batch["attention_mask"], IGNORE_INDEX)
    long = sampler.sample(batch["labels"], batch["attention_mask"], IGNORE_INDEX)
    assert short.mask_token_indices.tolist() == [0, 1]
    assert long.mask_token_indices.tolist() == [1, 1]
    for sampled in (short, long):
        positions = torch.arange(batch["labels"].size(1)).unsqueeze(0)
        assert torch.equal(sampled.future_mask, (
            (positions >= sampled.boundaries[:, None])
            & (positions < sampled.active_lengths[:, None])
        ))
        assert (sampled.boundaries > sampled.target_starts).all()
        assert not (sampled.context_mask & sampled.future_mask).any()
    assert long.horizons.ge(short.horizons).all()


def test_context_distance_weights_are_continuous_across_source_and_prefix():
    sampler = CausalSuffixMaskSampler(tiny_config(seed=0))
    batch = reaction_batch()
    sampled = sampler.sample(batch["labels"], batch["attention_mask"], IGNORE_INDEX)
    weights = context_distance_weights(sampled, torch.float32)
    for row, boundary in enumerate(sampled.boundaries.tolist()):
        expected = torch.tensor([
            1.0 / math_value**0.5 for math_value in range(boundary, 0, -1)
        ])
        torch.testing.assert_close(weights[row, :boundary], expected)
        assert weights[row, boundary:].eq(0).all()


def test_predictor_has_dense_shapes_and_cannot_read_future_student_features():
    torch.manual_seed(1)
    config = tiny_config()
    predictor = CausalLanguagePredictor(config)
    batch = reaction_batch()
    mask = CausalSuffixMaskSampler(config).sample(
        batch["labels"], batch["attention_mask"], IGNORE_INDEX
    )
    levels = [torch.randn(2, 7, 8) for _ in config.encoder_depths]
    first_future, first_context = predictor(levels, mask)
    changed = [value.clone() for value in levels]
    for value in changed:
        value[mask.future_mask] = torch.randn_like(value[mask.future_mask]) * 1000
    second_future, second_context = predictor(changed, mask)
    assert first_future.shape == (2, 7, 32)
    assert first_context.shape == (2, 7, 32)
    torch.testing.assert_close(first_future, second_future)
    torch.testing.assert_close(first_context, second_context)


def test_real_llama_prefix_features_have_no_future_token_leakage():
    torch.manual_seed(9)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_dropout=0.0,
    )).eval()
    first_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    second_ids = first_ids.clone()
    second_ids[:, 4:] = torch.tensor([[11, 12]])
    attention = torch.ones_like(first_ids)
    with torch.inference_mode(), LayerCapture(model.model, (1, 2, 3, 4)) as first:
        model(input_ids=first_ids, attention_mask=attention, use_cache=False)
        first_levels = tuple(value.clone() for value in first.values())
    with torch.inference_mode(), LayerCapture(model.model, (1, 2, 3, 4)) as second:
        model(input_ids=second_ids, attention_mask=attention, use_cache=False)
        second_levels = second.values()
    for left, right in zip(first_levels, second_levels):
        torch.testing.assert_close(left[:, :4], right[:, :4], rtol=0, atol=0)


def test_progressive_context_schedule_matches_reference_shape():
    schedule = ProgressiveContextSchedule(0.5, total_steps=90)
    assert schedule.start == 10
    assert schedule.end == 20
    assert schedule.value(9) == 0.0
    assert schedule.value(10) == 0.0
    assert schedule.value(15) == 0.25
    assert schedule.value(20) == 0.5


def test_jepa_disabled_is_exact_native_ntp_parity():
    torch.manual_seed(2)
    model = TinyCausalLM()
    method = DenseVJEPA21(tiny_config(), total_steps=10, ignore_index=IGNORE_INDEX)
    batch = reaction_batch()
    native = model(**batch, use_cache=False, return_dict=True)
    output = method(model, batch, jepa_weight=0.0, global_step=0)
    torch.testing.assert_close(output.loss, native.loss, rtol=0, atol=0)
    assert output.jepa_loss is None
    assert method.mask_sampler.sample_step == 0


def test_ema_target_is_stop_gradient_and_updates_only_by_polyak_rule():
    torch.manual_seed(3)
    model = TinyCausalLM()
    config = tiny_config(ema_start=0.5, ema_end=0.5)
    method = DenseVJEPA21(config, total_steps=10, ignore_index=IGNORE_INDEX)
    method.initialize_ema(model)
    first_name = method.ema_parameter_names[0]
    first_buffer = method.ema_buffer_names[0]
    before = getattr(method, first_buffer).clone()
    online = dict(model.model.named_parameters())[first_name]
    with torch.no_grad():
        online.add_(2.0)
    tau = method.update_ema(model)
    assert tau == 0.5
    torch.testing.assert_close(getattr(method, first_buffer), before + 1.0)
    assert getattr(method, first_buffer).requires_grad is False
    assert all(not parameter.requires_grad for parameter in method.target_norms.parameters())

    output = method(model, reaction_batch(), jepa_weight=1.0, global_step=5)
    output.jepa_loss.backward()
    assert all(getattr(method, name).grad is None for name in method.ema_buffer_names)
    assert all(parameter.grad is None for parameter in method.target_norms.parameters())
    assert any(parameter.grad is not None for parameter in method.predictor.parameters())


def test_checkpoint_resume_restores_predictor_ema_schedule_and_next_mask():
    torch.manual_seed(4)
    model = TinyCausalLM()
    method = DenseVJEPA21(tiny_config(), total_steps=12, ignore_index=IGNORE_INDEX)
    method.initialize_ema(model)
    method(model, reaction_batch(), jepa_weight=1.0, global_step=0)
    method.update_ema(model)
    saved = copy_state = method.checkpoint_state()

    resumed_model = TinyCausalLM()
    resumed_model.load_state_dict(model.state_dict())
    resumed = DenseVJEPA21(tiny_config(), total_steps=12, ignore_index=IGNORE_INDEX)
    resumed.load_checkpoint_state(copy_state, resumed_model)
    assert resumed.ema_update_step == method.ema_update_step
    assert resumed.mask_sampler.sample_step == method.mask_sampler.sample_step
    for key, value in method.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[key], value)

    expected = method.mask_sampler.sample(
        reaction_batch()["labels"], reaction_batch()["attention_mask"], IGNORE_INDEX
    )
    actual = resumed.mask_sampler.sample(
        reaction_batch()["labels"], reaction_batch()["attention_mask"], IGNORE_INDEX
    )
    torch.testing.assert_close(actual.boundaries, expected.boundaries)
    torch.testing.assert_close(actual.mask_token_indices, expected.mask_token_indices)


def test_functional_ema_target_runs_through_a_real_peft_llama():
    torch.manual_seed(10)
    base = LlamaForCausalLM(LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_dropout=0.0,
    ))
    model = get_peft_model(
        base,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=2,
            lora_alpha=2,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
        ),
        adapter_name="dense-test",
    )
    config = DenseVJEPA21Config(
        encoder_dim=16,
        encoder_depths=(1, 2, 3, 4),
        predictor_dim=16,
        predictor_depth=2,
        predictor_heads=4,
        num_mask_tokens=2,
        short_mask_tokens=1,
        seed=2,
    )
    method = DenseVJEPA21(config, total_steps=8, ignore_index=IGNORE_INDEX)
    batch = reaction_batch()
    output = method(model, batch, jepa_weight=1.0, global_step=4)
    assert output.loss.isfinite()
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in method.predictor.parameters())
    method.update_ema(model)


def test_component_gradient_norms_are_exact_and_do_not_mutate_grad_buffers():
    left = nn.Parameter(torch.tensor([3.0, 4.0]))
    right = nn.Parameter(torch.tensor([2.0]))
    metrics = component_gradient_norms(
        {"quadratic": left.square().sum() + 2.0 * right.square().sum()},
        {"student": (left,), "predictor": (right,)},
    )
    assert metrics == {
        "quadratic_student_gradient_norm": 10.0,
        "quadratic_predictor_gradient_norm": 8.0,
    }
    assert left.grad is None
    assert right.grad is None


def test_dense_module_can_inherit_bfloat16_chemfm_dtype():
    method = DenseVJEPA21(
        tiny_config(), total_steps=4, ignore_index=IGNORE_INDEX,
    ).to(dtype=torch.bfloat16)
    assert method.online_norms[0].weight.dtype == torch.bfloat16
    assert method.predictor.fusion[0].weight.dtype == torch.bfloat16
    values = torch.randn(2, 7, 8, dtype=torch.bfloat16)
    normalized = method.online_norms[0](values)
    assert normalized.dtype == torch.bfloat16
