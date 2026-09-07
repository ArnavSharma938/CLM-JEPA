"""Reuse Native evidence; run projected treatment sequentially on one GPU."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from run_stp_matrix import TRAIN, VALIDATION, PANEL, read_jsonl, sha256, evaluation_env
from run_stp_completion import native_evaluation

DEFAULT_OUTPUT = ROOT / 'runs/decoder_projected'
PANEL_HASH = 'a2e6202a4abaf9a70f4700e04299a09964d38c10fce004022dc43e759aa6057d'


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def decision(effects):
    if len(effects) == 2:
        if all(x > 0 for x in effects):
            return 'proceed'
        if all(x <= 0 for x in effects):
            return 'stop'
        return 'seed_1301_required'
    if len(effects) == 3:
        # Operationalize three-seed direction/mean conservatively, before results.
        return 'proceed' if sum(x > 0 for x in effects) >= 2 and sum(effects) > 0 else 'stop'
    raise ValueError('decision requires two or three complete effects')


def preflight(root):
    import torch
    from chemfm import ReactionCollator, load_reaction_tokenizer, TOKENIZER_DIR, MODEL_DIR, canonicalize
    from train import read_rows, validate_serialization_endings
    from decoder_projected import eligible_transitions
    rows = read_rows('uspto_mit_synthesis', path=TRAIN)
    groups = Counter(row['group_id'] for row in rows)
    assert len(rows) == 1280 and len(groups) == 256 and set(groups.values()) == {5}
    assert all({int(row['augmentation_index']) for row in rows if row['group_id'] == group} == set(range(5)) for group in groups)
    identities = {(canonicalize(row['src']), canonicalize(row['tgt'])) for row in rows}
    assert len(identities) == 256
    assert sha256(PANEL) == PANEL_HASH
    panel = read_jsonl(PANEL)
    assert len(panel) == 512
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    collator = ReactionCollator(tokenizer)
    validate_serialization_endings(collator, rows, tokenizer.eos_token_id)
    transitions, lengths = [], []
    for row in rows:
        batch = collator([row])
        mask = eligible_transitions(batch, tokenizer.convert_tokens_to_ids('<prostart>'), tokenizer.eos_token_id)
        transitions.append(int(mask.sum()))
        lengths.append(int(batch['attention_mask'].sum()))
    controls = {}
    for seed in (533, 917, 1301):
        path = native_evaluation(8, seed) / 'predictions.jsonl'
        predictions = read_jsonl(path)
        assert len(predictions) == 512
        assert [p['reaction_identity'] for p in predictions] == [p['reaction_identity'] for p in panel]
        assert all(len(p['sources']) == 5 for p in predictions)
        summary = json.loads(path.with_name('summary.json').read_text())
        assert summary['manifest_sha256'] == PANEL_HASH
        controls[seed] = {'path': str(path), 'sha256': sha256(path),
            **{f'top{k}_percent': 100 * sum(p['target'] in p['ranked_candidates'][:k] for p in predictions) / 512 for k in (1, 3, 5, 10)},
            'validity_percent': 100 * sum(bool(c) for p in predictions for view in p['canonical_candidates_by_view'] for c in view) / (512 * 5 * 10)}
    historical = json.loads((ROOT / 'runs/stp/a6000/results/seed_917/stp/training/result.json').read_text())
    data = {'train_sha256': sha256(TRAIN), 'panel_sha256': sha256(PANEL),
        'rows': len(rows), 'unique_reactions': len(groups), 'views': 5,
        'eligible_transitions': sum(transitions), 'max_serialized_length': max(lengths),
        'controls': controls, 'model_path': str(MODEL_DIR), 'model_config': json.loads((MODEL_DIR / 'config.json').read_text()),
        'historical_stp_peak_bytes': historical['compute']['peak_vram_bytes'],
        'historical_stp_training_seconds': historical['compute']['wall_time_seconds'],
        'estimate': 'About 25-45 minutes training per seed on A6000-class GPU; one-worker five-view evaluation roughly 1-2 hours per seed, hardware dependent.',
        'resume': 'epoch checkpoints; official evaluator reuses completed reaction rows',
        'initialization': 'same seeded load_lora_model initialization as Native/STP; W0 is its initial active head, not epoch-4 continuation',
        'third_seed_rule': 'proceed iff >=2 positive effects and positive three-seed mean',
        'cuda_available': torch.cuda.is_available()}
    if torch.cuda.is_available():
        data.update(device=torch.cuda.get_device_name(), device_bytes=torch.cuda.get_device_properties(0).total_memory)
        data['matched_training_memory_preflight'] = data['device_bytes'] > data['historical_stp_peak_bytes'] + 512 * 1024**2
    write_json(root / 'preflight.json', data)
    print(json.dumps({k: data[k] for k in ['rows','unique_reactions','eligible_transitions','controls','cuda_available']}, indent=2))
    return data


def smoke(root):
    """One shortest valid row, no optimizer update, isolated from treatment runs."""
    import torch
    from transformers import set_seed
    from chemfm import MODEL_DIR, TOKENIZER_DIR, ReactionCollator, load_lora_model, load_reaction_tokenizer, add_predictor_tokens
    from train import read_rows
    from decoder_projected import DecoderProjectedObjective
    path = root / 'smoke.json'
    if path.exists():
        return json.loads(path.read_text())
    torch.set_num_threads(1)
    set_seed(533)
    tokenizer = load_reaction_tokenizer(TOKENIZER_DIR)
    native_size = len(tokenizer)
    add_predictor_tokens(tokenizer)
    collator = ReactionCollator(tokenizer)
    rows = read_rows('uspto_mit_synthesis', path=TRAIN)
    row = min(rows, key=lambda row: collator([row])['input_ids'].numel())
    model = load_lora_model(MODEL_DIR, tokenizer, chemfm_vocab_size=native_size, attn_implementation='sdpa').cuda()
    model.train()
    method = DecoderProjectedObjective(model.get_output_embeddings().weight[:392],
        product_start_token_id=tokenizer.convert_tokens_to_ids('<prostart>'), eos_token_id=tokenizer.eos_token_id).cuda()
    batch = {k: v.cuda() for k, v in collator([row]).items() if torch.is_tensor(v)}
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    calibration = method.calibrate(model, batch)
    output = method(model, batch)
    output.loss.backward()
    assert torch.isfinite(output.loss)
    assert method.predictor[-1].weight.grad.norm() > 0
    model.eval()
    model.zero_grad(set_to_none=True)
    keys = tuple(model.state_dict())
    assert not any('predictor' in k or 'projection' in k for k in keys)
    with torch.no_grad():
        direct = model(**batch).logits
        through_objective = method(model, batch).logits
    torch.testing.assert_close(direct, through_objective, rtol=0, atol=0)
    assert not model.get_base_model().model.norm._forward_hooks
    result = {'status': 'passed', 'batch_size': 1, 'sequence_length': batch['input_ids'].shape[1],
        'calibration_smoke_only': calibration, 'svd': method.metadata,
        'ntp': float(output.native_loss), 'L_z': float(output.lz), 'L_KL': float(output.kl),
        'peak_vram_bytes': torch.cuda.max_memory_allocated(), 'seconds': time.perf_counter() - started,
        'ordinary_logits_exact': True, 'optimizer_steps': 0}
    write_json(path, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'svd'}))
    return result


def launch(command, log, env=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({'launch': command, 'log': str(log)}), flush=True)
    with log.open('a') as handle:
        subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def train_evaluate(root, seed, condition='decoder_projected'):
    base = root / f'{condition}/seed_{seed}'
    training = base / 'training'
    result_path = training / 'result.json'
    checkpoint = training / 'checkpoints/epoch_4'
    if not result_path.exists():
        command = [sys.executable, '-u', 'src/train.py', '--gate', '5', '--dataset', 'uspto_mit_synthesis',
            '--condition', condition, '--seed', str(seed), '--learning-rate', '1e-4',
            '--lora-rank', '8', '--lora-alpha', '8', '--epochs', '4', '--stop-after-epoch', '4',
            '--evaluation-epochs', '4', '--batch-size', '4', '--gradient-accumulation-steps', '4',
            '--no-gradient-checkpointing', '--fused-adamw', '--attention-implementation', 'sdpa', '--pin-memory',
            '--eval-generation-batch-size', '1', '--train-manifest', str(TRAIN), '--validation-manifest', str(VALIDATION),
            '--max-validation-rows', '2', '--checkpoint-dir', str(training / 'checkpoints'), '--no-wandb', '--output', str(result_path)]
        complete = [training / f'checkpoints/epoch_{e}' for e in (1,2,3,4)
                    if (training / f'checkpoints/epoch_{e}/training_state.pt').exists()]
        if complete:
            command.extend(['--resume-from', str(complete[-1])])
        env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'}
        launch(command, training / 'train.log', env)
    result = json.loads(result_path.read_text())
    assert result['condition'] == condition and result['seed'] == seed
    assert result['compute']['optimizer_steps'] == 320 and result['selected_epoch'] == 4
    assert result['config']['train_manifest_sha256'] == sha256(TRAIN)
    # Standard adapter artifacts carry only ChemFM; auxiliary state is a separate file.
    from safetensors import safe_open
    adapter = checkpoint / 'USPTO-MIT-Synthesis/adapter_model.safetensors'
    with safe_open(adapter, framework='pt', device='cpu') as f:
        assert not any('predictor' in k or 'projection' in k for k in f.keys())
    evaluation = base / 'evaluation'
    if not (evaluation / 'predictions.jsonl').exists() or len(read_jsonl(evaluation / 'predictions.jsonl')) != 512:
        launch([sys.executable, '-u', 'src/eval_uspto_mit_five_view_a6000.py', 'run',
            '--checkpoint', str(checkpoint), '--manifest', str(PANEL), '--workers', '4',
            '--threads-per-worker', '1', '--prompt-batch-size', '1', '--batch-mode', 'left-pad',
            '--output-dir', str(evaluation)], evaluation / 'evaluate.log', evaluation_env())
    comparison = base / 'comparison.json'
    if not comparison.exists():
        launch([sys.executable, 'src/eval_uspto_mit_five_view_a6000.py', 'summarize', '--manifest', str(PANEL),
            '--native-predictions', str(native_evaluation(8, seed) / 'predictions.jsonl'),
            '--clm-predictions', str(evaluation / 'predictions.jsonl'), '--seed', str(seed),
            '--output', str(comparison)], base / 'comparison.log')
    return json.loads(comparison.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['preflight', 'smoke', 'run'])
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    pre = preflight(args.output)
    if args.stage == 'preflight':
        return
    if args.stage == 'smoke':
        smoke(args.output)
        return
    if not pre.get('matched_training_memory_preflight', False):
        write_json(args.output / 'execution_status.json', {'status': 'blocked_before_training',
            'reason': 'matched training VRAM estimate exceeds local capacity',
            'treatment_optimizer_steps': 0, 'decision': 'pending', 'needed': 'existing larger GPU host/access'})
        raise RuntimeError('Matched training exceeds local VRAM estimate; use existing larger GPU host. No treatment launched.')
    # A smoke subprocess releases all GPU allocations before training begins.
    launch([sys.executable, __file__, 'smoke', '--output', str(args.output)], args.output / 'smoke.log')
    results = {}
    for seed in (533, 917):
        results[seed] = train_evaluate(args.output, seed)
    effects = [r['primary_top1']['absolute_difference'] for r in results.values()]
    two_seed_mean = sum(effects) / 2
    if decision(effects) == 'seed_1301_required':
        results[1301] = train_evaluate(args.output, 1301)
        effects.append(results[1301]['primary_top1']['absolute_difference'])
    baseline = {}
    if decision(effects) == 'proceed':
        for seed in (533, 917):
            baseline[seed] = train_evaluate(args.output, seed, 'full_state_nextlat')
    write_json(args.output / 'decision.json', {'decision': decision(effects),
        'two_seed_mean_effect_pp': 100 * two_seed_mean, 'mean_effect_pp': 100 * sum(effects) / len(effects),
        'per_seed': results, 'full_state_baseline': baseline if baseline else 'not_authorized_by_gate'})


if __name__ == '__main__':
    main()
