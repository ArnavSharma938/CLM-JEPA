"""Fail-closed orchestration of the frozen four-cell second-640 evaluation only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/decoder_projected/confirmation_second640'
PARENT = ROOT / 'data/clm_jepa_uspto_mit_stp_confirmation/untouched_1280.jsonl'
PANEL = PARENT.with_name('untouched_second_640.jsonl')
EXPECTED = 'f454d1f12e8035dc63c3992ce59a21809ecf3745b049dc910aec4538ed3c36f8'
CELLS = [(s,a) for s in (2027,3163) for a in ('native','decoder_projected')]

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(path.read_text())

def jsonl(path):
    return [json.loads(x) for x in path.read_bytes().splitlines()]

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')

def checkpoint(seed, arm):
    return ROOT / f'runs/decoder_projected/confirmation/seed_{seed}/{arm}/training/checkpoints/epoch_4'

def panel_check():
    lines = PARENT.read_bytes().splitlines(keepends=True)
    assert len(lines) == 1280
    b = b''.join(lines[640:1280])
    assert hashlib.sha256(b).hexdigest() == EXPECTED
    expected = [json.loads(x) for x in lines[640:1280]]
    first = [json.loads(x) for x in lines[:640]]
    ids = [r['reaction_identity'] for r in expected]
    assert len(ids) == len(set(ids)) == 640
    assert not set(ids) & {r['reaction_identity'] for r in first}
    assert [r['panel_index'] for r in expected] == list(range(640,1280))
    assert all(len(r['sources']) == 5 for r in expected)
    if PANEL.exists():
        assert PANEL.read_bytes() == b
    else:
        with PANEL.open('xb') as f:
            f.write(b)
    assert sha(PANEL) == EXPECTED
    assert [r['reaction_identity'] for r in jsonl(PANEL)] == ids
    return expected

def preflight():
    panel_check()
    hashes = {}
    for seed, arm in CELLS:
        cp = checkpoint(seed,arm)
        result = read(cp.parents[1] / 'result.json')
        assert result['seed'] == seed and result['condition'] == arm
        assert result['selected_epoch'] == 4 and result['compute']['optimizer_steps'] == 320
        previous = read(cp.parents[2] / 'evaluation_l40_8w/summary.json')
        assert sha(cp / 'USPTO-MIT-Synthesis/adapter_model.safetensors') == previous['checkpoint_adapter_sha256']
        for f in cp.rglob('*'):
            if f.is_file() and f.suffix not in ('.pt', '.md'):
                hashes[str(f.relative_to(ROOT)).replace('\\','/')] = sha(f)
    for directory in (ROOT/'models/ChemFM-1B', ROOT/'references/chemfm/finetuning/reaction_prediction/tokenizer'):
        for f in directory.rglob('*'):
            if f.is_file():
                hashes[str(f.relative_to(ROOT)).replace('\\','/')] = sha(f)
    for name in ('chemfm.py','eval_uspto_mit_five_view_a6000.py'):
        p = ROOT/'src'/name
        hashes[str(p.relative_to(ROOT)).replace('\\','/')] = sha(p)
    value = {'manifest_relative_path': str(PANEL.relative_to(ROOT)).replace('\\','/'), 'manifest_sha256': EXPECTED,
             'rows':640,'unique_reactions':640,'overlap_first640':0,'ordered_parent_rows':[640,1279],
             'file_sha256': hashes, 'bootstrap_repetitions':20000,'crossed_bootstrap_seed':20273163,
             'decision_rule':'both >0: PROCEED; both <=0: STOP; otherwise: SEED_4211_REQUIRED',
             'execution':'8 workers, prompt batch 1, left-pad; same exact fastpaths as archived Report-05 launch',
             'estimate':'Prior L40 runs took about 30 minutes per cell and about 35 GB GPU memory; four sequential cells about 2 hours plus provisioning/transfer.'}
    path = OUT/'frozen_inputs.json'
    if path.exists():
        assert read(path) == value
    else:
        write(path,value)
    print(json.dumps({'stage':'preflight_passed','panel':str(PANEL),'panel_sha256':sha(PANEL),'frozen_inputs_sha256':sha(path),'files_hashed':len(hashes)}),flush=True)

def verify_inputs():
    panel_check()
    frozen = read(OUT/'frozen_inputs.json')
    assert frozen['manifest_sha256'] == EXPECTED
    for name, digest in frozen['file_sha256'].items():
        assert sha(ROOT/name) == digest, f'input hash mismatch: {name}'

def verify_cell(seed, arm):
    rows = panel_check()
    dest = OUT/f'seed_{seed}'/arm/'evaluation'
    summary = read(dest/'summary.json')
    cp = checkpoint(seed,arm)
    assert Path(summary['checkpoint']).as_posix().endswith(cp.relative_to(ROOT).as_posix())
    digest = sha(cp/'USPTO-MIT-Synthesis/adapter_model.safetensors')
    assert summary['checkpoint_adapter_sha256'] == digest
    assert summary['manifest_sha256'] == EXPECTED
    assert Path(summary['manifest']).as_posix().endswith(PANEL.relative_to(ROOT).as_posix())
    assert summary['views_per_reaction'] == 5 and summary['beam_size'] == summary['returned_candidates_per_view'] == 10
    predictions = jsonl(dest/'predictions.jsonl')
    assert len(predictions) == summary['reactions'] == 640
    assert len({r['reaction_identity'] for r in predictions}) == 640
    for key in ('reaction_identity','official_group_index','panel_index','sources'):
        assert [r[key] for r in predictions] == [r[key] for r in rows], key
    assert [r['target'] for r in predictions] == [r['canonical_target'] for r in rows], 'canonical_target'
    assert sha(dest/'predictions.jsonl') == summary['predictions_sha256']
    return {'seed':seed,'arm':arm,'checkpoint':summary['checkpoint'],'checkpoint_adapter_sha256':digest,
            'manifest':summary['manifest'],'manifest_sha256':EXPECTED,'predictions':str(dest/'predictions.jsonl'),
            'predictions_sha256':sha(dest/'predictions.jsonl'),'prediction_count':640,'ordered_identity_equality':True}

def run():
    verify_inputs()
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), TOKENIZERS_PARALLELISM='false')
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS'):
        env[name] = '1'
    for name in ('PREALLOCATED_CACHE','BEAM_SCORER','LORA_FASTPATH','LORA_CUDAGRAPH','LAYER_CUDAGRAPH'):
        env['CHEMFM_EXACT_'+name] = '1'
    env['CHEMFM_DECODE_BATCH_SIZE'] = '10'
    for seed, arm in CELLS:
        dest = OUT/f'seed_{seed}'/arm/'evaluation'
        dest.mkdir(parents=True,exist_ok=True)
        command = [sys.executable,'-u',str(ROOT/'src/eval_uspto_mit_five_view_a6000.py'),'run',
                   '--checkpoint',str(checkpoint(seed,arm)),'--manifest',str(PANEL),'--workers','8',
                   '--prompt-batch-size','1','--batch-mode','left-pad','--threads-per-worker','1','--output-dir',str(dest)]
        if (seed,arm) == CELLS[0] and not (dest/'worker_00.jsonl').exists() and not (dest/'summary.json').exists():
            # Smallest complete official endpoint check; its reaction is reused by worker 0.
            smoke = [sys.executable,'-u',str(ROOT/'src/eval_uspto_mit_five_view_a6000.py'),'worker',
                     '--checkpoint',str(checkpoint(seed,arm)),'--manifest',str(PANEL),'--workers','640','--worker-index','0',
                     '--prompt-batch-size','1','--batch-mode','left-pad','--threads-per-worker','1','--output',str(dest/'worker_00.jsonl')]
            write(dest/'first_reaction_launch.json',{'command':smoke,'seed':seed,'arm':arm,'manifest_sha256':EXPECTED})
            print(json.dumps({'stage':'first_reaction_check','time':time.time()}),flush=True)
            with (dest/'first_reaction.log').open('a') as log:
                subprocess.run(smoke,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            first = jsonl(dest/'worker_00.jsonl')
            assert len(first)==1 and first[0]['reaction_identity']==panel_check()[0]['reaction_identity']
            write(dest/'first_reaction_stats.json',read(dest/'worker_00.stats.json'))
        if not (dest/'summary.json').exists():
            write(dest/'launch.json',{'command':command,'seed':seed,'arm':arm,'manifest_sha256':EXPECTED,
                  'environment':{k:v for k,v in env.items() if k.startswith('CHEMFM_') or k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','TOKENIZERS_PARALLELISM')}})
            print(json.dumps({'stage':'launch','seed':seed,'arm':arm,'time':time.time()}),flush=True)
            with (dest/'evaluate.log').open('a') as log:
                subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        proof = verify_cell(seed,arm)
        write(dest/'verified_provenance.json',proof)
        print(json.dumps({'stage':'verified_cell','seed':seed,'arm':arm,'time':time.time()}),flush=True)
    write(OUT/'all_evaluations_verified.json',[verify_cell(s,a) for s,a in CELLS])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=('preflight','verify-inputs','run','verify'))
    args = parser.parse_args()
    if args.stage == 'preflight': preflight()
    elif args.stage == 'verify-inputs': verify_inputs()
    elif args.stage == 'run': run()
    else:
        verify_inputs()
        write(OUT/'all_evaluations_verified.json',[verify_cell(s,a) for s,a in CELLS])
