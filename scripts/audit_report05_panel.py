"""Read-only panel provenance audit; JSON evidence is printed to stdout."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def sha(b):
    return hashlib.sha256(b).hexdigest()
def compare(a, b):
    return {'ordered_equal': a == b, 'set_equal': set(a) == set(b),
            'overlap': len(set(a) & set(b)),
            'first_mismatch_index_zero_based': next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)) if len(a) != len(b) else None)}

parent = ROOT / 'data/clm_jepa_uspto_mit_stp_confirmation/untouched_1280.jsonl'
lines = parent.read_bytes().splitlines(keepends=True)
assert len(lines) == 1280
panels = {name: [json.loads(x) for x in part] for name, part in [('first', lines[:640]), ('second', lines[640:])]}
output = {'parent_sha256': sha(parent.read_bytes()), 'slice_sha256': {'first': sha(b''.join(lines[:640])), 'second': sha(b''.join(lines[640:]))}, 'files': {}}
baseline = None
hits = {}
for seed in (2027, 3163):
    for arm in ('native', 'decoder_projected'):
        path = ROOT / f'runs/decoder_projected/confirmation/seed_{seed}/{arm}/evaluation_l40_8w/predictions.jsonl'
        rows = [json.loads(x) for x in path.read_bytes().splitlines()]
        ids = [r['reaction_identity'] for r in rows]
        groups = [r['official_group_index'] for r in rows]
        if baseline is None:
            baseline = ids
        summary = json.loads(path.with_name('summary.json').read_text())
        log = [json.loads(x) for x in path.with_name('evaluate.log').read_text().splitlines() if x.startswith('{')]
        item = {'path': str(path.relative_to(ROOT)), 'sha256': sha(path.read_bytes()), 'rows': len(rows), 'unique_identities': len(set(ids)), 'same_order_as_2027_native': ids == baseline,
                'ordered_identity_sha256': sha(('\n'.join(ids)+'\n').encode()),
                'manifest': summary['manifest'], 'manifest_sha256': summary['manifest_sha256'],
                'log_manifest_evidence': [{k: x.get(k) for k in ('manifest', 'manifest_sha256', 'checkpoint', 'predictions')} for x in log if 'manifest' in x],
                'panels': {name: {'reaction_identity': compare(ids, [r['reaction_identity'] for r in panel]), 'official_group_index': compare(groups, [r['official_group_index'] for r in panel]), 'first_identity_actual_expected': [ids[0], panel[0]['reaction_identity']], 'first_group_actual_expected': [groups[0], panel[0]['official_group_index']]} for name, panel in panels.items()},
                'top_counts': {k: sum(r['target'] in r['ranked_candidates'][:k] for r in rows) for k in (1,3,5,10)},
                'view_valid_count': sum(bool(c) for r in rows for view in r['canonical_candidates_by_view'] for c in view),
                'view_candidate_count': sum(len(view) for r in rows for view in r['canonical_candidates_by_view']),
                'ranked_valid_count': sum(bool(c) for r in rows for c in r['ranked_candidates']),
                'ranked_candidate_count': sum(len(r['ranked_candidates']) for r in rows)}
        output['files'][f'{seed}/{arm}'] = item
        hits[seed,arm] = [int(r['target'] == r['ranked_candidates'][0]) for r in rows]
output['paired'] = {}
for seed in (2027,3163):
    d = [p-n for n,p in zip(hits[seed,'native'],hits[seed,'decoder_projected'])]
    output['paired'][seed] = {'projected_only': d.count(1), 'native_only': d.count(-1), 'ties': d.count(0), 'effect_pp': 100*sum(d)/len(d)}
output['conclusion'] = 'ACTUAL_PANEL = ' + ('FIRST_640' if all(x['panels']['first']['reaction_identity']['ordered_equal'] for x in output['files'].values()) else 'SECOND_640' if all(x['panels']['second']['reaction_identity']['ordered_equal'] for x in output['files'].values()) else 'OTHER/MIXED')
print(json.dumps(output, indent=2))
