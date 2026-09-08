"""Independent fail-fast check of Report 05 primary counts from saved candidates."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {2027: (25, 23, 8, 10), 3163: (31, 22, 3, 12)}

for seed, expected in EXPECTED.items():
    paired = {}
    for arm in ('native', 'decoder_projected'):
        path = ROOT / f'runs/decoder_projected/confirmation/seed_{seed}/{arm}/evaluation_l40_8w/predictions.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 640
        assert len({r['reaction_identity'] for r in rows}) == 640
        paired[arm] = rows
        counts = [sum(r['target'] in r['ranked_candidates'][:k] for r in rows) for k in (1, 3, 5, 10)]
        print(seed, arm, 'top-1/3/5/10 counts:', counts, flush=True)
        primary_expected = expected[0 if arm == 'native' else 1]
        if counts[0] != primary_expected:
            raise SystemExit(f'HARD STOP: {seed} {arm}: raw top-1={counts[0]}, Report 05={primary_expected}')
    native, projected = paired['native'], paired['decoder_projected']
    assert [r['reaction_identity'] for r in native] == [r['reaction_identity'] for r in projected]
    differences = [int(b['target'] == b['ranked_candidates'][0]) - int(a['target'] == a['ranked_candidates'][0]) for a, b in zip(native, projected)]
    wins, losses = differences.count(1), differences.count(-1)
    print(seed, 'projected-only/native-only/ties/effect-pp:', wins, losses, differences.count(0), 100 * sum(differences) / 640, flush=True)
    if (wins, losses) != expected[2:]:
        raise SystemExit(f'HARD STOP: discordances {(wins, losses)} != {expected[2:]}')
