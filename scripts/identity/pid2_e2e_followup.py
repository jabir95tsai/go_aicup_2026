"""After the two E1 development jobs: stop, or E0+B and a single sealed evaluation."""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/pid/e2e_adam"
BASE = "base=/home/jason1122/pid2_seqagg_cache:outputs/pid/pid2_seqagg/head_B/best.pt"


def select_arm(reports):
    scores = {}
    for name, report in reports.items():
        if report['first_draw'] != 100 or report['base'] != 'base':
            raise ValueError('LR selection must use development draws only')
        rows = {r['arm']: r for r in report['tables']['pool 0 | fused']}
        row = rows[name]
        if row['draws'] != 20 or not math.isfinite(row['d_all']):
            raise ValueError('incomplete development comparison')
        scores[name] = row['d_all']
    # Prefer the smaller LR on an exact tie.
    best = max(scores, key=lambda name: (scores[name], name == 'lr1e5'))
    return best if scores[best] > 0 else None, scores


def run(module, *args):
    subprocess.run([sys.executable, '-m', module, *map(str, args)], cwd=ROOT, check=True)


def main():
    reports = {arm: json.loads((OUT / f'dev_{arm}/report.json').read_text()) for arm in ('lr3e5', 'lr1e5')}
    chosen, scores = select_arm(reports)
    selection = {'chosen': chosen, 'development_deltas': scores, 'status': 'selected' if chosen else 'stopped_no_gain'}
    (OUT / 'selection.json').write_text(json.dumps(selection, indent=2))
    print(json.dumps(selection), flush=True)
    if chosen is None:
        return
    if shutil.disk_usage('/dev/shm').free < 27 * 2**30:
        raise RuntimeError('E0 requires 27 GiB free node-local temporary memory')
    config = json.loads((OUT / f'{chosen}.json').read_text())
    config['e2e']['arm'] = 'encoder'
    run_name = f'pid_e2e_adam_e0_{chosen}'
    config['paths']['output_dir'] = f'outputs/pid/ge2e/{run_name}'
    config_file = OUT / f'e0_{chosen}.json'
    if config_file.exists() and json.loads(config_file.read_text()) != config:
        raise ValueError('refusing to change E0 config')
    config_file.write_text(json.dumps(config, indent=2))
    run('go_aicup.tasks.identity.train_e2e', '--config', config_file)
    cache = ROOT / f'data/processed/identity/e2e_adam/e0_{chosen}'
    run('scripts.identity.pid2_seqagg', 'extract', '--run', run_name, '--cache', cache,
        '--parts', 'bench', '--bench-run', 'pid_ge2e_v4_d4_hd_9600')
    # The 24 GiB training cache is disposable; avoid exhausting the shared /work quota.
    with tempfile.TemporaryDirectory(prefix='pid_e2e_e0_', dir='/dev/shm') as scratch:
        run('scripts.identity.pid2_seqagg', 'extract', '--run', run_name, '--cache', scratch,
            '--parts', 'train', '--bench-run', 'pid_ge2e_v4_d4_hd_9600')
        run('scripts.identity.pid2_seqagg', 'train', '--arm', 'B', '--cache', scratch,
            '--steps', '20000', '--seed', '2026', '--tag', f'_e2e_e0_{chosen}', '--budget-hours', '0.75')
    e1 = f'{chosen}=data/processed/identity/e2e_adam/{chosen}:outputs/pid/ge2e/pid_e2e_adam_{chosen}/head.pt'
    e0 = f'e0b={cache}:outputs/pid/pid2_seqagg/head_B_e2e_e0_{chosen}/best.pt'
    run('scripts.identity.pid2_seqagg', 'compare', '--arms', BASE, e1, e0,
        '--base', 'base', '--pools', '0', '1', '--first', '600', '--gate', '--minimum-gain', '0.015',
        '--gated', chosen, 'e0b', '--out', OUT / 'sealed')
    # The same sealed predictions also give the predeclared E1 vs E0+B attribution contrast.
    import pandas as pd
    from scripts.identity.pid2_diag_d123a import paired
    frame = pd.read_csv(OUT / 'sealed/per_draw.csv')
    contrasts = {}
    for pool in (0, 1):
        part = frame[(frame.pool == pool) & (frame.what == 'fused') & frame.arm.isin([chosen, 'e0b'])]
        contrasts[str(pool)] = paired(part, 'e0b').to_dict(orient='records')
    (OUT / 'sealed/e1_vs_e0b.json').write_text(json.dumps(contrasts, indent=2))
    selection['status'] = 'sealed_evaluation_complete'
    (OUT / 'selection.json').write_text(json.dumps(selection, indent=2))


if __name__ == '__main__':
    main()
