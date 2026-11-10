"""Report humansl_residual_b2 teacher progress and an ETA from repeated snapshots.

Read-only on teacher shards (published done.npy + committed chunks). Each call appends one snapshot to
outputs/rank/humansl_residual_b2/teacher_progress.jsonl; the rate is measured
between the first and last snapshot of the current Slurm job set, so call it at
least twice a few minutes apart.

    python -m scripts.strength.hrb2_teacher_progress [--phase dev] [--config ...]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.tasks.strength.humansl_residual_b2.config import read_config

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / 'outputs' / 'rank' / 'humansl_residual_b2' / 'teacher_progress.jsonl'


def running_teacher_jobs() -> list[dict]:
    try:
        out = subprocess.run(['squeue', '--me', '--noheader', '--name=hrb2-teacher',
                              '--format=%i|%T|%M|%l|%N'], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = [line.split('|') for line in out.strip().splitlines() if line.strip()]
    return [dict(job=r[0], state=r[1], elapsed=r[2], limit=r[3], node=r[4]) for r in rows]


def _seconds(text: str) -> int:
    days, _, clock = text.rpartition('-')
    parts = [int(p) for p in clock.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0)
    return int(days or 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def snapshot(config: dict, phase: str) -> dict:
    prepared = Path(config['data']['prepared_dir'])
    if phase == 'release':
        prepared = prepared.parent / f'{prepared.name}_release'
    roles = pd.read_parquet(prepared / 'games.parquet', columns=['game_index', 'role'])
    wanted = roles.loc[roles.role.eq('P' if phase == 'release' else 'D'), 'game_index']
    teacher = Path(config['data']['teacher_dir'])
    shards = []
    for shard_dir in sorted(teacher.glob('shard_*')):
        index = pd.read_parquet(shard_dir / 'index.parquet', columns=['game_index', 'row', 'length'])
        index = index[index.game_index.isin(wanted)]
        # Published arrays plus fsynced chunks committed since the last publish (teacher.py writer).
        # Read from this (login) node: this is the external check that labels really reached /work.
        done = np.load(shard_dir / 'done.npy').astype(bool) if (shard_dir / 'done.npy').exists() else None
        if done is None:
            done = np.zeros(int(index.row.max()) + 1 if len(index) else 0, dtype=bool)
        for chunk in sorted((shard_dir / 'chunks').glob('chunk_*.npz')):
            try:
                with np.load(chunk) as data:
                    done[data['rows']] = True
            except (OSError, ValueError, KeyError):
                continue
        flags = np.asarray(done[index.row.to_numpy()], dtype=bool)
        positions = index.length.to_numpy() - 30
        metrics = shard_dir / 'metrics.json'
        status = json.loads(metrics.read_text()).get('status') if metrics.exists() else None
        shards.append(dict(shard=shard_dir.name, games=int(len(index)), done_games=int(flags.sum()),
                           positions=int(positions.sum()), done_positions=int(positions[flags].sum()),
                           status=status))
    return dict(time=time.time(), phase=phase, shards=shards,
                jobs=[j['job'] for j in running_teacher_jobs()])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/strength/humansl_residual_train.json')
    parser.add_argument('--phase', default='dev', choices=['dev', 'release'])
    parser.add_argument('--no-log', action='store_true', help='do not append this snapshot')
    args = parser.parse_args()
    now = snapshot(read_config(args.config), args.phase)

    history = []
    if LOG.exists():
        history = [json.loads(line) for line in LOG.read_text().splitlines() if line.strip()]
    if not args.no_log:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open('a') as handle:
            handle.write(json.dumps(now) + '\n')

    games = sum(s['games'] for s in now['shards']); done = sum(s['done_games'] for s in now['shards'])
    pos = sum(s['positions'] for s in now['shards']); done_pos = sum(s['done_positions'] for s in now['shards'])
    print(f"phase={now['phase']}  games {done:,}/{games:,} ({done / games:.1%})  "
          f"positions {done_pos:,}/{pos:,} ({done_pos / pos:.1%})")
    for s in now['shards']:
        print(f"  {s['shard']}: {s['done_games']:,}/{s['games']:,} games  "
              f"({s['done_games'] / max(s['games'], 1):.1%})  status={s['status']}")

    jobs = running_teacher_jobs()
    for j in jobs:
        print(f"  job {j['job']} {j['state']} {j['elapsed']} / {j['limit']} on {j['node']}")

    # Rate from the earliest snapshot that saw the same running jobs (same job set => same run).
    same = [h for h in history if h.get('phase') == now['phase'] and h.get('jobs') == now['jobs'] and h['jobs']]
    if not same:
        print('rate: need another snapshot while the same jobs run (re-run in a few minutes)')
        return
    first = same[0]
    dt = now['time'] - first['time']
    d_pos = done_pos - sum(s['done_positions'] for s in first['shards'])
    d_games = done - sum(s['done_games'] for s in first['shards'])
    if dt <= 0 or d_pos <= 0:
        print(f'rate: no progress over {dt / 60:.1f} min')
        return
    rate = d_pos / dt
    eta = (pos - done_pos) / rate
    print(f"rate over {dt / 60:.1f} min: {rate:,.0f} positions/s, {d_games / dt * 3600:,.0f} games/h "
          f"(includes imported SE3 positions, so the remaining-new-work rate may be lower)")
    print(f"ETA at this rate: {eta / 3600:.1f} h from now")
    if jobs:
        left = min(_seconds(j['limit']) - _seconds(j['elapsed']) for j in jobs if j['state'] == 'RUNNING')
        if eta > left:
            print(f"WARNING: wall-clock limit leaves {left / 3600:.1f} h; resubmit "
                  f"(resumes from done.npy) to finish")


if __name__ == '__main__':
    main()
