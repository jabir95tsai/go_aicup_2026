"""plan.md step 6-7: one dev evaluation of each five-arm selected checkpoint, then paired bootstrap.

dev policy (7,200 games) is extracted once with the frozen HumanSL teacher + C vectors and kept in
RAM only (never written to /work).  1,419 games reuse the C4x monitor packs (``c4x/monitor.zip``);
the other 5,781 are replayed.  Every pack game is checked move-by-move against the move store and
the first game of each pack against a fresh replay.  Before dev, each arm's selected checkpoint is
re-scored on monitor (loaded from the policy ZIP) and must reproduce the training run's number.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd
import torch

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.metrics import K_BUCKETS
from go_aicup.paths import ROOT
from . import policy_zip
from .branches import BranchHead, gate, paired_difference
from .data import VECTORS, Cache
from .five_arms import CASES, build, evaluate_role
from .persistent_cache import C4X, encode_supplement, infer, validate_inputs, validate_policy
from .run import finalize_cache
from .train import atomic_save

PAIRS = [('R10', 'A0'), ('Z32', 'A0'), ('Z32', 'R10'), ('CNN-R10', 'R10'), ('CNN-Z32', 'Z32'),
         ('CNN-Z32', 'CNN-R10'), ('CNN-R10', 'A0'), ('CNN-Z32', 'A0')]


def teacher_identity():
    from go_aicup.features.humansl_torch import CKPT
    recorded = json.loads(policy_zip._Reader(policy_zip.ARCHIVE).read('policy/identity.json'))['sources']
    for path in (CKPT, VECTORS):
        name = str(path.relative_to(ROOT))
        if file_hash(path) != recorded[name]:
            raise ValueError(f'teacher input differs from the TRAIN/monitor cache: {name}')
    return {str(p.relative_to(ROOT)): recorded[str(p.relative_to(ROOT))] for p in (CKPT, VECTORS)}


def pack_inputs(dev, store):
    """Inputs of the reused dev games from c4x/monitor.zip, in dev row order."""
    from go_aicup.features.humansl_torch import _init_worker, encode_game, CKPT
    net_config = torch.load(CKPT, map_location='cpu', weights_only=False)['config']
    wanted = dev[dev.reuse_pack]
    expected = pd.concat([pd.read_parquet(C4X / f'monitor_hd{hd}.parquet') for hd in (0, 1)]).set_index('game_index')
    found = {}
    with zipfile.ZipFile(C4X / 'monitor.zip') as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f'ZIP CRC failure: monitor.zip:{bad}')
        for name in sorted(n for n in archive.namelist() if n.endswith('/meta.json')):
            prefix = name.removesuffix('meta.json')
            hd = int(re.search(r'/pack_hd([01])_', prefix)[1])
            details = json.loads(archive.read(name))
            table = pd.read_parquet(io.BytesIO(archive.read(prefix + 'games.parquet')))
            if details['komi'] != (.5 if hd else 6.5) or details['games'] != len(table):
                raise ValueError(f'incompatible pack manifest: {prefix}')
            reference = expected.loc[table.game_index]
            if any(not np.array_equal(table[c], reference[c]) for c in ('rank_id', 'player_color')):
                raise ValueError(f'pack metadata mismatch: {prefix}')
            arrays = {f: np.load(io.BytesIO(archive.read(prefix + f + '.npy')), allow_pickle=False)
                      for f in ('bits', 'globals', 'target', 'own', 'game')}
            lengths = store.meta.set_index('game_index').length.loc[table.game_index].to_numpy()
            bounds = np.r_[0, np.cumsum(lengths)]
            if (details['positions'] != bounds[-1]
                    or not np.array_equal(arrays['game'], np.repeat(np.arange(len(table)), lengths))):
                raise ValueError(f'pack position count or game order mismatch: {prefix}')
            parity = False
            for slot, g in enumerate(table.game_index):
                if g not in wanted.game_index.values:
                    continue
                row = wanted[wanted.game_index.eq(g)]
                if int(row.handicap.iloc[0]) != hd:
                    raise ValueError('pack handicap differs from selection')
                ids = np.arange(bounds[slot], bounds[slot + 1])
                bits, glob = arrays['bits'][ids], arrays['globals'][ids]
                target = arrays['target'][ids].astype(np.int64)
                side = (~arrays['own'][ids]).astype(np.uint8)
                raw = [store.game(int(g))]
                validate_inputs(row, bits, glob, target, side, raw)
                if not parity:
                    _init_worker(net_config, .5 if hd else 6.5)
                    bins = np.unpackbits(bits, axis=1)[:, :22 * 361].reshape(-1, 22, 19, 19)
                    for got, want in zip((bins, glob, target), encode_game(raw[0])):
                        np.testing.assert_array_equal(got, want, err_msg='monitor.zip/replay parity failed')
                    parity = True
                found[int(g)] = (bits, glob, target, side)
    if set(found) != set(wanted.game_index):
        raise ValueError('monitor.zip does not cover every reused dev game')
    return found


def extract(dev, config, device, *, conditions=10, model_factory=None, infer_policy=None, check_policy=None):
    """Return logp/q/target/side arrays in dev row order, plus extraction statistics."""
    from go_aicup.features.humansl_torch import _init_worker
    from go_aicup.tasks.strength.humansl_finetune import RankPolicy
    from go_aicup.tasks.strength.humansl_residual_b2.data import MoveStore
    store = MoveStore(ROOT / 'data/processed/boards/strength_v1')
    bounds = np.r_[0, np.cumsum(dev.length.to_numpy())].astype(np.int64)
    n = int(bounds[-1])
    arrays = dict(logp=np.empty((n, conditions, 362), np.float16), q=np.empty((n, conditions), np.float32),
                  target=np.empty(n, np.int16), side=np.empty(n, np.uint8))
    done = np.zeros(len(dev), dtype=bool)
    model = (model_factory(device) if model_factory else
             RankPolicy(device, torch.load(VECTORS, map_location=device, weights_only=False)['vectors']))
    model.requires_grad_(False).eval()
    forward = torch.compile(model, dynamic=False) if config['compile_teacher'] else model
    copy_stream = torch.cuda.Stream(device=device)
    stats = dict(max_mass_error=0., max_q_error=0., max_C_error=0., inference_gpu_ms=0.)
    started = time.monotonic()
    slot_of = pd.Series(np.arange(len(dev)), index=dev.game_index)

    def run(rows, bits, glob, target, side):
        values, checks, events = (infer_policy or infer)(forward, device, rows, bits, glob, target, side,
                                       config['inference_positions'], copy_stream)
        events[2].synchronize()
        metrics = (check_policy or validate_policy)(values, checks)
        for key in ('max_mass_error', 'max_q_error', 'max_C_error'):
            stats[key] = max(stats[key], metrics[key])
        stats['inference_gpu_ms'] += events[0].elapsed_time(events[1])
        start = 0
        for g, length in zip(rows.game_index, rows.length):
            slot = int(slot_of.loc[g])
            a, b = bounds[slot], bounds[slot + 1]
            for name in arrays:
                arrays[name][a:b] = values[name][start:start + length]
            done[slot] = True
            start += length

    tick = time.monotonic()
    reused = pack_inputs(dev, store)
    stats['pack_read_seconds'] = time.monotonic() - tick
    rows = dev[dev.reuse_pack].reset_index(drop=True)
    for start in range(0, len(rows), config['games_per_batch']):
        part = rows.iloc[start:start + config['games_per_batch']].reset_index(drop=True)
        items = [reused[int(g)] for g in part.game_index]
        run(part, *[np.concatenate([x[j] for x in items]) for j in range(4)])
    stats['reused_seconds'] = time.monotonic() - started
    print(json.dumps({'stage': 'reused_done', 'games': len(rows), 'positions': int(rows.length.sum()),
                      'seconds': round(stats['reused_seconds'], 1)}), flush=True)

    tick = time.monotonic()
    replay = dev[~dev.reuse_pack].reset_index(drop=True)
    batch = config['games_per_batch']
    with ProcessPoolExecutor(max_workers=config['encoder_workers'], mp_context=mp.get_context('spawn'),
                             initializer=_init_worker, initargs=(model.net.config,)) as pool:
        pending = []

        def submit(start):
            part = replay.iloc[start:start + batch].reset_index(drop=True)
            raw = [store.game(int(g)) for g in part.game_index]
            futures = [pool.submit(encode_supplement, (int(hd), item)) for hd, item in zip(part.handicap, raw)]
            pending.append((part, raw, futures))

        starts = list(range(0, len(replay), batch))
        for s in starts[:3]:
            submit(s)
        for index in range(len(starts)):
            part, raw, futures = pending.pop(0)
            parts = [f.result() for f in futures]
            if index + 3 < len(starts):
                submit(starts[index + 3])
            bits, glob, target = [np.concatenate([p[j] for p in parts]) for j in range(3)]
            side = np.concatenate([(colors != color).astype(np.uint8)
                                   for (_, colors), color in zip(raw, part.player_color)])
            validate_inputs(part, bits, glob, target, side, raw)
            run(part, bits, glob, target, side)
            if index % 20 == 0:
                done_rows = replay.iloc[:starts[index] + len(part)]
                print(json.dumps({'stage': 'replay', 'games': len(done_rows),
                                  'positions': int(done_rows.length.sum()),
                                  'seconds': round(time.monotonic() - tick, 1)}), flush=True)
    stats['replay_seconds'] = time.monotonic() - tick
    if not done.all():
        raise ValueError('dev extraction incomplete')
    stats.update(seconds=time.monotonic() - started, positions=n, games=len(dev),
                 reused_games=int(dev.reuse_pack.sum()), replayed_games=int((~dev.reuse_pack).sum()),
                 peak_gpu_gib=torch.cuda.max_memory_allocated(device) / 2**30)
    del model, forward
    torch.cuda.empty_cache()
    return bounds, arrays, stats


def make_cache(games, bounds, arrays, device):
    tensors = {k: torch.from_numpy(v) for k, v in arrays.items()}
    tensors['done'] = torch.ones(len(games), dtype=torch.bool)
    error = finalize_cache(games, bounds, tensors)
    for name in list(tensors):
        tensors[name] = tensors[name].to(device)
    return Cache(games.reset_index(drop=True), bounds, tensors, device), error


def load_arm(arm, run, initial, seed, device):
    model = build(arm, seed).to(device)
    head = BranchHead(model.width, initial).to(device)
    selected = torch.load(run / arm / 'model.pt', map_location=device, weights_only=False)
    model.load_state_dict(selected['model'])
    head.load_state_dict(selected['head'])
    return model.eval(), head.eval(), int(selected['step'])


def compare(frames, seed):
    ref = frames['A0']
    k, player = ref.k.to_numpy(), ref.player_id.to_numpy()
    out = {}
    for a, b in PAIRS:
        for frame in (frames[a], frames[b]):
            if not frame[['question_id', 'player_id', 'rank_id', 'k']].equals(ref[['question_id', 'player_id', 'rank_id', 'k']]):
                raise ValueError('unpaired dev questions')
        delta = frames[a].score.to_numpy() - frames[b].score.to_numpy()
        diff = paired_difference(frames[a].score.to_numpy(), frames[b].score.to_numpy(), k, player,
                                 np.random.default_rng(seed + 99))
        diff['by_k'] = {str(x): float(delta[k == x].mean()) for x in K_BUCKETS}
        diff['equal_k'] = float(np.mean(list(diff['by_k'].values())))
        diff['gate_pass'] = gate(diff)
        out[f'{a}-{b}'] = diff
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/strength/policy_five_arms_dev.json')
    parser.add_argument('--smoke-players', type=int, default=0,
                        help='only this many dev players per rank (engineering check, not a result)')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    run = ROOT / config['run']
    run_identity = json.loads((run / 'identity.json').read_text())
    for arm in CASES:
        marker = json.loads((run / arm / 'complete.json').read_text())
        if marker['identity'] != run_identity['identity'] or marker['status'] != 'complete':
            raise ValueError(f'arm not complete for this run: {arm}')
    train_config = json.loads((run / 'config.json').read_text())
    seed = train_config['seed']
    initial = torch.load(run / 'initial_head.pt', map_location='cpu', weights_only=False)

    games = policy_zip.selection('games')
    dev = games[games.role.eq('dev')].reset_index(drop=True)
    questions = policy_zip.selection('dev_questions')
    train_count = len(policy_zip.selection('train_questions'))
    if len(dev) != 7200 or len(questions) != 8000 or set(dev.player_id) & set(games.loc[~games.role.eq('dev'), 'player_id']):
        raise ValueError('unexpected dev selection or player overlap with TRAIN/monitor')
    if args.smoke_players:
        keep = dev.drop_duplicates('player_id').groupby('rank_id').head(args.smoke_players).player_id
        dev = dev[dev.player_id.isin(keep)].reset_index(drop=True)
        questions = questions[questions.player_id.isin(keep)].reset_index(drop=True)
    for item in questions.itertuples():
        if not set(item.games) <= set(dev.game_index):
            raise ValueError('dev question references a game outside the dev selection')

    record = dict(config=config, smoke_players=args.smoke_players, run_identity=run_identity['identity'],
                  teacher=teacher_identity(), monitor_zip_sha256=file_hash(C4X / 'monitor.zip'),
                  sources={n: file_hash(Path(__file__).with_name(n)) for n in
                           ('dev_eval.py', 'five_arms.py', 'persistent_cache.py', 'branches.py', 'data.py',
                            'run.py', 'train.py', 'model.py', 'policy_zip.py')},
                  plan_sha256=file_hash(ROOT / 'plan.md'),
                  extra_sources={name: file_hash(ROOT / name) for name in config.get('source_files', [])})
    identity = object_hash(record)
    out = run / ('dev_smoke' if args.smoke_players else 'dev') / os.environ.get('SLURM_JOB_ID', 'local')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'identity.json', dict(record, identity=identity))
    started = time.monotonic()

    # 1. Monitor reproduction of every selected checkpoint (guards the checkpoint/model loading path).
    tick = time.monotonic()
    loaded = policy_zip.load(roles=('monitor',), threads=config['zip_threads'])
    mon_games = loaded.games
    mon_bounds = np.r_[mon_games.start.to_numpy(), int(mon_games.start.iloc[-1] + mon_games.length.iloc[-1])]
    monitor_cache, _ = make_cache(mon_games, mon_bounds.astype(np.int64), loaded.arrays, device)
    del loaded
    mon_questions = {'monitor': policy_zip.selection('monitor_questions')}
    reproduction = {}
    for arm in CASES:
        model, head, step = load_arm(arm, run, initial, seed, device)
        report = evaluate_role(model, head, monitor_cache, mon_questions, 'monitor', config['position_chunk'], train_count)[0]
        saved = json.loads((run / arm / 'result.json').read_text())['roles']['monitor']['equal_k']
        reproduction[arm] = dict(step=step, recomputed=report['equal_k'], recorded=saved,
                                 diff=report['equal_k'] - saved)
        if abs(report['equal_k'] - saved) > config['monitor_tolerance']:
            raise ValueError(f'monitor reproduction failed: {arm} {reproduction[arm]}')
    del monitor_cache
    torch.cuda.empty_cache()
    print(json.dumps({'stage': 'monitor_reproduced', 'seconds': round(time.monotonic() - tick, 1),
                      'arms': reproduction}), flush=True)

    # 2. dev policy extraction (RAM only).
    bounds, arrays, extraction = extract(dev, config, device)
    print(json.dumps({'stage': 'extracted', **extraction}), flush=True)
    cache, c_error = make_cache(dev, bounds, arrays, device)
    del arrays
    extraction['cached_' + config.get('feature_name', 'C') + '_mean_max_abs_error'] = c_error

    # 3. One dev evaluation per selected checkpoint.
    role = {'dev': questions}
    frames, results = {}, {}
    for arm in CASES:
        model, head, step = load_arm(arm, run, initial, seed, device)
        report, frame, features, ids = evaluate_role(model, head, cache, role, 'dev', config['position_chunk'], train_count)
        directory = out / arm
        directory.mkdir(exist_ok=True)
        frame.to_parquet(directory / 'dev_predictions.parquet', index=False)
        np.savez(directory / 'dev_features.npz', game_index=cache.games.iloc[ids].game_index.to_numpy(), c=features)
        results[arm] = dict(selected_step=step, **report)
        write_json(directory / 'dev_result.json', results[arm])
        frames[arm] = frame
        print(json.dumps({'stage': 'dev', 'arm': arm, 'selected_step': step, 'equal_k': report['equal_k'],
                          'by_k': report['by_k'], 'ce': report['ce']}), flush=True)

    differences = compare(frames, seed)
    summary = dict(identity=identity, smoke=bool(args.smoke_players), dev_games=len(dev),
                   dev_players=int(dev.player_id.nunique()), dev_questions=len(questions),
                   monitor_reproduction=reproduction, extraction=extraction, arms=results,
                   differences=differences,
                   claims={'cnn_beats_set_residual': differences['CNN-R10-R10']['gate_pass'],
                           'cnn_beats_set_independent': differences['CNN-Z32-Z32']['gate_pass'],
                           'independent_beats_residual_set': differences['Z32-R10']['gate_pass'],
                           'independent_beats_residual_cnn': differences['CNN-Z32-CNN-R10']['gate_pass'],
                           'candidate_vs_A0': {a: differences[f'{a}-A0']['gate_pass']
                                               for a in ('R10', 'Z32', 'CNN-R10', 'CNN-Z32')}},
                   wall_seconds=time.monotonic() - started,
                   persistent_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))
    write_json(out / 'summary.json', summary)
    write_json(out / 'complete.json', {'identity': identity, 'status': 'complete'})
    print(json.dumps({'status': 'complete', 'arms': {a: r['equal_k'] for a, r in results.items()},
                      'differences': {p: {k: d[k] for k in ('equal_k', 'by_k', 'bootstrap_95', 'prob_positive', 'gate_pass')}
                                      for p, d in differences.items()},
                      'wall_seconds': summary['wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()
