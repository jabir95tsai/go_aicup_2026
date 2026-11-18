from __future__ import annotations

import gc
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.data.cv import make_policy_residual_games, make_rank_questions
from go_aicup.paths import ROOT

VECTORS = ROOT / 'outputs/rank/humansl/cvec_v2/learn/vectors.pt'


def prepare(config, out):
    from scripts.strength import rank_train20_hc as t20, humansl_hd_filter as hf
    meta = t20.metadata().merge(pd.read_parquet(hf.CANONICAL, columns=['game_index', 'length']),
                               on='game_index', validate='one_to_one')
    halves_path = ROOT / 'outputs/rank/humansl/c4x/halves.parquet'
    dev_path = ROOT / 'outputs/rank/humansl/c4g/dev_games.parquet'
    games = make_policy_residual_games(meta, pd.read_parquet(halves_path), pd.read_parquet(dev_path),
                                      config['seed'], config['train_per_half_rank'],
                                      config['monitor_per_half_rank'])
    games['color'] = games.player_color.map({0: 'B', 1: 'W'})
    games.to_parquet(out / 'games.parquet', index=False)
    question_sets = {}
    table = games.set_index('game_index', drop=False)
    for offset, role in enumerate(('train', 'monitor', 'dev')):
        part = table[table.role.eq(role)]
        q = make_rank_questions(part, part.player_id.unique(), config[f'{role}_questions'],
                                seed=config['seed'] + 10 + offset)
        for item in q.itertuples():
            chosen = part.loc[list(item.games)]
            if (len(chosen) != item.k or chosen.player_id.nunique() != 1
                    or chosen.player_color.nunique() != 1 or chosen.rank_id.nunique() != 1
                    or not chosen.rank_id.eq(item.rank_id).all()):
                raise ValueError('invalid question membership')
        q.to_parquet(out / f'{role}_questions.parquet', index=False)
        question_sets[role] = q
    from go_aicup.features.humansl_torch import CKPT
    identities = {'config': config, 'halves_sha256': file_hash(halves_path),
                  'dev_sha256': file_hash(dev_path), 'vectors_sha256': file_hash(VECTORS),
                  'humansl_sha256': file_hash(CKPT), 'games_sha256': file_hash(out / 'games.parquet'),
                  'canonical_sha256': file_hash(hf.CANONICAL), 'handicap_sha256': file_hash(hf.HANDICAP),
                  'board_manifest_sha256': file_hash(ROOT / 'data/processed/boards/strength_v1/manifest.json'),
                  'sources': {str(Path(__file__).name): file_hash(Path(__file__))}}
    for filename in ('model.py', 'run.py', 'train.py'):
        identities['sources'][filename] = file_hash(Path(__file__).with_name(filename))
    identities['sources']['cv.py'] = file_hash(ROOT / 'src/go_aicup/data/cv.py')
    for relative in ('features/humansl.py', 'features/humansl_torch.py', 'features/humansl_rankvec.py',
                     'tasks/strength/humansl_finetune.py', 'metrics.py'):
        identities['sources'][relative] = file_hash(ROOT / 'src/go_aicup' / relative)
    identities['roles'] = {role: {'players': int(g.player_id.nunique()), 'games': len(g),
                                 'positions': int(g.length.sum()),
                                 'players_per_rank': g.groupby('rank_id').player_id.nunique().to_dict()}
                          for role, g in games.groupby('role')}
    # C vectors were trained before this experiment; disclose any pre-existing player overlap.
    learn_ids = []
    for path in VECTORS.parent.glob('learn_hd*.parquet'):
        learn_ids.extend(pd.read_parquet(path).game_index.tolist())
    seen = set(meta.loc[meta.game_index.isin(learn_ids), 'player_id'])
    identities['c_vector_learning_player_overlap'] = {
        role: len(set(g.player_id) & seen) for role, g in games.groupby('role')}
    identities['identity'] = object_hash(identities)
    write_json(out / 'identity.json', identities)
    return games, question_sets, identities


def allocate(games):
    bounds = np.r_[0, np.cumsum(games.length.to_numpy(), dtype=np.int64)]
    n = int(bounds[-1])
    needed = n * (10 * 362 * 2 + 10 * 4 + 2 + 1)
    stat = os.statvfs('/dev/shm')
    if stat.f_bavail * stat.f_frsize < needed * 1.15:
        raise RuntimeError(f'RAM tmpfs insufficient for {needed / 2**30:.2f} GiB shared cache')
    # Shared-memory tensors are RAM-backed tmpfs, never persistent policy files.
    arrays = {'logp': torch.empty(n, 10, 362, dtype=torch.float16).share_memory_(),
              'q': torch.empty(n, 10, dtype=torch.float32).share_memory_(),
              'target': torch.empty(n, dtype=torch.int16).share_memory_(),
              'side': torch.empty(n, dtype=torch.uint8).share_memory_(),
              'done': torch.zeros(len(games), dtype=torch.bool).share_memory_()}
    return arrays, bounds, needed


def extract_worker(rank, workers, games, bounds, arrays, config, out):
    from go_aicup.features.humansl import cache_games
    from go_aicup.features.humansl_torch import encode_game, _init_worker
    from go_aicup.tasks.strength.humansl_finetune import RankPolicy
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    model = RankPolicy(device, torch.load(VECTORS, map_location=device, weights_only=False)['vectors'])
    model.requires_grad_(False).eval()
    forward = torch.compile(model, dynamic=False) if config['compile_teacher'] else model
    started, positions, max_mass_error, max_q_error = time.monotonic(), 0, 0., 0.
    size = config['inference_positions']
    ranks = torch.arange(10, device=device).repeat(size)
    rows = np.arange(rank, len(games), workers)
    for hd in (0, 1):
        selected = rows[games.iloc[rows].handicap.to_numpy() == hd]
        if not len(selected):
            continue
        raw_games = cache_games(games.iloc[selected].game_index.to_numpy())
        context = mp.get_context('spawn')
        with context.Pool(config['encoder_workers'], initializer=_init_worker,
                          initargs=(model.net.config, .5 if hd else 6.5)) as pool:
            encoded = pool.imap(encode_game, raw_games, chunksize=1)
            for offset in range(0, len(selected), 16):
                current = selected[offset:offset + 16]
                parts = [next(encoded) for _ in current]
                ids, side_parts = [], []
                for row, (bins, glob, target), (_, colors) in zip(current, parts, raw_games[offset:offset + 16]):
                    if len(target) != games.iloc[row].length or len(target) != len(colors):
                        raise ValueError('encoder dropped positions')
                    ids.append(np.arange(bounds[row], bounds[row + 1]))
                    side_parts.append((colors != games.iloc[row].player_color).astype(np.int64))
                ids = np.concatenate(ids)
                bins = np.concatenate([p[0] for p in parts])
                glob = np.concatenate([p[1] for p in parts])
                targets = np.concatenate([p[2] for p in parts])
                sides = np.concatenate(side_parts)
                for start in range(0, len(ids), size):
                    take = np.arange(start, min(start + size, len(ids)))
                    actual = len(take)
                    if actual < size:
                        take = np.r_[take, np.repeat(take[-1], size - actual)]
                    x = torch.as_tensor(bins[take], device=device).float()
                    g = torch.as_tensor(glob[take], device=device)
                    s = torch.as_tensor(sides[take], device=device)
                    target = torch.as_tensor(targets[take], device=device)
                    with torch.no_grad():
                        logits = forward(x.repeat_interleave(10, 0), g.repeat_interleave(10, 0),
                                         ranks, s.repeat_interleave(10, 0)).reshape(size, 10, 362)
                        legal = (x[:, 0] * (1 - x[:, 1]) * (1 - x[:, 2]) * (1 - x[:, 6])).flatten(1).gt(0)
                        legal = torch.cat((legal, torch.ones_like(legal[:, :1])), 1)
                        if not legal.gather(1, target[:, None]).all():
                            raise ValueError('played target excluded by existing C mask')
                        lp = logits.masked_fill(~legal[:, None], -torch.inf).log_softmax(-1)
                        q = lp.gather(2, target[:, None, None].expand(-1, 10, 1)).squeeze(-1)
                        cached = lp[:actual].half()
                        if not torch.isfinite(q).all() or not torch.isfinite(cached[legal[:actual, None].expand_as(cached)]).all():
                            raise ValueError('nonfinite legal policy')
                        mass = (cached.float().exp().sum(-1) - 1).abs().max().item()
                        err = (cached.float().gather(2, target[:actual, None, None].expand(-1, 10, 1)).squeeze(-1)
                               - q[:actual]).abs().max().item()
                    max_mass_error, max_q_error = max(max_mass_error, mass), max(max_q_error, err)
                    dest = torch.as_tensor(ids[start:start + actual])
                    arrays['logp'][dest] = cached.cpu()
                    arrays['q'][dest] = q[:actual].cpu()
                    arrays['target'][dest] = target[:actual].short().cpu()
                    arrays['side'][dest] = s[:actual].byte().cpu()
                    positions += actual
                arrays['done'][current] = True
                if offset % 128 == 0:
                    print(json.dumps({'stage': 'extract', 'gpu': rank, 'hd': hd,
                                      'games': offset + len(current), 'positions': positions,
                                      'seconds': round(time.monotonic() - started, 1)}), flush=True)
    if max_mass_error > .01 or max_q_error > .05:
        raise ValueError('FP16 policy precision gate failed')
    report = {'rank': rank, 'positions': positions, 'seconds': time.monotonic() - started,
              'max_cached_mass_error': max_mass_error, 'max_cached_q_error': max_q_error,
              'peak_gpu_gib': torch.cuda.max_memory_allocated() / 2**30}
    write_json(out / f'extract_{rank}.json', report)
    del model, forward
    gc.collect()


class Cache:
    def __init__(self, games, bounds, arrays, device):
        self.games, self.bounds, self.arrays, self.device = games, bounds, arrays, device

    def chunks(self, rows, chunk):
        rows = np.asarray(rows, dtype=np.int64)
        ids = np.concatenate([np.arange(self.bounds[r], self.bounds[r + 1]) for r in rows])
        owner = np.repeat(np.arange(len(rows)), np.diff(self.bounds)[rows])
        for start in range(0, len(ids), chunk):
            # Gather where the cache lives: CPU RAM by default, or GPU when the whole cache was moved there.
            index = torch.as_tensor(ids[start:start + chunk], device=self.arrays['q'].device)
            q = self.arrays['q'][index].to(self.device)
            lp = self.arrays['logp'][index].to(self.device)
            target = self.arrays['target'][index].long().to(self.device)
            slot = torch.as_tensor(2 * owner[start:start + chunk], device=self.device)
            slot = slot + self.arrays['side'][index].long().to(self.device)
            yield lp, target, q, slot

    def counts(self, rows):
        if 'counts' in self.arrays:
            return self.arrays['counts'][rows].flatten().to(self.device)
        result = []
        for r in rows:
            side = self.arrays['side'][self.bounds[r]:self.bounds[r + 1]].numpy()
            result.append(np.bincount(side, minlength=2))
        values = torch.tensor(np.asarray(result).reshape(-1), device=self.device)
        if (values == 0).any():
            raise ValueError('empty own/opponent side')
        return values


def question_rows(games, q):
    lookup = pd.Series(np.arange(len(games)), index=games.game_index)
    return [lookup.loc[list(ids)].to_numpy() for ids in q.games]
