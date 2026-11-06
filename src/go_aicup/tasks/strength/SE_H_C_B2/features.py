"""Read verified legacy features and fill only missing frozen-model game rows."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
import torch

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.data.board_cache import BoardCache
from go_aicup.features import humansl as human
from go_aicup.tasks.strength.reproduce_paper.dataset import PaperDataset, pre_action_inputs
from go_aicup.tasks.strength.reproduce_paper_3.evaluate import infer_games
from go_aicup.tasks.strength.reproduce_paper_3.model import StrengthEstimator
from .common import A, S, H, C, checked_features, load_table, publish_table, state_hash, assert_frozen, writer_lock
from .data import load_prepared, questions, match_metadata

COLUMNS = {'se3': S, 'a': A, 'h': H, 'c': C}


def merge_features(parts, columns):
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=['game_index', *columns])
    frame = pd.concat(parts, ignore_index=True)
    duplicates = frame[frame.game_index.duplicated(False)]
    for _, rows in duplicates.groupby('game_index'):
        if not np.allclose(rows[columns], rows[columns].iloc[0].to_numpy()[None], atol=2e-6, rtol=0):
            raise ValueError('conflicting cached features for one game')
    frame = frame.drop_duplicates('game_index').reset_index(drop=True)
    checked_features(frame, columns)
    return frame


def legacy_h(directory, metadata):
    parts = []
    for shard in sorted(Path(directory).glob('shard_*')):
        if not (shard / 'meta.json').exists():
            continue
        info = json.loads((shard / 'meta.json').read_text())
        if (info['profiles'] != list(human.PROFILES) or info['rules'] != human.RULES
                or info['komi'] != human.KOMI):
            raise ValueError('legacy HumanSL feature recipe differs')
        match_metadata(pd.read_parquet(shard / 'games.parquet'), metadata)
        frame = human.shard_features(shard)
        if len(frame):
            if list(frame) != ['game_index', *H]:
                raise ValueError('legacy H columns differ')
            parts.append(frame)
    return merge_features(parts, H)


def legacy_c(directory, metadata):
    parts = []
    for shard in sorted(Path(directory).glob('shard_*')):
        if not (shard / 'meta.json').exists():
            continue
        info = json.loads((shard / 'meta.json').read_text())
        columns = [c.removeprefix('c_') for c in C]
        if info['columns'] != columns:
            raise ValueError('legacy C column order differs')
        games = pd.read_parquet(shard / 'games.parquet')
        match_metadata(games, metadata)
        values = np.load(shard / 'feature.npy', allow_pickle=False)
        done = np.load(shard / 'done.npy', allow_pickle=False)
        if values.shape != (len(games), 20) or done.shape != (len(games),) or done.dtype != np.bool_:
            raise ValueError('invalid legacy C shard')
        frame = pd.DataFrame(values[done], columns=C)
        frame.insert(0, 'game_index', games.game_index.to_numpy()[done])
        parts.append(frame)
    return merge_features(parts, C)


def legacy_a(run, metadata):
    parts = []
    for name in ('validation_unseen_category.npy', 'validation_seen_category.npy'):
        path = Path(run) / name
        if not path.exists():
            continue
        category = np.load(path, allow_pickle=False)
        if np.any(category['num_moves'] <= 0):
            raise ValueError('legacy A has zero-move games')
        ids = category['game_index'].astype(np.int64)
        if not np.isin(ids, metadata.game_index).all():
            raise ValueError('A game IDs do not belong to this cache')
        frame = pd.DataFrame(category['log_probability_sum'] / category['num_moves'][:, None], columns=A)
        frame.insert(0, 'game_index', ids)
        parts.append(frame)
    return merge_features(parts, A)


def legacy_se3(config, metadata, source):
    directory = Path(config['paths']['se3_evaluation'])
    manifest = json.loads((directory / 'manifest.json').read_text())
    if (manifest['model_sha256'] != source['models']['se3_model']
            or manifest['data']['games_sha256'] != source['dataset']):
        raise ValueError('SE3 validation cache source differs')
    expected_precision = 'fp32' if config['inference']['device'] == 'cpu' else 'bf16'
    if manifest['precision'] != expected_precision:
        return pd.DataFrame(columns=['game_index', *S])
    parts = []
    for path in sorted(directory.glob('games_rank*.parquet')):
        marker = json.loads(path.with_suffix('.json').read_text())
        if marker != dict(identity=object_hash(manifest), sha256=file_hash(path)):
            raise ValueError('SE3 validation shard checksum mismatch')
        frame = pd.read_parquet(path)
        if (frame.position_count.le(0).any() or
                not np.isin(frame.game_index, metadata.loc[metadata.split.eq('valid'), 'game_index']).all()):
            raise ValueError('SE3 cache must contain validation games')
        parts.append(frame[['game_index', *[f'logp_{i}' for i in range(10)]]].rename(
            columns={f'logp_{i}': S[i] for i in range(10)}))
    return merge_features(parts, S)


def context(config, release=None):
    prepared = load_prepared(config)
    if release is None:
        cache = Path(config['paths']['cache'])
        metadata = pd.read_parquet(Path(config['paths']['dataset']) / 'games.parquet')
        directory = Path(config['paths']['output']) / 'features/train'
    else:
        release = Path(release)
        cache = release / 'cache'
        metadata = pd.read_parquet(cache / 'games.parquet')
        directory = release / 'se3_hc_b2_features'
    identity = dict(source=prepared['identity']['source'], cache=file_hash(cache / 'manifest.json'),
                    metadata=file_hash(cache / 'games.parquet'), inference=config['inference'], scope='train' if release is None else 'release')
    return metadata, cache, directory, identity


def read_component(config, kind, *, release=None, ctx=None, include_extracted=True):
    metadata, cache, directory, identity = context(config, release) if ctx is None else ctx
    parts = []
    root = Path(config['paths']['human_root'])
    if release is None:
        if kind == 'se3':
            parts.append(legacy_se3(config, metadata, identity['source']))
        elif kind == 'a':
            parts.append(legacy_a(config['paths']['a_run'], metadata))
        elif kind == 'h':
            parts += [legacy_h(root / n, metadata) for n in ('humansl_q1000', 'humansl_qsel', 'humansl_vs_rest')]
        else:
            parts += [legacy_c(root / n, metadata) for n in ('humansl_rankvec_feat', 'humansl_rankvec_feat_vu')]
    else:
        if kind == 'h':
            parts.append(legacy_h(Path(release) / 'humansl', metadata))
        elif kind == 'c':
            parts.append(legacy_c(Path(release) / 'rankvec', metadata))
    for path in sorted((directory / kind).glob('*.parquet')) if include_extracted else ():
        if not path.with_suffix('.json').exists():
            continue
        frame = load_table(path, dict(**identity, component=kind))
        if kind == 'se3':
            if frame.position_count.le(0).any():
                raise ValueError('invalid SE3 effective position count')
            frame = frame.drop(columns='position_count')
        parts.append(frame)
    return checked_features(merge_features(parts, COLUMNS[kind]), COLUMNS[kind])


def assemble(config, arm, ids, *, release=None):
    ctx = context(config, release)
    ids = np.asarray(ids, np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError('duplicate requested game IDs')
    parts = []
    for kind in (arm, 'h', 'c'):
        table = read_component(config, kind, release=release, ctx=ctx)
        missing = sorted(set(ids) - set(table.index))
        if missing:
            raise ValueError(f'{kind}: {len(missing)} missing game features; run extract (first IDs: {missing[:5]})')
        parts.append(table.loc[ids])
    return pd.concat(parts, axis=1)


class ReleaseDataset:
    """Adapter using the exact SE pre-action encoder, without synthetic labels."""
    def __init__(self, cache):
        self.cache = BoardCache(cache, max_open_shards=8)
        self._indices = None

    def _ensure_open(self):
        if self._indices is not None:
            return
        games, plies, offsets = [], [], [0]
        for game in range(len(self.cache)):
            color = self.cache.player_colors[game]
            selected = np.flatnonzero(self.cache.colors(game) == color)
            selected = selected[selected >= 30]
            if len(selected):
                games.append(game); plies.extend(selected.tolist()); offsets.append(len(plies))
        self._indices = (np.asarray(games, np.int64), np.asarray(offsets, np.int64), np.asarray(plies, np.int64))

    def __getitems__(self, indices):
        self._ensure_open()
        games, offsets, plies = self._indices
        indices = np.asarray(indices, np.int64)
        selected_games = games[np.searchsorted(offsets, indices, side='right') - 1]
        planes, _ = pre_action_inputs(self.cache, selected_games, plies[indices])
        return [{'planes': p} for p in planes]

    def close(self):
        self.cache.close()


def compute_c(runtime, table, cache, batch=409, *, allow_failures=False):
    from go_aicup.features.humansl_rankvec import game_logp, _encode_or_none
    from go_aicup.features.humansl_torch import encode_game
    model, forward, vectors, pool = runtime
    games = human.cache_games(table.game_index, cache)
    encoded = list(pool.map(_encode_or_none if allow_failures else encode_game, games, chunksize=4))
    keep = [i for i, value in enumerate(encoded) if value is not None
            and (games[i][1] == table.player_color.iloc[i]).any()
            and (games[i][1] != table.player_color.iloc[i]).any()]
    if len(keep) != len(table) and not allow_failures:
        raise ValueError('C replay failed or one side has no moves')
    table = table.iloc[keep]
    games, encoded = [games[i] for i in keep], [encoded[i] for i in keep]
    if not len(table):
        return pd.DataFrame(columns=['game_index', *C])
    bins, globals_, played = (np.concatenate([v[i] for v in encoded]) for i in range(3))
    own = np.concatenate([color == row.player_color for (_, color), row in zip(games, table.itertuples())])
    spans = np.repeat(np.arange(len(table)), [len(g[0]) for g in games])
    slots = 2 * spans + (~own).astype(np.int64)
    sums = np.zeros((2 * len(table), 10), np.float64)
    counts = np.bincount(slots, minlength=2 * len(table))
    with torch.inference_mode():
        for start in range(0, len(played), batch):
            stop = min(start + batch, len(played))
            chunks = [a[start:stop] for a in (bins, globals_, played, own)]
            if stop - start < batch:
                chunks = [np.concatenate([a, np.repeat(a[-1:], batch - len(a), axis=0)]) for a in chunks]
            logp = game_logp(forward, vectors, *chunks, batch=batch * 10)[:stop-start].cpu().numpy()
            np.add.at(sums, slots[start:stop], logp)
    valid = (counts.reshape(-1, 2) > 0).all(1)
    frame = pd.DataFrame((sums / np.maximum(counts[:, None], 1)).reshape(-1, 20)[valid], columns=C)
    frame.insert(0, 'game_index', table.game_index.to_numpy()[valid])
    return frame


def extract(config, kind, scope='fit', *, release=None, shard=0, shards=1, limit=None, device=None):
    device = device or config['inference']['device']
    config = json.loads(json.dumps(config))
    config['inference']['device'] = device
    if kind in ('h', 'c') and device != 'cuda':
        raise ValueError('H/C extraction requires CUDA; CPU is for SE3 engineering checks')
    if not 0 <= shard < shards or (limit is not None and limit < 1):
        raise ValueError('invalid shard or limit')
    if kind not in ('se3', 'h', 'c'):
        raise ValueError('extract supports se3, h, c; A must use original V2 cache')
    ctx = context(config, release)
    metadata, cache, directory, identity = ctx
    existing = read_component(config, kind, release=release, ctx=ctx)
    if release is None:
        names = ('dev', 'select') if scope == 'dev' else ('dev', 'select', 'final') if scope == 'fit' else ('newgames',) if scope == 'newgames' else ('old_u',)
        ids = np.unique(np.concatenate([np.concatenate(questions(config, n).games) for n in names]))
    else:
        ids = metadata.game_index.to_numpy()
    pending = metadata[metadata.game_index.isin(ids)].sort_values('game_index')
    if limit is not None:
        baseline = read_component(config, kind, release=release, ctx=ctx, include_extracted=False)
        pending = pending[~pending.game_index.isin(baseline.index)]
        # Round-robin shuffled rank/color cells, before distributing over workers.
        groups = ['rank_id', 'player_color'] if release is None else ['player_color']
        pending = pending.sample(frac=1, random_state=config['seed'])
        pending = pending.assign(_n=pending.groupby(groups).cumcount()).sort_values('_n').head(limit).drop(columns='_n')
    pending = pending[(pending.game_index % shards == shard) & ~pending.game_index.isin(existing.index)]
    paths = {k: Path(v) for k, v in config['paths'].items()}
    target = directory / kind
    target.mkdir(parents=True, exist_ok=True)
    runtime, upstream, before, dataset = None, None, None, None
    start = time.perf_counter(); positions = completed = failed = 0
    if device == 'cuda':
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
    try:
        if len(pending) and kind == 'se3':
            payload = torch.load(paths['se3_model'], map_location='cpu', weights_only=False)
            se_config = payload['config']
            se_config = json.loads(json.dumps(se_config))
            se_config.update(device=device, precision='bf16' if device == 'cuda' else 'fp32')
            se_config['evaluation']['per_gpu_batch_size'] = config['inference']['batch_size']
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            upstream = StrengthEstimator(**se_config['model']).to(device)
            upstream.load_state_dict(payload['model']); upstream.eval(); upstream.requires_grad_(False)
            upstream.to(memory_format=torch.channels_last)
            before = state_hash(upstream)
            dataset = ReleaseDataset(cache) if release is not None else None
        elif len(pending) and kind == 'c':
            from go_aicup.features.humansl_rankvec import load_frozen
            from go_aicup.features.humansl_torch import _init_worker, CKPT
            if paths['human_checkpoint'] != CKPT:
                raise ValueError('rank-vector loader checkpoint path differs from config')
            upstream, _ = load_frozen(torch.device(device))
            upstream.eval(); upstream.requires_grad_(False); before = state_hash(upstream)
            vectors = torch.load(paths['vectors'], map_location=device, weights_only=False)['vectors'].detach()
            if vectors.shape != (10, 2, 384) or not torch.isfinite(vectors).all():
                raise ValueError('invalid frozen C vectors')
            workers = ProcessPoolExecutor(max_workers=config['inference']['workers'], mp_context=mp.get_context('spawn'),
                                          initializer=_init_worker, initargs=(upstream.config,))
            vectors_before = vectors.clone()
            runtime = (upstream, torch.compile(upstream, dynamic=False), vectors, workers)
        size = config['inference']['games_per_shard']
        for begin in range(0, len(pending), size):
            rows = pending.iloc[begin:begin + size]
            path = target / f'shard_{object_hash(rows.game_index.tolist())[:20]}.parquet'
            with writer_lock(target / f'worker_{shard:02d}'):
                if path.with_suffix('.json').exists():
                    frame = load_table(path, dict(**identity, component=kind))
                elif kind == 'se3':
                    parts = []
                    for split, group in rows.groupby('split' if release is None else lambda _: 'release'):
                        current = dataset if release is not None else PaperDataset(paths['dataset'], split=split,
                                    augment=False, cache_dir=cache, label_dir=None)
                        try:
                            current._ensure_open()
                            available = set(current._indices[0])
                            usable = group[group.game_index.isin(available)]
                            if release is None and len(usable) != len(group):
                                raise ValueError('prepared SE3 game missing')
                            part = infer_games(upstream, current, usable.game_index.to_numpy(), se_config, torch.device(device))
                            part = part.rename(columns={f'logp_{i}': S[i] for i in range(10)})
                            parts.append(part)
                        finally:
                            if release is None:
                                current.close()
                    frame = pd.concat(parts, ignore_index=True)
                elif kind == 'h':
                    with tempfile.TemporaryDirectory(dir=target) as temporary:
                        tmp = Path(temporary); rows.to_parquet(tmp / 'games.parquet', index=False)
                        human.run(human.cache_games(rows.game_index, cache), human.PROFILES, tmp,
                                  katago=paths['engine'], model=paths['human_model'], config=paths['analysis_config'],
                                  overrides=f'numAnalysisThreads=64,logFile={tmp / "katago.log"}',
                                  allow_failures=release is not None)
                        frame = human.shard_features(tmp)
                        if frame.empty:
                            frame = pd.DataFrame(columns=['game_index', *H])
                else:
                    frame = compute_c(runtime, rows, cache, allow_failures=release is not None)
                columns = COLUMNS[kind]
                checked_features(frame[['game_index', *columns]], columns)
                if not set(frame.game_index).issubset(set(rows.game_index)):
                    raise ValueError('inference returned unrequested games')
                if release is None and set(frame.game_index) != set(rows.game_index):
                    raise ValueError('incomplete fitting/evaluation feature shard')
                if upstream is not None:
                    assert_frozen(upstream, before)
                publish_table(path, frame, dict(**identity, component=kind))
                completed += len(frame); failed += len(rows) - len(frame)
                positions += int(frame.position_count.sum()) if kind == 'se3' else int(rows.loc[rows.game_index.isin(frame.game_index), 'length'].sum())
            print(json.dumps(dict(component=kind, completed=completed, pending=len(pending), failed=failed)), flush=True)
        if upstream is not None:
            assert_frozen(upstream, before)
        if runtime is not None and not torch.equal(runtime[2], vectors_before):
            raise ValueError('frozen C vectors changed')
        for key in ('se3_model',) if kind == 'se3' else ('human_checkpoint', 'vectors') if kind == 'c' else ('human_model', 'engine', 'analysis_config'):
            if file_hash(paths[key]) != identity['source']['models'][key]:
                raise ValueError(f'{key} changed during inference')
    finally:
        if dataset is not None:
            dataset.close()
        if runtime is not None:
            runtime[3].shutdown(wait=True, cancel_futures=True)
    seconds = time.perf_counter() - start
    result = dict(component=kind, games=completed, failed=failed, positions=positions, seconds=seconds,
                  positions_per_second=positions / seconds if seconds else 0,
                  allocated_peak_bytes=torch.cuda.max_memory_allocated() if device == 'cuda' else 0,
                  note='End-to-end wall time includes model loading/compile/I/O; not a warm steady-state rate.')
    write_json(target / f'timing_{shard:02d}.json', result)
    return result
