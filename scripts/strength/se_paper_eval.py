"""CPU paper-style rank pooling over existing SE scores; no new inference.

Retains designated-player, ply>=30, BF16 scores: not an all-move reproduction.
Run from the repository root with PYTHONPATH=src and the go Python environment.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from go_aicup.artifacts import file_hash
from go_aicup.metrics import NEAR
from go_aicup.tasks.strength.reproduce_paper.evaluate import nearest_centroid

OUT = Path('outputs/rank/paper_eval_20261103')
RUNS = {
    'SE_35k': Path('outputs/rank/reproduce_paper/se_seed2026'),
    'SE2_50k': Path('outputs/rank/reproduce_paper_2/se_seed2026_adamw70k/milestones/update_00050000'),
    'SE2_70k': Path('outputs/rank/reproduce_paper_2/se_seed2026_adamw70k'),
}
SEED, REPEATS = 2026, 500


def pooled_scores(sums, counts, draws):
    return sums[draws].sum(axis=1) / counts[draws].sum(axis=1)


def main():
    # Unequal-length games must contribute by positions, not by game means.
    assert np.allclose(pooled_scores(np.array([0., 18.]), np.array([1, 9]),
                                    np.array([[0, 1], [0, 0]])), [1.8, 0.])
    assert nearest_centroid([.5], np.arange(10)).tolist() == [0]
    OUT.mkdir(parents=True, exist_ok=False)
    source = {}
    prepared = Path('data/processed/reproduce_paper/seed2026')
    metadata = pd.read_parquet(prepared / 'games.parquet').set_index('game_index', verify_integrity=True)
    frames, reports = {}, {}
    for name, run in RUNS.items():
        evaluation = run / 'evaluation'
        report = json.loads((evaluation / 'report.json').read_text())
        assert report['status'] == 'completed'
        assert file_hash(run / 'model.pt') == report['identity']['model_sha256']
        frame = pd.read_parquet(evaluation / 'valid_games.parquet').sort_values('game_index').reset_index(drop=True)
        assert len(frame) == 200000 and frame.game_index.is_unique
        assert np.isfinite(frame[['score_sum', 'score_mean']]).all().all()
        assert (frame.position_count > 0).all()
        assert np.allclose(frame.score_sum / frame.position_count, frame.score_mean)
        assert (metadata.loc[frame.game_index, 'split'] == 'valid').all()
        assert np.array_equal(metadata.loc[frame.game_index, 'rank_id'], frame.rank_id)
        assert np.allclose(np.load(evaluation / 'centroids.npy'), report['centroids'])
        if frames:
            first = next(iter(frames.values()))
            assert frame[['game_index', 'rank_id', 'position_count']].equals(
                first[['game_index', 'rank_id', 'position_count']])
            assert report['identity']['dataset_id'] == next(iter(reports.values()))['identity']['dataset_id']
        frames[name], reports[name] = frame, report
        source[name] = {'run': str(run), 'model_sha256': report['identity']['model_sha256'],
                        'valid_games_sha256': file_hash(evaluation / 'valid_games.parquet'),
                        'report_sha256': file_hash(evaluation / 'report.json'),
                        'centroids_sha256': file_hash(evaluation / 'centroids.npy')}

    first = next(iter(frames.values()))
    pools, splits = {}, []
    for level in range(10):
        rows = np.flatnonzero(first.rank_id.to_numpy() == level)
        assert len(rows) == 20000
        order = np.random.default_rng(np.random.SeedSequence([SEED, level, 1])).permutation(rows)
        candidate, query = order[:100], order[100:1000]
        assert not np.intersect1d(candidate, query).size
        pools[level] = (rows, candidate, query)
        for role, indices in [('candidate', candidate), ('query', query)]:
            splits.extend({'rank_id': level, 'role': role, 'game_index': int(first.game_index.iloc[i])}
                          for i in indices)
    pd.DataFrame(splits).to_csv(OUT / 'candidate_query_split.csv', index=False)

    result = []
    candidate_centroids = {}
    for name, frame in frames.items():
        sums = frame.score_sum.to_numpy(np.float64)
        counts = frame.position_count.to_numpy(np.int64)
        candidate_centroids[name] = [float(sums[pools[r][1]].sum() / counts[pools[r][1]].sum())
                                     for r in range(10)]
        for mode_id, mode in enumerate(['pooled_training_centroid', 'pooled_candidate100_query900']):
            centroids = np.asarray(reports[name]['centroids'] if mode_id == 0 else candidate_centroids[name])
            for level in range(10):
                pool = pools[level][0 if mode_id == 0 else 2]
                for n in range(1, 101):
                    # Model-independent seeds pair every model on identical sampled games.
                    rng = np.random.default_rng(np.random.SeedSequence([SEED, mode_id, level, n, 2]))
                    draws = pool[rng.integers(len(pool), size=(REPEATS, n))]
                    scores = pooled_scores(sums, counts, draws)
                    predicted = nearest_centroid(scores, centroids)
                    error = np.abs(predicted - level)
                    exact = (error == 0).astype(float)
                    within = (error <= 1).astype(float)
                    official = exact + (error == 1) * NEAR
                    result.append(dict(model=name, protocol=mode, n=n, rank_id=level,
                                       trials=REPEATS, accuracy=exact.mean(), within1=within.mean(),
                                       official_score=official.mean()))
    ranks = pd.DataFrame(result)
    ranks.to_csv(OUT / 'per_rank.csv', index=False)
    summary = ranks.groupby(['model', 'protocol', 'n'], sort=False)[
        ['accuracy', 'within1', 'official_score']].mean().reset_index()
    # Conditional Monte Carlo SE: excludes uncertainty from fixed pools and candidate selection.
    ranks['mc_var'] = ranks.accuracy * (1 - ranks.accuracy) / REPEATS
    variances = ranks.groupby(['model', 'protocol', 'n'], sort=False).mc_var.sum().to_numpy() / 100
    summary['accuracy_mc_se'] = np.sqrt(variances)
    summary.to_csv(OUT / 'summary.csv', index=False)
    baseline = []
    for name, report in reports.items():
        for k in [5, 10, 15, 20]:
            metrics = report['by_k'][str(k)]
            baseline.append(dict(model=name, n=k, accuracy=metrics['acc'],
                                 within1=metrics['within1'], official_score=metrics['score']))
    pd.DataFrame(baseline).to_csv(OUT / 'same_player_baseline.csv', index=False)
    manifest = dict(seed=SEED, repeats_per_rank_n=REPEATS, n=list(range(1,101)),
        author_commit='d9498964e2ba84b5798c40e3f5b8a91a8791d66c',
        source=source, candidate_centroids=candidate_centroids,
        split_sha256=file_hash(OUT / 'candidate_query_split.csv'), script_sha256=file_hash(__file__),
        protocol='Uniform game sampling with replacement within rank; no player/color constraint; position-weighted scores; nearest centroid; 500 trials per rank per N; ranks averaged equally.',
        limitations=['Scores retain designated-player plies>=30, BF16, no D4; author uses all plies, random query color and both colors for candidate.',
                     'Ten Yugen ranks, not eleven Fox ranks; game-disjoint with overlapping players.',
                     'Candidate/query disjoint by game only; one fixed 100/900 split per rank.',
                     'Monte Carlo SE is conditional on fixed data and centroids; not population uncertainty.',
                     'Same-player baseline uses existing eligible-player questions, not a matched query population.'])
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    fig, axes = plt.subplots(1,2,figsize=(12,4),sharey=True)
    for axis, mode in zip(axes, ['pooled_training_centroid','pooled_candidate100_query900']):
        for name in RUNS:
            data=summary[(summary.model==name)&(summary.protocol==mode)]
            line,=axis.plot(data.n,data.accuracy,label=name)
            old=pd.DataFrame(baseline); old=old[old.model==name]
            axis.scatter(old.n,old.accuracy,marker='x',color=line.get_color())
        axis.set(xlabel='Number of games',ylabel='Exact accuracy',title=mode,ylim=(0,1))
        axis.grid(alpha=.2); axis.legend()
    fig.tight_layout(); fig.savefig(OUT/'accuracy.png',dpi=160); plt.close(fig)
    print(summary[summary.n.isin([5,10,15,20,100])].to_string(index=False))
    print('OUTPUT', OUT)


if __name__ == '__main__':
    main()
