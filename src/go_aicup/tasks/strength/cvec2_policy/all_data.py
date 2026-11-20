"""Read-only cvec2 verification and resumable CPU inputs for the eight-arm run."""

from dataclasses import dataclass
import fcntl
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from go_aicup.artifacts import file_hash, write_json
from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import PROCESSED, ROOT
from go_aicup.tasks.strength import humansl
from go_aicup.tasks.strength.policy_residual.epoch import epoch_batches
from go_aicup.tasks.strength.v2.aggregate import official_scores
from scripts.strength import humansl_hd_filter as hf, rank_prod_a0_fusion as prod
from scripts.strength.rank_more_players import PriorMatched, player_prior
from .played import H_COLUMNS

SEED = 2027
STAGE1 = ROOT / "outputs/rank/humansl/cvec2_policy/stage1/812279"
FOLDS = PROCESSED / "cv/strength_folds_seed2027.parquet"
BOARD = PROCESSED / "boards/strength_v1/games.parquet"
DRAW_SHA256 = "cd336f09ec163c517d391de6fda5d89fbd0b036bc12c8c3fba3f0621c4a52d2f"
QUESTIONS_PER_K, PER_K_BATCH = 20_000, 8
STAGE1_FILES = {
    "games.parquet", "E1.parquet", "E2.parquet", "features.npz",
    "result.json", "question_scores.parquet",
}
GROUPS = ("fit", "e2", "reserve")
COUNTS = {"fit": (149_615, 33_363_661), "e2": (15_460, 3_445_528),
          "reserve": (18_196, 4_057_717)}
BASELINE = {"5": .7022, "10": .7494, "15": .7762, "20": .7740, "equal_k": .7505}
NORMALIZED_ARMS = ("A/C10-logsoftmax", "A/H15-C10-logsoftmax")
HEAD_FIELDS = ("means", "scales", "weights", "biases")
PROJECTION = np.r_[H_COLUMNS, 30:52]
PREPARED_FILES = {
    *(f"{group}.parquet" for group in GROUPS), "E2.parquet", "baseline_scores.parquet",
    "baseline_fit.npz", "baseline_e2.npz", "baseline_fit_40.npy", "nominal_lengths.npy",
    "questions.npz", "initial_b2.npz",
}


@dataclass
class Data:
    groups: dict
    questions: pd.DataFrame
    scores: pd.DataFrame
    features: dict
    draws: dict
    batches: np.ndarray
    labels: np.ndarray
    k: np.ndarray
    prior: dict
    heads: dict
    identity: dict
    indices: list
    nominal_lengths: np.ndarray

    @property
    def games(self):
        return pd.concat([self.groups["fit"], self.groups["e2"]], ignore_index=True)

    @property
    def base(self):
        return self.features

    @property
    def expected(self):
        return self.features["A"][:, :50]


def _check_files(directory, files, expected):
    if set(files) != expected or any(file_hash(directory / name) != files[name] for name in expected):
        raise ValueError(f"artifact checksums failed in {directory}")


def _source_identity(stage1_out, fold_path):
    stage1 = json.loads((stage1_out / "complete.json").read_text())
    if stage1["stage"] != 1:
        raise ValueError("expected the cvec2 Stage 1 manifest")
    _check_files(stage1_out, stage1["files"], STAGE1_FILES)
    return {
        "stage1": str(stage1_out.resolve()),
        "stage1_files": stage1["files"],
        "folds": {"path": str(fold_path.resolve()), "sha256": file_hash(fold_path)},
        "board": {"path": str(BOARD.resolve()), "sha256": file_hash(BOARD)},
        "code": file_hash(Path(__file__)),
        "seed": SEED,
    }


def _source_groups(games, questions, fold_path):
    folds = pd.read_parquet(fold_path, columns=["player_id", "rank_id", "fold"]).set_index("player_id")
    is_fit = games.role.ne("U").to_numpy()
    if (not games.game_index.is_unique or len(games) != 184_698
            or not games.role.isin(["S", "T", "U"]).all()
            or not games.player_color.isin([0, 1]).all() or not folds.index.is_unique):
        raise ValueError("invalid Stage 1 game or fold metadata")
    matched = folds.loc[games.player_id]
    if (not np.array_equal(matched.rank_id, games.rank_id)
            or not matched.fold.iloc[is_fit].isin([1, 2, 3, 4]).all()
            or not matched.fold.iloc[~is_fit].eq(0).all()
            or set(games.loc[is_fit, "player_id"]) & set(games.loc[~is_fit, "player_id"])):
        raise ValueError("cvec2 fitting and Valid-U players do not match the fixed folds")

    metadata = ["rank_id", "game_id", "player_id", "player_color"]
    board = pd.read_parquet(BOARD, columns=["game_index", "length", *metadata]).set_index("game_index")
    indexed = board.loc[games.game_index]
    if any(not np.array_equal(indexed[col], games[col]) for col in metadata) or (indexed.length <= 0).any():
        raise ValueError("Stage 1 game indices differ from the board replay metadata")
    games = games.assign(length=indexed.length.to_numpy(dtype=np.int64))
    fit = games.loc[is_fit]
    valid = games.loc[~is_fit & games.handicap.le(1)]
    if not questions.question_id.is_unique or questions[["k", "rank_id"]].value_counts().to_dict() != {
        (k, rank): 50 for k in K_BUCKETS for rank in range(10)
    }:
        raise ValueError("wrong fixed E2 question counts")
    if any(len(np.unique(draw)) != k or len(draw) != k for k, draw in zip(questions.k, questions.games)):
        raise ValueError("E2 questions require k distinct games")
    e2_ids = np.concatenate(questions.games.to_numpy())
    e2_metadata = valid.set_index("game_index").loc[e2_ids]
    for game_column, question_column in [("player_id", "player_id"), ("rank_id", "rank_id"),
                                          ("player_color", "color")]:
        if not np.array_equal(e2_metadata[game_column], np.repeat(questions[question_column], questions.k)):
            raise ValueError("E2 games differ in player, rank or color")
    is_e2 = valid.game_index.isin(e2_ids)
    groups = {"fit": fit, "e2": valid.loc[is_e2], "reserve": valid.loc[~is_e2]}
    if (fit.player_id.nunique() != 5728 or valid.player_id.nunique() != 1433
            or not fit.handicap.isin([0, 1]).all()
            or any((len(games), int(games.length.sum())) != COUNTS[name] for name, games in groups.items())):
        raise ValueError("wrong cvec2 game or position counts")
    return groups


def _training_questions(fit, prior):
    draws, digest = {}, hashlib.sha256()
    game_index = fit.game_index.to_numpy()
    for k, rows, labels in PriorMatched(prior).synthetic_rows(fit, fit.rank_id.to_numpy(), SEED):
        draws[k] = (game_index[rows], labels)
        for value in ([k, len(rows)], labels, draws[k][0]):
            digest.update(np.ascontiguousarray(value, dtype="<i8").tobytes())
    if (tuple(draws) != K_BUCKETS or any(len(labels) != QUESTIONS_PER_K for _, labels in draws.values())
            or digest.hexdigest() != DRAW_SHA256):
        raise ValueError(f"seed {SEED} production B2 draw mismatch: {digest.hexdigest()}")
    labels = np.concatenate([draws[k][1] for k in K_BUCKETS])
    k = np.repeat(K_BUCKETS, QUESTIONS_PER_K)
    batches = np.asarray(epoch_batches(pd.DataFrame({"k": k}), SEED, 1, PER_K_BATCH))
    return draws, labels, k, batches


def _head_arrays(fitted, dtype):
    scalers, models = zip(*(fitted.per_k[k] for k in K_BUCKETS))
    if any(not np.array_equal(model.classes_, np.arange(10)) for model in models):
        raise ValueError("the initial B2 requires all ten ranks")
    return {
        "means": np.stack([scaler.mean_ for scaler in scalers]).astype(dtype),
        "scales": np.stack([scaler.scale_ for scaler in scalers]).astype(dtype),
        "weights": np.stack([model.coef_ for model in models]).astype(dtype),
        "biases": np.stack([model.intercept_ for model in models]).astype(dtype),
    }


def preflight(stage1_out=STAGE1, fold_path=FOLDS):
    """Verify the original pools and reproduce the 2,000 saved E2 scores without writing files."""
    started = time.monotonic()
    identity = _source_identity(stage1_out, fold_path)
    games = pd.read_parquet(stage1_out / "games.parquet")
    questions = pd.read_parquet(stage1_out / "E2.parquet")
    groups = _source_groups(games, questions, fold_path)
    fit, e2 = groups["fit"], groups["e2"]
    report = json.loads((stage1_out / "result.json").read_text())
    parity = report["parity"]
    if (not parity["pass"] or not parity["local_matches"] or parity["export_changed"] != 0
            or parity["export_questions"] != 800 or parity["local"]["E2"] != BASELINE
            or report["per_seed"]["E2"][str(SEED)]["A"] != BASELINE):
        raise ValueError("Stage 1 cvec2 parity did not reproduce the fixed E2 baselines")
    scores = pd.read_parquet(stage1_out / "question_scores.parquet")
    scores = scores.loc[scores.split.eq("E2") & scores.seed.eq(SEED)].reset_index(drop=True)
    keys = ["question_id", "player_id", "rank_id", "k"]
    if not scores[keys].equals(questions[keys]):
        raise ValueError("seed 2027 score rows do not match the fixed E2 question order")

    prior = player_prior(fit.loc[fit.role.eq("S")])
    draws, labels, k, batches = _training_questions(fit, prior)
    flags = np.column_stack([games.handicap.eq(1) & games.player_color.eq(color) for color in (0, 1)])
    with np.load(stage1_out / "features.npz", allow_pickle=False) as archive:
        raw = archive["A"]
    if (raw.shape != (len(games), 52) or not np.isfinite(raw).all()
            or not np.array_equal(raw[:, -2:], flags)):
        raise ValueError("wrong Stage 1 A feature layout or HD1 flags")
    old_seed = humansl.SEED
    try:
        with threadpool_limits(limits=1):
            fitted = prod._fit(fit, raw[fit.index], prior, SEED)
            probabilities = fitted.probabilities(raw[e2.index], e2.game_index, questions)
    finally:
        humansl.SEED = old_seed
    values = official_scores(score_rank(probabilities), questions.rank_id)
    if not np.array_equal(values, scores["A"]) or hf.by_k(values, questions.k.to_numpy()) != BASELINE:
        raise ValueError("A E2 scores did not reproduce exactly")
    scores = scores[[*keys, "split", "seed", "A"]]
    rows = np.r_[fit.index, e2.index]
    nominal_lengths = np.r_[fit.length.to_numpy(), e2.length.to_numpy()]
    groups["fit"] = fit.reset_index(drop=True)
    groups["e2"] = e2.set_axis(np.arange(len(fit), len(rows)))
    identity |= {"counts": COUNTS, "baseline": BASELINE, "draw_sha256": DRAW_SHA256,
                 "A_features": "52-column A with separate H9 compatibility projection",
                 "preflight_seconds": time.monotonic() - started}
    return Data(groups, questions, scores, {"A": raw[rows].astype(np.float32)}, draws, batches,
                labels, k, prior, {"A": _head_arrays(fitted, np.float32)}, identity, [], nominal_lengths)


def _selected(data, smoke_steps):
    batches = data.batches[:smoke_steps]
    source_questions = np.unique(batches)
    if smoke_steps is None:
        return data.groups, data.questions, source_questions

    # Keep training batches fixed and add only the draws needed to fit a ten-class CPU head per k.
    groups = data.groups.copy()
    questions = data.questions.groupby("k", sort=False).head(1)
    fit_ids = np.unique(np.concatenate([
        rows[source_questions[data.k[source_questions] == k] - bucket * QUESTIONS_PER_K].ravel()
        for bucket, (k, (rows, _)) in enumerate(data.draws.items())
    ]))
    added = []
    for bucket, (k, (rows, ranks)) in enumerate(data.draws.items()):
        present = data.labels[source_questions[data.k[source_questions] == k]]
        for missing in np.setdiff1d(np.arange(10), present):
            candidates = np.flatnonzero(ranks == missing)
            best = candidates[np.isin(rows[candidates], fit_ids).sum(axis=1).argmax()]
            added.append(bucket * QUESTIONS_PER_K + best)
            fit_ids = np.union1d(fit_ids, rows[best])
    groups["fit"] = groups["fit"].loc[groups["fit"].game_index.isin(fit_ids)]
    e2_ids = np.concatenate(questions.games.to_numpy())
    groups["e2"] = groups["e2"].loc[groups["e2"].game_index.isin(e2_ids)]
    groups["reserve"] = groups["reserve"].iloc[:0]
    return groups, questions, np.union1d(source_questions, np.asarray(added, dtype=np.int64))


def prepare(out, stage1_out=STAGE1, fold_path=FOLDS, smoke_steps=None):
    """Publish immutable CPU inputs. Smoke keeps unchanged full-pool B2 fits and whole batches."""
    if smoke_steps is not None and not 1 <= smoke_steps <= QUESTIONS_PER_K // PER_K_BATCH:
        raise ValueError("smoke_steps must select complete batches from the fixed epoch")
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (out / "complete.json").is_file():
            completed = load(out)
            identity = _source_identity(stage1_out, fold_path) | {"smoke_steps": smoke_steps}
            if any(completed.identity.get(key) != value for key, value in identity.items()):
                raise ValueError("prepared CPU source or smoke configuration changed")
            return completed

        data = preflight(stage1_out, fold_path)
        groups, questions, selected = _selected(data, smoke_steps)
        with TemporaryDirectory(prefix=".prepare-", dir=out) as temporary:
            stage = Path(temporary)
            for group, games in groups.items():
                games.to_parquet(stage / f"{group}.parquet", index=False)
            questions.to_parquet(stage / "E2.parquet", index=False)
            data.scores.loc[questions.index].to_parquet(stage / "baseline_scores.parquet", index=False)
            raw = data.features["A"]
            for group in ("fit", "e2"):
                np.savez(stage / f"baseline_{group}.npz", A=raw[groups[group].index])
            np.save(stage / "baseline_fit_40.npy", raw[np.ix_(groups["fit"].index, PROJECTION)], allow_pickle=False)
            np.save(stage / "nominal_lengths.npy", data.nominal_lengths, allow_pickle=False)
            draw_arrays = {f"{name}_{k}": value for k, (rows, ranks) in data.draws.items()
                           for name, value in [("games", rows), ("ranks", ranks), ("prior", data.prior[k])]}
            np.savez(stage / "questions.npz", batches=data.batches, selected=selected, **draw_arrays)
            np.savez(stage / "initial_b2.npz",
                     **{f"{arm}_{field}": value for arm, head in data.heads.items() for field, value in head.items()})
            files = {name: file_hash(stage / name) for name in PREPARED_FILES}
            for name in files:
                (stage / name).replace(out / name)
        fit_ranks = {str(size): np.bincount(data.labels[selected][data.k[selected] == size],
                                           minlength=10).tolist() for size in K_BUCKETS}
        identity = data.identity | {"smoke_steps": smoke_steps, "files": files, "format": 2,
                                    "normalization_fit_ranks": fit_ranks}
        write_json(out / "complete.json", identity)
        return load(out)


def load(out):
    """Load validated CPU inputs in the exact teacher game order."""
    identity = json.loads((out / "complete.json").read_text())
    if identity["format"] != 2 or identity["draw_sha256"] != DRAW_SHA256:
        raise ValueError("invalid cvec2 CPU manifest")
    _check_files(out, identity["files"], PREPARED_FILES)
    groups = {name: pd.read_parquet(out / f"{name}.parquet") for name in GROUPS}
    questions = pd.read_parquet(out / "E2.parquet")
    scores = pd.read_parquet(out / "baseline_scores.parquet")
    with np.load(out / "baseline_fit.npz", allow_pickle=False) as fit, \
            np.load(out / "baseline_e2.npz", allow_pickle=False) as e2:
        features = {"A": np.concatenate([fit["A"], e2["A"]])}
    with np.load(out / "initial_b2.npz", allow_pickle=False) as archive:
        heads = {"A": {field: archive[f"A_{field}"] for field in HEAD_FIELDS}}
    with np.load(out / "questions.npz", allow_pickle=False) as archive:
        draws = {k: (archive[f"games_{k}"], archive[f"ranks_{k}"]) for k in K_BUCKETS}
        prior = {k: archive[f"prior_{k}"] for k in K_BUCKETS}
        steps = identity["smoke_steps"]
        source_batches, selected = archive["batches"][:steps], archive["selected"]

    batches = np.searchsorted(selected, source_batches)
    if not np.array_equal(selected[batches], source_batches):
        raise ValueError("an optimizer batch references an unselected training question")
    fit_index = pd.Index(groups["fit"].game_index)
    indices, k, labels = [], [], []
    for bucket, (size, (rows, ranks)) in enumerate(draws.items()):
        bucket_rows = selected[selected // QUESTIONS_PER_K == bucket] % QUESTIONS_PER_K
        game_rows = rows[bucket_rows]
        cache_rows = fit_index.get_indexer(game_rows.ravel()).reshape(-1, size)
        if np.any(cache_rows < 0):
            raise ValueError("an optimizer batch references a game outside its fitting pool")
        indices.extend(cache_rows)
        labels.append(ranks[bucket_rows])
        k.append(np.full(len(bucket_rows), size))
    nominal_lengths = np.load(out / "nominal_lengths.npy", allow_pickle=False)
    return Data(groups, questions, scores, features, draws, batches, np.concatenate(labels),
                np.concatenate(k), prior, heads, identity, indices, nominal_lengths)


class _FixedQuestions(PriorMatched):
    """Use the production B2 fitter with the saved, ordered question draws."""

    def __init__(self, prior, rows):
        super().__init__(prior)
        self.rows = rows

    def synthetic_rows(self, games, y, seed):
        return iter(self.rows)


def fit_normalized(data, offsets):
    """Apply the shared teacher offsets and fit both controls on the fixed fit draws."""
    raw = data.features["A"]
    if set(offsets) != {"H", "C"} or any(
        value.shape != (len(raw), 2) or not np.isfinite(value).all() for value in offsets.values()
    ):
        raise ValueError("incomplete or invalid per-side H15/C10 teacher offsets")
    c_normalized = raw.copy()
    c_normalized[:, 30:50] -= np.repeat(offsets["C"], 10, axis=1)
    hc_normalized = c_normalized.copy()
    hc_normalized[:, :30] -= np.repeat(offsets["H"], 15, axis=1)

    draws = [(k, np.stack([data.indices[i] for i in np.flatnonzero(data.k == k)]), data.labels[data.k == k])
             for k in K_BUCKETS]
    fit = data.groups["fit"]
    result = {}
    with threadpool_limits(limits=1):
        for arm, values in zip(NORMALIZED_ARMS, (c_normalized, hc_normalized), strict=True):
            fitted = _FixedQuestions(data.prior, draws).fit(fit, pd.DataFrame(), values[:len(fit)].astype(np.float64))
            result[arm] = values, _head_arrays(fitted, np.float64)
    return result
