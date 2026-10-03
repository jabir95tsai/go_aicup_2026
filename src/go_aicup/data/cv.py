"""The only place splits and synthetic questions are made (CLAUDE.md section 5).

Group key everywhere is `player_id`. Every generator asserts its own
disjointness invariant before returning.

Competition 2 (rank)
    `make_rank_folds`: players -> fold 0..4, or -1 for a fixed holdout
    (10% of players, the "second opinion" set). Stratified on
    rank x log(games) at the player level, so every fold has every rank and a
    similar game volume. `make_rank_questions` then draws test-shaped
    questions from a set of players: one player, one colour, k games.

Competition 1 (player id)
    `make_pid_splits`: replicate the pool structure -- 400 players, 100 black
    games each as the gallery, and one question per player built from
    *other* black games, k in {5,10,15,20} x 100. Only players with
    >= 120 black games qualify (1,281 of them, RESULTS A-001 A4), so the first
    floor(1281/400) = 3 splits are player-disjoint and the rest re-sample
    (their scores are NOT independent of the first three).

    Players in a pid split's pool must be excluded from any model that is
    evaluated on that split (the embedding would have memorised them).
    `pid_train_players(split)` gives the allowed training players.

Usage:
    python -m go_aicup.data.cv        # writes data/processed/cv/*.parquet and prints a summary
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from go_aicup.metrics import K_BUCKETS
from go_aicup.paths import PROCESSED

CV_DIR = PROCESSED / "cv"

SEED = 2026
N_FOLDS = 5
HOLDOUT = -1
HOLDOUT_FRAC = 0.10

POOL_SIZE = 400
GALLERY = 100
MAX_K = max(K_BUCKETS)
N_PID_SPLITS = 5


# --------------------------------------------------------------------------
# players
# --------------------------------------------------------------------------

def player_table(meta: pd.DataFrame) -> pd.DataFrame:
    color = meta["color"].astype(str)
    table = (
        meta.assign(is_black=(color == "B"), is_white=(color == "W"))
        .groupby("player_id", observed=True)
        .agg(rank_id=("rank_id", "first"), n_rank=("rank_id", "nunique"),
             n_games=("rank_id", "size"), n_black=("is_black", "sum"),
             n_white=("is_white", "sum"))
    )
    assert (table["n_rank"] == 1).all(), "rank must be a player-level attribute"
    return table.drop(columns="n_rank").reset_index()


# --------------------------------------------------------------------------
# competition 2
# --------------------------------------------------------------------------

def make_rank_folds(meta: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    players = player_table(meta)
    volume_bin = np.digitize(np.log10(players["n_games"]), [1.0, 1.5, 2.0, 2.5])
    strata = players["rank_id"].astype(int) * 10 + volume_bin
    # merge strata too small to split 5 ways into their rank-level stratum
    counts = strata.map(strata.value_counts())
    strata = pd.Series(np.where(counts >= 2 * N_FOLDS, strata,
                                players["rank_id"] * 10 + 9))
    # a merged remainder that is still too small joins its rank's largest bin
    counts = strata.map(strata.value_counts())
    largest = strata.groupby(players["rank_id"]).agg(lambda s: s.mode().iloc[0])
    strata = np.where(counts >= N_FOLDS, strata,
                      players["rank_id"].map(largest)).astype(int)

    rng = np.random.default_rng(seed)
    fold = np.full(len(players), HOLDOUT, dtype=np.int8)
    holdout = np.zeros(len(players), dtype=bool)
    for stratum in np.unique(strata):
        members = np.flatnonzero(strata == stratum)
        n_hold = int(round(len(members) * HOLDOUT_FRAC))
        holdout[rng.choice(members, size=n_hold, replace=False)] = True

    rest = np.flatnonzero(~holdout)
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    for f, (_, idx) in enumerate(skf.split(rest, strata[rest])):
        fold[rest[idx]] = f
    players["fold"] = fold
    assert_player_disjoint(players)
    return players


def assert_player_disjoint(players: pd.DataFrame) -> None:
    assert players["player_id"].is_unique, "a player appears in two folds"
    groups = players.groupby("fold")["player_id"].apply(set)
    folds = list(groups.index)
    for i, a in enumerate(folds):
        for b in folds[i + 1:]:
            assert not (groups[a] & groups[b]), f"player overlap between folds {a},{b}"


def game_folds(meta: pd.DataFrame, players: pd.DataFrame) -> np.ndarray:
    """Per-game fold id aligned with `meta` rows."""
    mapping = players.set_index("player_id")["fold"]
    return meta["player_id"].astype(str).map(mapping).to_numpy(np.int8)


def make_rank_questions(meta: pd.DataFrame, player_ids, n_questions: int,
                        seed: int = SEED, balance_rank: bool = True) -> pd.DataFrame:
    """Test-shaped questions: one player, one colour, k games, k in K_BUCKETS.

    Buckets are equal-sized. With `balance_rank`, each (k, rank) cell gets the
    same number of questions -- the training set is rank-balanced, so this is
    our best guess for the test; whether the test really is balanced is
    unknown until LB evidence (L4).

    Players are drawn uniformly among those that have >= k games in the chosen
    colour, so heavy players are not over-represented. The same player may
    appear in several questions (a fold holds ~1,400 players, we draw 10,000).
    """
    rng = np.random.default_rng(seed)
    subset = meta[meta["player_id"].astype(str).isin(set(map(str, player_ids)))]
    index = subset.index.to_numpy()
    by_player_color: dict[tuple[str, str], np.ndarray] = {}
    for (pid, color), rows in subset.groupby(
        [subset["player_id"].astype(str), subset["color"].astype(str)], observed=True
    ).indices.items():
        by_player_color[(pid, color)] = index[rows]
    rank_of = subset.groupby(subset["player_id"].astype(str), observed=True)[
        "rank_id"].first()

    per_bucket = n_questions // len(K_BUCKETS)
    ranks = np.sort(rank_of.unique())
    keys = sorted(by_player_color)
    eligible: dict[tuple[str, int, int], list] = {}
    for key in keys:
        rank = int(rank_of[key[0]])
        for k in K_BUCKETS:
            if len(by_player_color[key]) >= k:
                eligible.setdefault((key[1], k, rank), []).append(key)
                eligible.setdefault((key[1], k, -1), []).append(key)

    records = []
    for k in K_BUCKETS:
        if balance_rank:
            cells = np.repeat(ranks, per_bucket // len(ranks))
            cells = np.concatenate([cells, rng.choice(ranks, per_bucket - len(cells))])
        else:
            cells = np.full(per_bucket, -1)
        for rank in cells:
            color = "B" if rng.random() < 0.5 else "W"
            candidates = eligible.get((color, k, int(rank)))
            if not candidates:  # nobody of this rank has k games in this colour
                color = "W" if color == "B" else "B"
                candidates = eligible[(color, k, int(rank))]
            pid, color = candidates[rng.integers(len(candidates))]
            games = rng.choice(by_player_color[(pid, color)], size=k, replace=False)
            records.append({"player_id": pid, "rank_id": int(rank_of[pid]), "k": k,
                            "color": color, "games": np.sort(games).astype(np.int64)})
    questions = pd.DataFrame(records)
    questions.insert(0, "question_id", np.arange(len(questions)))
    for games, k in zip(questions["games"], questions["k"]):
        assert len(set(games.tolist())) == k
    return questions


# --------------------------------------------------------------------------
# competition 1
# --------------------------------------------------------------------------

def make_pid_splits(meta: pd.DataFrame, n_splits: int = N_PID_SPLITS,
                    seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (gallery, questions) for all splits.

    gallery:   split, player_id, game (one row per gallery game)
    questions: split, question_id, player_id, k, games (array of game rows)
    """
    rng = np.random.default_rng(seed)
    black = meta[meta["color"].astype(str) == "B"]
    black_rows = black.groupby(black["player_id"].astype(str), observed=True).indices
    black_index = black.index.to_numpy()
    eligible = np.array(sorted(p for p, r in black_rows.items()
                               if len(r) >= GALLERY + MAX_K))
    n_disjoint = len(eligible) // POOL_SIZE

    order = rng.permutation(eligible)
    galleries, questions = [], []
    for split in range(n_splits):
        if split < n_disjoint:
            pool = order[split * POOL_SIZE:(split + 1) * POOL_SIZE]
        else:
            pool = rng.choice(eligible, size=POOL_SIZE, replace=False)
        ks = rng.permutation(np.repeat(K_BUCKETS, POOL_SIZE // len(K_BUCKETS)))
        for qid, (pid, k) in enumerate(zip(pool, ks)):
            games = rng.permutation(black_index[black_rows[pid]])
            gallery, query = games[:GALLERY], games[GALLERY:GALLERY + k]
            assert not set(gallery.tolist()) & set(query.tolist())
            galleries.append(pd.DataFrame({"split": split, "player_id": pid,
                                           "game": np.sort(gallery)}))
            questions.append({"split": split, "question_id": qid, "player_id": pid,
                              "k": int(k), "games": np.sort(query).astype(np.int64)})
    gallery_df = pd.concat(galleries, ignore_index=True)
    question_df = pd.DataFrame(questions)
    assert_pid_disjoint(gallery_df, question_df)
    return gallery_df, question_df


def assert_pid_disjoint(gallery: pd.DataFrame, questions: pd.DataFrame) -> None:
    for split, group in questions.groupby("split"):
        query_games = set(np.concatenate(group["games"].to_list()).tolist())
        gallery_games = set(gallery.loc[gallery["split"] == split, "game"].tolist())
        assert not query_games & gallery_games, f"split {split}: query in gallery"
        assert group["player_id"].is_unique
        pool = set(gallery.loc[gallery["split"] == split, "player_id"])
        assert set(group["player_id"]) == pool, f"split {split}: answer not in pool"


def pid_train_players(meta: pd.DataFrame, gallery: pd.DataFrame, split: int) -> np.ndarray:
    held = set(gallery.loc[gallery["split"] == split, "player_id"])
    players = meta["player_id"].astype(str).unique()
    return np.array(sorted(p for p in players if p not in held))


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------

RANK_QUESTIONS_PER_FOLD = 10_000
RANK_QUESTIONS_HOLDOUT = 4_000


def build_all(meta: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    players = make_rank_folds(meta)
    out["rank_folds"] = players
    for fold in [*range(N_FOLDS), HOLDOUT]:
        ids = players.loc[players["fold"] == fold, "player_id"]
        n = RANK_QUESTIONS_HOLDOUT if fold == HOLDOUT else RANK_QUESTIONS_PER_FOLD
        name = "holdout" if fold == HOLDOUT else f"f{fold}"
        out[f"rank_questions_{name}"] = make_rank_questions(meta, ids, n, seed=SEED + fold)
    gallery, questions = make_pid_splits(meta)
    out["pid_gallery"] = gallery
    out["pid_questions"] = questions
    return out


def save(tables: dict[str, pd.DataFrame]) -> None:
    CV_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(CV_DIR / f"{name}.parquet", index=False)


def load(name: str) -> pd.DataFrame:
    table = pd.read_parquet(CV_DIR / f"{name}.parquet")
    if "games" in table:
        table["games"] = table["games"].map(lambda g: np.asarray(g, dtype=np.int64))
    return table


def summarize(meta: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> str:
    lines = []
    players = tables["rank_folds"]
    per_fold = players.pivot_table(index="rank_id", columns="fold", values="player_id",
                                   aggfunc="count")
    games = players.groupby("fold")["n_games"].sum()
    lines.append("rank folds: players per (rank x fold), fold -1 = holdout")
    lines.append(per_fold.to_string())
    lines.append("games per fold: " + ", ".join(f"{f}={n:,}" for f, n in games.items()))
    for name, table in tables.items():
        if name.startswith("rank_questions"):
            lines.append(f"{name}: {len(table):,} questions, "
                         f"{table['player_id'].nunique():,} distinct players, "
                         f"k={table['k'].value_counts().sort_index().to_dict()}, "
                         f"color={table['color'].value_counts().to_dict()}")
    q = tables["pid_questions"]
    players_black = tables["rank_folds"]["n_black"]
    lines.append(f"pid: {q['split'].nunique()} splits x {POOL_SIZE} players; "
                 f"eligible (>= {GALLERY + MAX_K} black games) = "
                 f"{int((players_black >= GALLERY + MAX_K).sum())}, "
                 f"distinct used across splits = "
                 f"{tables['pid_gallery']['player_id'].nunique()}")
    splits = q.groupby("split")["player_id"].apply(set)
    overlap = [[len(splits[a] & splits[b]) for b in splits.index] for a in splits.index]
    lines.append("pid split player overlap:\n" + pd.DataFrame(overlap).to_string())
    return "\n".join(lines)


def main() -> int:
    from go_aicup.data.store import GameStore

    store = GameStore("train")
    tables = build_all(store.meta)
    save(tables)
    print(summarize(store.meta, tables))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
