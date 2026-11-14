"""The only place splits and synthetic questions are made (CLAUDE.md section 5).

The original generators use player-disjoint splits. Strength V1 explicitly
uses the approved per-rank game split; its separate API records player
overlap and preserves game disjointness instead (STRENGTH_V1_PLAN.md).
Strength V2 returns to player folds and adds a seen-player game holdout
(`make_strength_v2_split`), approved 2026-10-19.

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

def make_strength_v1_split(meta: pd.DataFrame, seed: int = SEED, *,
                           expected_games_per_rank: int = 100_000,
                           train_games_per_rank: int = 80_000) -> pd.DataFrame:
    """Fixed game-level split; unique key is (rank, game_id), never game_id.

    Smaller counts must be explicitly supplied for development data. This
    function deliberately does not apply the legacy player-disjoint policy.
    """
    required = {"game_index", "rank", "rank_id", "game_id", "player_id", "color"}
    if not required.issubset(meta.columns):
        raise ValueError(f"missing metadata columns: {sorted(required - set(meta.columns))}")
    if not 0 < train_games_per_rank < expected_games_per_rank:
        raise ValueError("train count must be between zero and expected count")
    if meta.duplicated(["rank", "game_id"]).any() or not meta["game_index"].is_unique:
        raise ValueError("duplicate game keys or game_index")
    if set(meta["rank_id"].astype(int)) != set(range(10)):
        raise ValueError("Strength V1 requires all ten ranks")
    if (meta.groupby("player_id", observed=True)["rank_id"].nunique() > 1).any():
        raise ValueError("a player has multiple ranks")
    table = meta.copy().sort_values("game_index").reset_index(drop=True)
    split = np.full(len(table), "valid", dtype="U5")
    rng = np.random.default_rng(seed)
    for rank in range(10):
        rows = np.flatnonzero(table["rank_id"].to_numpy() == rank)
        if len(rows) != expected_games_per_rank:
            raise ValueError(f"rank {rank}: expected {expected_games_per_rank} games, found {len(rows)}")
        split[rng.permutation(rows)[:train_games_per_rank]] = "train"
    table["split"] = split
    train = set(table.loc[table["split"] == "train", "game_index"])
    valid = set(table.loc[table["split"] == "valid", "game_index"])
    assert not train.intersection(valid), "game overlap between V1 splits"
    return table


def strength_v1_overlap(table: pd.DataFrame) -> dict:
    train = set(table.loc[table["split"] == "train", "player_id"].astype(str))
    valid = set(table.loc[table["split"] == "valid", "player_id"].astype(str))
    overlap = len(train & valid)
    return {"train_players": len(train), "valid_players": len(valid),
            "shared_players": overlap,
            "valid_player_overlap_fraction": overlap / len(valid) if valid else 0.0}


def make_strength_v1_questions(table: pd.DataFrame, split: str,
                               n_questions: int = 10_000,
                               seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same-player/color questions from one game split, plus eligibility.

    Each k/rank cell has the same number of questions. No repeated games
    within a question; missing eligibility is an explicit error.
    """
    if split not in set(table["split"]) or n_questions < 40 or n_questions % 40:
        raise ValueError("split must exist in the table and n_questions be a positive multiple of 40")
    subset = table.loc[table["split"] == split].set_index("game_index", drop=False)
    eligibility = []
    for rank in range(10):
        ranked = subset.loc[subset["rank_id"] == rank]
        counts = ranked.groupby(["player_id", "color"], observed=True).size()
        for k in K_BUCKETS:
            eligible = counts[counts >= k]
            eligibility.append({"split": split, "rank_id": rank, "k": k,
                                "player_color_groups": len(counts),
                                "eligible_groups": len(eligible),
                                "eligible_players": len({key[0] for key in eligible.index})})
            if eligible.empty:
                raise ValueError(f"{split}: no eligible players for rank={rank}, k={k}")
    questions = make_rank_questions(subset, subset["player_id"].unique(),
                                     n_questions, seed=seed, balance_rank=True)
    allowed = set(subset.index)
    for row in questions.itertuples(index=False):
        games = np.asarray(row.games)
        chosen = subset.loc[games]
        assert len(np.unique(games)) == row.k and set(games).issubset(allowed)
        assert chosen["player_id"].nunique() == chosen["color"].nunique() == 1
        assert (chosen["rank_id"] == row.rank_id).all()
    return questions, pd.DataFrame(eligibility)


V2_SPLITS = ("train", "valid_seen", "valid_unseen", "excluded")


def make_strength_v2_split(meta: pd.DataFrame, seed: int, *,
                           seen_holdout_fraction: float = 0.1) -> pd.DataFrame:
    """Player-disjoint V2 split built on make_rank_folds(seed).

    fold 0 -> valid_unseen (players never trained on); fold -1 -> excluded
    (fixed second-opinion holdout, never touched); folds 1..4 -> train, except
    floor(n * fraction) random games of every (player, colour) group, which
    become valid_seen (players also present in train). Groups with fewer than
    1 / fraction games contribute no seen games, so valid_seen leans towards
    heavier players.
    """
    if not 0 < seen_holdout_fraction < 1:
        raise ValueError("seen_holdout_fraction must be between zero and one")
    if not meta["game_index"].is_unique:
        raise ValueError("duplicate game_index")
    table = meta.copy().sort_values("game_index").reset_index(drop=True)
    folds = make_rank_folds(table, seed)
    table["fold"] = game_folds(table, folds)
    split = np.where(table["fold"] == HOLDOUT, "excluded",
                     np.where(table["fold"] == 0, "valid_unseen", "train")).astype("U12")
    order = np.random.default_rng(seed).random(len(table))
    training = table.loc[split == "train"].assign(order=order[split == "train"])
    groups = training.groupby([training["player_id"].astype(str), training["color"].astype(str)],
                              observed=True)
    position = groups["order"].rank(method="first").to_numpy() - 1
    held = position < np.floor(groups["order"].transform("size").to_numpy() * seen_holdout_fraction)
    split[training.index[held]] = "valid_seen"
    table["split"] = split
    players = {name: set(table.loc[table["split"] == name, "player_id"].astype(str)) for name in V2_SPLITS}
    assert not players["train"] & players["valid_unseen"], "unseen players leaked into train"
    assert not players["excluded"] & (players["train"] | players["valid_unseen"]), "holdout leaked"
    assert players["valid_seen"] <= players["train"], "seen players must also be trained on"
    return table


def strength_v2_overlap(table: pd.DataFrame) -> dict:
    players = {name: set(table.loc[table["split"] == name, "player_id"].astype(str)) for name in V2_SPLITS}
    return {f"{name}_players": len(players[name]) for name in V2_SPLITS} | {
        "valid_unseen_overlap_fraction": len(players["train"] & players["valid_unseen"]) / len(players["valid_unseen"]),
        "valid_seen_overlap_fraction": len(players["train"] & players["valid_seen"]) / len(players["valid_seen"])}


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


def make_humansl_residual_roles(games: pd.DataFrame, original_folds: pd.DataFrame,
                                seed: int = 2027, *, release: bool = False) -> pd.DataFrame:
    """Keep original identities; only M selection/release four-fold mapping are new."""
    original = original_folds[['player_id', 'fold']].copy()
    original['player_id'] = original.player_id.astype(str)
    if not original.player_id.is_unique or not original.fold.isin([-1, 0, 1, 2, 3, 4]).all():
        raise ValueError('original folds must assign each player exactly once')
    players = games.groupby('player_id', sort=True).agg(rank_id=('rank_id', 'first'),
                                                        n_games=('rank_id', 'size')).reset_index()
    players = players.merge(original, on='player_id', how='left', validate='one_to_one')
    if players.fold.isna().any():
        raise ValueError('filtered players missing from original full-data folds')
    players['role'] = np.where(players.fold.eq(-1), 'R', np.where(players.fold.eq(0), 'U', 'D'))
    players['oof_fold'] = players.fold.astype(int)
    rng = np.random.default_rng(seed)
    if release:
        players.loc[players.fold.ne(-1), 'role'] = 'P'
    for rank in range(10):
        allowed = players.index[players.rank_id.eq(rank) & players.role.eq('P' if release else 'D')]
        if release and len(allowed) < 4:
            raise ValueError(f'release rank {rank} needs at least four players')
        volume = np.digitize(np.log10(players.loc[allowed, 'n_games']), [1, 1.5, 2, 2.5])
        bins = {int(b): allowed[volume == b].to_numpy() for b in np.unique(volume)}
        small = [v for v in bins.values() if len(v) < (8 if release else 10)]
        bins = [v for v in bins.values() if len(v) >= (8 if release else 10)]
        if small:
            merged = np.concatenate(small)
            if len(merged) < (8 if release else 10) and bins:
                bins[0] = np.concatenate([bins[0], merged])
            else:
                bins.append(merged)
        offset = 0
        for members in bins:
            members = rng.permutation(members)
            if release:
                players.loc[members, 'oof_fold'] = (np.arange(len(members)) + offset) % 4 + 1
                offset = (offset + len(members)) % 4
            else:
                count = int(round(len(members) * .1))
                players.loc[members[:count], 'role'] = 'M'
    return players


# --------------------------------------------------------------------------
# competition 1, added test set (2026-11-11): 600 candidates x 10 games of both
# colours, questions of k in {1, 2, 3, 5} games, several questions may share a player
# --------------------------------------------------------------------------

PID2_POOL = 600
PID2_GALLERY = 10
PID2_K = (1, 2, 3, 5)
PID2_QUESTIONS = 400
PID2_BANK = 40


def make_pid2_bank(meta: pd.DataFrame, handicap: np.ndarray, gallery: pd.DataFrame,
                   bank: int = PID2_BANK, seed: int = SEED) -> pd.DataFrame:
    """Game bank for the added-set bench: pool, player_id, game, order.

    Players are those of synthetic splits 0-2 (never seen by the GE2E encoder), cut
    into two disjoint pools of PID2_POOL (fewer when
    not enough players have `bank` such games). Each player gets `bank` random games of
    either colour with HD <= 1 (the official files have no HD >= 2 game); `order` is
    the random order `draw_pid2` permutes again per draw.
    """
    rng = np.random.default_rng(seed + 2)
    players = np.array(sorted(gallery.loc[gallery["split"].isin([0, 1, 2]), "player_id"].unique()))
    rows = meta.groupby(meta["player_id"].astype(str), observed=True).indices
    usable = {p: rows[p][handicap[rows[p]] <= 1] for p in players}
    players = rng.permutation([p for p in players if len(usable[p]) >= bank])
    size = min(PID2_POOL, len(players) // 2)      # a few players have mostly HD >= 2 games
    out = []
    for i, player in enumerate(players[:2 * size]):
        chosen = rng.choice(usable[player], size=bank, replace=False)
        out.append(pd.DataFrame({"pool": i // size, "player_id": player, "game": chosen,
                                 "order": np.arange(bank)}))
    return pd.concat(out, ignore_index=True)


def draw_pid2(bank: pd.DataFrame, pool: int, seed: int):
    """One bench draw -> (candidates, gallery game lists, questions frame).

    Every candidate keeps PID2_GALLERY gallery games; each of the PID2_QUESTIONS
    questions picks a candidate uniformly with replacement and takes k of that
    player's unused non-gallery games, so query and gallery games never overlap and
    two questions of one player share no game.
    """
    rng = np.random.default_rng([seed, pool])
    part = bank[bank["pool"] == pool]
    candidates = np.array(sorted(part["player_id"].unique()))
    games = {p: rng.permutation(g["game"].to_numpy()) for p, g in part.groupby("player_id")}
    used = {p: PID2_GALLERY for p in candidates}
    gallery = [np.sort(games[p][:PID2_GALLERY]) for p in candidates]
    ks = rng.permutation(np.repeat(PID2_K, PID2_QUESTIONS // len(PID2_K)))
    questions = []
    for qid, k in enumerate(ks):
        while True:
            player = candidates[rng.integers(len(candidates))]
            if used[player] + k <= len(games[player]):
                break
        query = games[player][used[player]:used[player] + k]
        used[player] += int(k)
        questions.append({"question_id": qid, "player_id": player, "k": int(k), "games": np.sort(query)})
    frame = pd.DataFrame(questions)
    assert not set(np.concatenate(gallery).tolist()) & set(np.concatenate(frame["games"].to_list()).tolist())
    return candidates, gallery, frame


def pid2_fixed_gallery(bank: pd.DataFrame, pool: int, gallery_seed: int):
    """(candidates, gallery game lists, remaining game lists): one fixed choice of the
    PID2_GALLERY gallery games per candidate, for encoders that train on the gallery
    (candidate adaptation). The remaining games are the only source of questions."""
    rng = np.random.default_rng([gallery_seed, pool, 11])
    part = bank[bank["pool"] == pool]
    candidates = np.array(sorted(part["player_id"].unique()))
    games = {p: rng.permutation(np.sort(g["game"].to_numpy())) for p, g in part.groupby("player_id")}
    gallery = [np.sort(games[p][:PID2_GALLERY]) for p in candidates]
    rest = [np.sort(games[p][PID2_GALLERY:]) for p in candidates]
    return candidates, gallery, rest


def draw_pid2_fixed(bank: pd.DataFrame, pool: int, gallery_seed: int, seed: int):
    """draw_pid2 with the gallery of pid2_fixed_gallery: every draw has the same gallery and
    draws its questions from the candidates' other games only."""
    candidates, gallery, rest = pid2_fixed_gallery(bank, pool, gallery_seed)
    rng = np.random.default_rng([seed, pool, gallery_seed, 12])
    games = [rng.permutation(g) for g in rest]
    used = np.zeros(len(candidates), dtype=np.int64)
    ks = rng.permutation(np.repeat(PID2_K, PID2_QUESTIONS // len(PID2_K)))
    questions = []
    for qid, k in enumerate(ks):
        while True:
            c = int(rng.integers(len(candidates)))
            if used[c] + k <= len(games[c]):
                break
        questions.append({"question_id": qid, "player_id": candidates[c], "k": int(k),
                          "games": np.sort(games[c][used[c]:used[c] + k])})
        used[c] += int(k)
    frame = pd.DataFrame(questions)
    asked = np.concatenate(frame["games"].to_list())
    assert len(asked) == len(set(asked.tolist())), "a game appears in two questions"
    assert not set(np.concatenate(gallery).tolist()) & set(asked.tolist()), "question game in the gallery"
    return candidates, gallery, frame
