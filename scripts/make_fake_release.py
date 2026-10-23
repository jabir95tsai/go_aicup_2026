"""Fake official release files from validation data, for the T1 pipeline rehearsal.

Writes the six official CSVs (CLAUDE.md section 1 layout) plus answer keys:

    competition 2  Valid-U questions of the seed-2027 V2 split (A-024 model never
                   trained on these players); 100 per k for public, another 100
                   per k for private.
    competition 1  synthetic pools split 1 -> public, split 2 -> private
                   (400 players x 100 black gallery games; A-029 report splits).

SGF strings are copied byte-for-byte from the training CSVs, so the rehearsal
exercises the same parser path as 11/04. Question order is shuffled; IDs are
fresh. Answer keys keep the source question ids for the gate comparison.

    python scripts/make_fake_release.py [--out outputs/rehearsal]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.data.preprocess import MAX_SLOTS, RANKS, RELEASE_FILES
from go_aicup.paths import DATA, PROCESSED, ROOT

TRAIN_CSV = DATA / "raw" / "training"
CACHE_GAMES = PROCESSED / "boards" / "strength_v1" / "games.parquet"
V2_DIR = PROCESSED / "strength_v2" / "seed2027"
PID_DIR = PROCESSED / "identity"
PER_K = 100
SEED = 2027


def sgf_lookup(games: pd.DataFrame, needed: np.ndarray) -> dict[int, str]:
    """game_index -> raw sgf_content, reading each rank CSV once."""
    rows = games.loc[np.unique(needed), ["rank", "source_row"]]
    result = {}
    for rank, group in rows.groupby("rank", observed=True):
        frame = pd.read_csv(TRAIN_CSV / f"train_{rank}.csv", usecols=["sgf_content"], dtype=str)
        contents = frame["sgf_content"].to_numpy()
        for game_index, source_row in zip(group.index, group["source_row"]):
            result[int(game_index)] = contents[int(source_row)]
    return result


def wide(questions: pd.DataFrame, sgf: dict[int, str], colors) -> pd.DataFrame:
    """One official row per question: question_id, num_games, sgf_1..20, color_1..20."""
    records = []
    for question, color in zip(questions.itertuples(), colors):
        games = list(question.games)
        record = {"question_id": question.new_id, "num_games": len(games)}
        for slot in range(1, MAX_SLOTS + 1):
            record[f"sgf_{slot}"] = sgf[int(games[slot - 1])] if slot <= len(games) else ""
        for slot in range(1, MAX_SLOTS + 1):
            record[f"color_{slot}"] = color if slot <= len(games) else ""
        records.append(record)
    return pd.DataFrame(records)


def pick_balanced(questions: pd.DataFrame, rng, parts: int) -> list[pd.DataFrame]:
    """`parts` disjoint draws of PER_K questions per k, each shuffled."""
    chosen = [[] for _ in range(parts)]
    for _, group in questions.groupby("k"):
        order = rng.permutation(len(group))
        for part in range(parts):
            chosen[part].append(group.iloc[order[part * PER_K:(part + 1) * PER_K]])
    return [pd.concat(c).sample(frac=1.0, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
            for c in chosen]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "rehearsal")
    args = parser.parse_args()
    raw, answers = args.out / "raw", args.out / "answers"
    raw.mkdir(parents=True, exist_ok=True)
    answers.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    games = pd.read_parquet(CACHE_GAMES).set_index("game_index")

    # ---- competition 2 ----------------------------------------------------
    unseen = pd.read_parquet(V2_DIR / "strength_v2_valid_unseen_questions.parquet")
    rank_parts = dict(zip(("public", "private"), pick_balanced(unseen, rng, 2)))
    # ---- competition 1 ----------------------------------------------------
    gallery = pd.read_parquet(PID_DIR / "pid_gallery.parquet")
    pid_questions = pd.read_parquet(PID_DIR / "pid_questions.parquet")
    pid_parts = {"public": 1, "private": 2}

    needed = [np.concatenate(list(q["games"])) for q in rank_parts.values()]
    for split in pid_parts.values():
        needed.append(gallery.loc[gallery["split"] == split, "game"].to_numpy())
        needed.append(np.concatenate(list(pid_questions.loc[pid_questions["split"] == split, "games"])))
    sgf = sgf_lookup(games, np.concatenate(needed).astype(np.int64))

    for part, questions in rank_parts.items():
        questions = questions.assign(new_id=np.arange(1, len(questions) + 1))
        wide(questions, sgf, questions["color"]).to_csv(raw / RELEASE_FILES[f"rank_{part}"], index=False)
        questions.assign(rank=[RANKS[r] for r in questions["rank_id"]])[
            ["new_id", "rank", "rank_id", "k", "color", "question_id", "player_id"]].rename(
            columns={"new_id": "question_id", "question_id": "source_question_id"}).to_csv(
            answers / f"rank_{part}.csv", index=False)
        print(f"rank_{part}: {len(questions)} questions, k={questions['k'].value_counts().sort_index().to_dict()}")

    for part, split in pid_parts.items():
        pool = gallery[gallery["split"] == split]
        candidates = pd.DataFrame({"game_id": [f"c{int(g):07d}" for g in pool["game"]],
                                   "player_id": pool["player_id"].astype(str).to_numpy(),
                                   "sgf_content": [sgf[int(g)] for g in pool["game"]], "color": "B"})
        candidates.to_csv(raw / RELEASE_FILES[f"pool_{part}"], index=False)
        questions = pid_questions[pid_questions["split"] == split]
        questions = questions.sample(frac=1.0, random_state=SEED + split).reset_index(drop=True)
        questions = questions.assign(new_id=np.arange(1, len(questions) + 1))
        wide(questions, sgf, ["B"] * len(questions)).to_csv(raw / RELEASE_FILES[f"pid_{part}"], index=False)
        questions[["new_id", "player_id", "k", "question_id"]].rename(
            columns={"new_id": "question_id", "question_id": "source_question_id"}).to_csv(
            answers / f"pid_{part}.csv", index=False)
        print(f"pid_{part}: {len(candidates)} candidate games, {len(questions)} questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
