"""Competition-2 rank model on HumanSL + A-024 per-game features (RESULTS A-040, A-041).

Features per game: A-024 per-game mean log-prob (10, `a024_<rank>`) followed by
the HumanSL mean log-likelihood of the labelled player's and the opponent's
moves for each profile (`own_<profile>`, `opp_<profile>`; features/humansl.py).

Question aggregation methods (k = the question's game count, nearest bucket):
  B0  per-game LR; logit = T_k * mean_g(log posterior)
  B1  per-game LR; logit = T_k * W mean_g(x) + b       (the prior is not scaled)
  B2  one LR per k on synthetic questions (k games of one fitting player,
      features averaged over the games); no temperature
T_k (B0, B1) are fitted on the fitting questions with two-fold, player-grouped
out-of-fold values. Probabilities are decoded with metrics.score_rank.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from go_aicup.metrics import K_BUCKETS
from go_aicup.tasks.strength.v2.aggregate import TEMPERATURE_GRID

SEED, SYNTHETIC = 2027, 20000
METHODS = ("B0", "B1", "B2")


def k_bucket(num_games: int) -> int:
    return min(K_BUCKETS, key=lambda bucket: (abs(bucket - int(num_games)), bucket))


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


class GameLR:
    """Per-game multinomial LR on standardised features."""

    def __init__(self, x, y):
        self.scaler = StandardScaler().fit(x)
        self.model = LogisticRegression(C=1.0, max_iter=5000).fit(self.scaler.transform(x), y)

    def linear(self, x) -> np.ndarray:
        return self.scaler.transform(x) @ self.model.coef_.T

    def log_posterior(self, x) -> np.ndarray:
        return self.model.predict_log_proba(self.scaler.transform(x))


def question_mean(values: np.ndarray, game_index, questions: pd.DataFrame) -> np.ndarray:
    """Per question, the mean of its games' rows (games referenced by game_index)."""
    position = pd.Series(np.arange(len(game_index)), index=np.asarray(game_index))
    return np.stack([values[position.loc[list(g)].to_numpy()].mean(axis=0) for g in questions["games"]])


def player_folds(games: pd.DataFrame) -> np.ndarray:
    players = np.sort(games["player_id"].astype(str).unique())
    half = set(np.random.default_rng(SEED + 1).permutation(players)[:len(players) // 2])
    return games["player_id"].astype(str).isin(half).to_numpy()


class RankModel:
    def __init__(self, method: str, C: float = 1.0):
        """C: inverse L2 strength of the B2 per-k LRs (A-052); B0/B1 keep C=1."""
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.method = method
        self.C = C

    # ---------------------------------------------------------------- fitting
    def fit(self, games: pd.DataFrame, questions: pd.DataFrame, x: np.ndarray) -> "RankModel":
        """games: game_index, player_id, rank_id rows aligned with x; questions:
        games (lists of game_index), k, rank_id, player_id (B0/B1 temperatures)."""
        y = games["rank_id"].to_numpy()
        if self.method == "B2":
            self._fit_synthetic(games, x, y)
            return self
        self.game = GameLR(x, y)
        self.temps = self._fit_temperatures(games, questions, x, y)
        return self

    def _fit_temperatures(self, games, questions, x, y) -> dict:
        index, fold = games["game_index"].to_numpy(), player_folds(games)
        logits, offset = np.zeros((len(questions), 10)), np.zeros((len(questions), 10))
        q_players = questions["player_id"].astype(str)
        for side in (True, False):
            model = GameLR(x[fold != side], y[fold != side])
            q_mask = q_players.isin(set(games.loc[fold == side, "player_id"].astype(str))).to_numpy()
            values = model.log_posterior(x) if self.method == "B0" else model.linear(x)
            logits[q_mask] = question_mean(values, index, questions[q_mask])
            offset[q_mask] = 0.0 if self.method == "B0" else model.model.intercept_
        labels = questions["rank_id"].to_numpy()
        temps = {}
        for k in K_BUCKETS:
            mask = questions["k"].to_numpy() == k
            losses = []
            for t in TEMPERATURE_GRID:
                z = t * logits[mask] + offset[mask]
                z -= z.max(axis=1, keepdims=True)
                losses.append(float((np.log(np.exp(z).sum(axis=1)) - z[np.arange(mask.sum()), labels[mask]]).mean()))
            best = int(np.argmin(losses))
            if best in (0, len(TEMPERATURE_GRID) - 1):
                raise ValueError(f"k={k}: temperature on the grid edge")
            temps[k] = float(TEMPERATURE_GRID[best])
        return temps

    def _fit_synthetic(self, games, x, y) -> None:
        rng = np.random.default_rng(SEED)
        by_player = list(games.groupby(games["player_id"].astype(str)).indices.values())
        self.per_k = {}
        for k in K_BUCKETS:
            players = [rows for rows in by_player if len(rows) >= k]
            picks = rng.integers(0, len(players), SYNTHETIC)
            features = np.stack([x[rng.choice(players[p], k, replace=False)].mean(axis=0) for p in picks])
            labels = np.array([y[players[p][0]] for p in picks])
            scaler = StandardScaler().fit(features)
            model = LogisticRegression(C=getattr(self, "C", 1.0), max_iter=5000)
            self.per_k[k] = (scaler, model.fit(scaler.transform(features), labels))

    # ------------------------------------------------------------- prediction
    def probabilities(self, x: np.ndarray, game_index, questions: pd.DataFrame) -> np.ndarray:
        """[Q, 10] for questions with columns games (game_index lists, all present
        in game_index) and k or num_games."""
        k = (questions["k"] if "k" in questions else questions["num_games"].map(k_bucket)).to_numpy()
        out = np.empty((len(questions), 10))
        if self.method == "B2":
            mean = question_mean(x, game_index, questions)
            for bucket in K_BUCKETS:
                mask = k == bucket
                if mask.any():
                    scaler, model = self.per_k[bucket]
                    out[mask] = model.predict_proba(scaler.transform(mean[mask]))
            return out
        values = self.game.log_posterior(x) if self.method == "B0" else self.game.linear(x)
        offset = 0.0 if self.method == "B0" else self.game.model.intercept_
        logits = question_mean(values, game_index, questions)
        for bucket in K_BUCKETS:
            mask = k == bucket
            out[mask] = softmax(self.temps[bucket] * logits[mask] + offset)
        return out

    # ----------------------------------------------------------- persistence
    def save(self, path: Path, **info) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"format": "humansl_rank_1", "model": self, "info": info}, handle)

    @staticmethod
    def load(path: Path) -> tuple["RankModel", dict]:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("format") != "humansl_rank_1":
            raise ValueError("not a HumanSL rank model")
        return payload["model"], payload["info"]
