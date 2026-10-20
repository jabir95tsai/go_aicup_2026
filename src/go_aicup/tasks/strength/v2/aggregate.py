"""CPU game/question aggregation of V2 ten-class confidences; no torch.

Per game g with labelled-player moves T_g: confidence_sum[g] = sum_t p_t and
log_probability_sum[g] = sum_t log p_t. Single games decode by argmax of the
confidence sum (ties -> smaller rank_id). Official question rule (C2, A-025,
user-approved 2026-10-20): average per-game mean log-probabilities with weight
1/k, softmax(T_k * that) with the per-k temperature fitted on Valid-S, then
metrics.score_rank. Probability pooling and argmax are diagnostics only.
"""
from __future__ import annotations

import numpy as np

from go_aicup.metrics import NEAR, score_rank

N_RANKS = 10
CATEGORY_DTYPE = np.dtype([
    ("game_index", "<u4"), ("rank_id", "u1"), ("num_moves", "<u4"),
    ("confidence_sum", "<f8", (N_RANKS,)), ("log_probability_sum", "<f8", (N_RANKS,)),
    ("predicted_rank_id", "u1")])


def build_category(game_index, rank_id, num_moves, confidence_sum, log_probability_sum) -> np.ndarray:
    """Structured per-game result; rows must already be in game_index order."""
    game_index = np.asarray(game_index)
    if game_index.ndim != 1 or len(game_index) == 0 or np.any(np.diff(game_index.astype(np.int64)) <= 0):
        raise ValueError("game_index must be nonempty, unique and sorted")
    num_moves = np.asarray(num_moves)
    if num_moves.shape != game_index.shape or np.any(num_moves <= 0):
        raise ValueError("every game needs at least one labelled-player move")
    rank_id = np.asarray(rank_id)
    if rank_id.shape != game_index.shape or np.any((rank_id < 0) | (rank_id >= N_RANKS)):
        raise ValueError("rank_id must be 0..9 per game")
    confidence_sum, log_probability_sum = (np.asarray(value, dtype=np.float64)
                                           for value in (confidence_sum, log_probability_sum))
    for value in (confidence_sum, log_probability_sum):
        if value.shape != (len(game_index), N_RANKS) or not np.isfinite(value).all():
            raise ValueError("per-game sums must be finite [G,10]")
    result = np.empty(len(game_index), dtype=CATEGORY_DTYPE)
    result["game_index"], result["rank_id"], result["num_moves"] = game_index, rank_id, num_moves
    result["confidence_sum"], result["log_probability_sum"] = confidence_sum, log_probability_sum
    result["predicted_rank_id"] = game_predictions(confidence_sum)
    return result


def game_predictions(confidence_sum) -> np.ndarray:
    """Argmax of summed softmax; np.argmax keeps the first, i.e. smaller, rank."""
    return np.asarray(confidence_sum).argmax(axis=1)


def official_scores(pred, true) -> np.ndarray:
    error = np.abs(np.asarray(pred, dtype=np.int64) - np.asarray(true, dtype=np.int64))
    return np.where(error == 0, 1.0, np.where(error == 1, NEAR, 0.0))


def game_probabilities(category: np.ndarray) -> np.ndarray:
    return category["confidence_sum"] / category["num_moves"][:, None]


def monitor_metrics(category: np.ndarray) -> dict:
    """Move-weighted CE from stored log sums and game-equal official score."""
    log_true = np.take_along_axis(category["log_probability_sum"],
                                  category["rank_id"].astype(np.int64)[:, None], axis=1)
    moves = int(category["num_moves"].sum())
    return {"loss": float(-log_true.sum() / moves), "game_score": float(
        official_scores(category["predicted_rank_id"], category["rank_id"]).mean()),
        "games": len(category), "moves": moves}


def question_rows(category: np.ndarray, games) -> np.ndarray:
    """Map game_index values to category rows; every game must be present."""
    games = np.asarray(games, dtype=np.int64)
    rows = np.searchsorted(category["game_index"], games)
    if np.any(rows >= len(category)) or np.any(category["game_index"][np.minimum(rows, len(category) - 1)] != games):
        raise ValueError("question references a game without a saved result")
    return rows


def question_logits(category: np.ndarray, questions, method: str = "mean_probability") -> np.ndarray:
    """[Q,10] log-scale question scores; softmax(T * logits) gives probabilities.

    Each game weighs 1/k regardless of its length.
    mean_probability: log of the average per-game mean softmax (C0/C3,
    diagnostic). mean_log_probability: average per-game mean log-prob
    (normalised geometric mean at T=1; with fitted T_k it is the official C2).
    """
    if method == "mean_probability":
        per_game = game_probabilities(category)
    elif method == "mean_log_probability":
        per_game = category["log_probability_sum"] / category["num_moves"][:, None]
    else:
        raise ValueError(f"unknown aggregation method: {method}")
    result = np.empty((len(questions), N_RANKS))
    for number, games in enumerate(questions):
        result[number] = per_game[question_rows(category, games)].mean(axis=0)
    return np.log(np.maximum(result, 1e-300)) if method == "mean_probability" else result


def tempered_probabilities(logits, temperature: float = 1.0) -> np.ndarray:
    """softmax(temperature * logits) per row; temperature > 1 sharpens."""
    if not temperature > 0:
        raise ValueError("temperature must be positive")
    scaled = temperature * np.asarray(logits, dtype=np.float64)
    scaled = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return scaled / scaled.sum(axis=1, keepdims=True)


def tempered_by_k(logits, k, temperatures) -> np.ndarray:
    """Temper each question with its bucket's temperature; temperatures maps str(k) -> T_k."""
    logits, k = np.asarray(logits, dtype=np.float64), np.asarray(k)
    result = np.empty_like(logits)
    for bucket in np.unique(k):
        mask = k == bucket
        result[mask] = tempered_probabilities(logits[mask], temperatures[str(int(bucket))])
    return result


def question_probabilities(category: np.ndarray, questions, method: str = "mean_probability",
                           temperature: float = 1.0) -> np.ndarray:
    return tempered_probabilities(question_logits(category, questions, method), temperature)


TEMPERATURE_GRID = np.exp(np.linspace(np.log(0.1), np.log(100.0), 601))


def fit_temperature(logits, labels, grid=TEMPERATURE_GRID) -> tuple[float, bool]:
    """Grid temperature minimising mean question NLL; also flags a grid-edge optimum."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    losses = []
    for temperature in grid:
        scaled = temperature * logits
        scaled -= scaled.max(axis=1, keepdims=True)
        log_norm = np.log(np.exp(scaled).sum(axis=1))
        losses.append(float((log_norm - scaled[np.arange(len(labels)), labels]).mean()))
    best = int(np.argmin(losses))
    return float(grid[best]), best in (0, len(grid) - 1)


def decode_questions(probabilities, decoder: str) -> np.ndarray:
    if decoder == "score_rank":
        return score_rank(probabilities)
    if decoder == "argmax":
        return np.asarray(probabilities).argmax(axis=1)
    raise ValueError(f"unknown decoder: {decoder}")
