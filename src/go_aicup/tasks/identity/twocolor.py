"""Competition-1 models for games of either colour (added test set, 2026-11-11).

The original pipeline (predict.py, insurance.py) reads black moves only. Here a
game counts through the moves of the colour its player held:

opening   the pp+slot scorer run once per colour on that colour's plies (black
          0, 2, ...; white 1, 3, ...), each with its own candidate galleries and
          population prior; a question's score is the sum over its games. A
          candidate without a gallery game of that colour falls back to the
          population model, which is the same for every candidate.
style     the GE2E encoder was trained on black moves. A white move is encoded
          either as it is ("raw") or with the two stone colours exchanged so it
          looks like a black move ("swap").
"""
from __future__ import annotations

import numpy as np
import torch

from go_aicup.data.board_cache import BoardCache, assemble_inputs
from go_aicup.tasks.identity import opening
from go_aicup.tasks.identity.ge2e import MIN_PLY
from go_aicup.tasks.strength.shared.runtime import autocast_context


def color_plies(moves: int, color: int) -> np.ndarray:
    return np.arange(color, 2 * moves, 2)


def opening_scores(prefix, hashes, gallery_games, query_games, game_color, populations, params) -> np.ndarray:
    """[questions, candidates]; game_color[row] in {0 black, 1 white} is the player's colour,
    populations = (black population rows, white population rows)."""
    scores = np.zeros((len(query_games), len(gallery_games)))
    for color in (0, 1):
        gallery = [g[game_color[g] == color] for g in gallery_games]
        query = [q[game_color[q] == color] for q in query_games]
        has = np.array([len(q) > 0 for q in query])
        if not has.any():
            continue
        query = [q for q in query if len(q)]
        population = populations[color]
        plies = color_plies(params["moves"], color)
        pop = opening.population_prior(prefix, population, plies)
        base = opening.population_conditional(hashes, prefix, population, np.concatenate(query), plies, pop)
        part = opening.prefix_backoff(prefix, gallery, query, plies, pop, beta=params["beta"],
                                      hashes=hashes, base=base)
        if params["slot_moves"]:
            slot_plies = color_plies(params["slot_moves"], color)
            slot_pop = opening.population_prior(prefix, population, slot_plies)
            part = part + opening.slot_nb(prefix, gallery, query, slot_plies, slot_pop, params["alpha"])
        scores[has] += part
    return scores


@torch.no_grad()
def embed_games(model, config, cache: BoardCache, games: np.ndarray, colors: np.ndarray, lengths: np.ndarray,
                device, swap: bool, batch: int = 8192) -> np.ndarray:
    """float32 [len(games), D]: mean embedding of the player's moves at ply >= MIN_PLY.

    colors[i] is the player's colour in games[i]. swap=True exchanges the stone
    planes of white moves and marks them as black to move."""
    first = MIN_PLY + (colors.astype(np.int64) - MIN_PLY) % 2          # first own ply >= MIN_PLY
    plies = [np.arange(f, n, 2) for f, n in zip(first, lengths)]
    owner = np.concatenate([np.full(len(p), i) for i, p in enumerate(plies)])
    sample_game, sample_ply = games[owner], np.concatenate(plies)
    sums = np.zeros((len(games), config["model"]["embedding"]), np.float64)
    model.eval()
    for start in range(0, len(owner), batch):
        span = slice(start, start + batch)
        histories, side = cache.history_batch(sample_game[span], sample_ply[span])
        if not np.array_equal(side, colors[owner[span]]):
            raise ValueError("sampled ply is not the player's move")
        if swap:
            white = side == 1
            histories[white] = histories[white][:, :, ::-1]
            side = np.zeros_like(side)
        with autocast_context(config, device):
            vectors = model(torch.from_numpy(assemble_inputs(histories, side)).to(device, dtype=torch.float32))
        np.add.at(sums, owner[span], vectors.float().cpu().numpy().astype(np.float64))
    return (sums / np.array([len(p) for p in plies])[:, None]).astype(np.float32)


def normalise(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def style_scores(vectors: np.ndarray, gallery_games, query_games) -> np.ndarray:
    """cosine between the normalised mean of a question's game vectors and of each
    candidate's; `vectors` rows are indexed by the game rows used in the lists."""
    unit = normalise(vectors.astype(np.float64))
    candidates = np.stack([normalise(unit[g].mean(axis=0)) for g in gallery_games])
    queries = np.stack([normalise(unit[q].mean(axis=0)) for q in query_games])
    return queries @ candidates.T


def category_scores(codes: np.ndarray, gallery_games, query_games, n_codes: int, smoothing: float = 1.0) -> np.ndarray:
    """sum over a question's games of log P_c(code), P_c the candidate's gallery histogram
    smoothed toward the pooled one (as predict.hd_scores); code = HD + 2 * colour here."""
    prior = np.bincount(codes[np.concatenate(gallery_games)], minlength=n_codes) + 1.0
    prior /= prior.sum()
    counts = np.stack([np.bincount(codes[g], minlength=n_codes) for g in gallery_games])
    logp = np.log((counts + smoothing * prior) / (counts.sum(axis=1, keepdims=True) + smoothing))
    return np.stack([logp[:, codes[q]].sum(axis=1) for q in query_games])
