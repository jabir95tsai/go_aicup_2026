"""Where do the adjacent-rank errors of the base sit? (RESULTS 2026-11-13)

questions  cvec2 base (PriorMatched B2, fit Valid-S + train20 HD<=1) on the Valid-U E2 questions,
           five seeds pooled: official score, exact share and signed error (predicted minus true
           rank) of each true rank, split by question colour, HD1 share and k.
phase      per-move C v2 log p (rank_permove shards; Valid-S, train20, Valid-U, HD<=1). For each
           neighbouring pair (r, r+1) of A/1D, 1D/2D, 2D/3D and each slice of the game, the game
           statistic t = mean over the slice's moves of log p[r+1] - log p[r]:
             d_game    separation of single games of rank r and r+1 (difference of means / pooled sd)
             d_player  the same on player means (players with >= MIN_GAMES games in the slice)
             bias      share of the game variance that is between players (one-way ANOVA estimate)
           slices: ply windows x own / opponent moves, and whole games by colour and by HD.
           A slice whose d_player / d_game is high carries evidence that does not saturate with k.

    python -m scripts.strength.rank_error_split questions [--jobs 5]
    python -m scripts.strength.rank_error_split phase
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import ROOT

OUT = ROOT / "outputs" / "rank" / "humansl" / "error_split"
PERMOVE = ROOT / "outputs" / "rank" / "humansl" / "permove"
RANKS = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")
PAIRS = ((3, 4), (4, 5), (5, 6))
WINDOWS = ((0, 30), (30, 80), (80, 140), (140, 200), (200, 10_000))
MIN_GAMES = 10
E = float(np.exp(-1))


def _predict(seed, fit, x_fit, prior, eval_games, x_eval, q):
    import go_aicup.tasks.strength.humansl as hs
    import scripts.strength.rank_more_players as more
    hs.SEED = seed
    model = more.PriorMatched(prior).fit(fit, pd.DataFrame(), x_fit)
    return score_rank(model.probabilities(x_eval, eval_games["game_index"].to_numpy(), q))


def questions(args) -> None:
    from joblib import Parallel, delayed
    import scripts.strength.humansl_hd_filter as hf
    import scripts.strength.rank_cvec_v2 as cv2
    import scripts.strength.rank_more_players as more

    games, fit, eval_games, xs = cv2.data()
    x_fit, x_eval = xs["v2"]
    hd01 = (fit["handicap"] <= 1).to_numpy()
    prior = more.player_prior(fit[hd01 & (fit["role"] == "S").to_numpy()])
    fit, x_fit = fit[hd01].reset_index(drop=True), x_fit[hd01]
    q = hf.question_sets(games, rebuild=False)["E2"].reset_index(drop=True)
    preds = Parallel(n_jobs=args.jobs)(delayed(_predict)(s, fit, x_fit, prior, eval_games, x_eval, q) for s in hf.SEEDS)
    info = eval_games.set_index("game_index")
    hd1 = np.array([float((info.loc[list(g), "handicap"] == 1).mean()) for g in q["games"]])
    color = np.array([int(info.loc[g[0], "player_color"]) for g in q["games"]])
    table = pd.concat([pd.DataFrame({"seed": s, "rank": q["rank_id"], "k": q["k"], "color": color, "hd1": hd1,
                                     "pred": p, "player_id": q["player_id"].astype(str)})
                       for s, p in zip(hf.SEEDS, preds)], ignore_index=True)
    table["diff"] = table["pred"] - table["rank"]
    table["score"] = (table["diff"] == 0) + E * (table["diff"].abs() == 1)
    table["hd_bin"] = pd.cut(table["hd1"], [-0.01, 0.0, 0.2, 0.4, 1.0], labels=["0", "(0,0.2]", "(0.2,0.4]", ">0.4"])
    table["band"] = np.where(table["rank"] < 3, "D-B", np.where(table["rank"] <= 6, "A-3D", "4D-6D"))

    def cell(g):
        return pd.Series({"n": len(g) // len(hf.SEEDS), "score": g["score"].mean(), "exact": (g["diff"] == 0).mean(),
                          "over": (g["diff"] >= 1).mean(), "under": (g["diff"] <= -1).mean(),
                          "off2": (g["diff"].abs() >= 2).mean(), "signed": g["diff"].mean()})

    result = {}
    for name, keys in (("band", ["band"]), ("rank", ["rank"]), ("band_color", ["band", "color"]),
                       ("rank_color", ["rank", "color"]), ("band_hd", ["band", "hd_bin"]),
                       ("band_color_hd", ["band", "color", "hd_bin"]), ("band_k", ["band", "k"]),
                       ("mid_rank_k", ["rank", "k"])):
        part = table if name != "mid_rank_k" else table[table["band"] == "A-3D"]
        t = part.groupby(keys, observed=True).apply(cell).round(4)
        result[name] = t.reset_index().to_dict("records")
        print(f"--- {name}\n{t.to_string()}", flush=True)
    # do a player's questions err in the same direction? (signed error, A-3D, seed 2027)
    one = table[(table["seed"] == hf.SEEDS[0]) & (table["band"] == "A-3D")]
    multi = one.groupby("player_id").filter(lambda g: len(g) >= 2)
    between = multi.groupby("player_id")["diff"].mean().var()
    result["player_consistency"] = {"players": int(multi["player_id"].nunique()), "questions": int(len(multi)),
                                    "var_total": float(multi["diff"].var()), "var_player_means": float(between)}
    print(json.dumps(result["player_consistency"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "questions.json").write_text(json.dumps(result, indent=1, default=float))


def _separation(t, rank, player, pair):
    """(d_game, d_player, bias share, games) for the statistic t of games with rank in pair."""
    out = []
    keep = np.isfinite(t) & np.isin(rank, pair)
    frame = pd.DataFrame({"t": t[keep], "rank": rank[keep], "player": player[keep]})
    lo, hi = frame[frame["rank"] == pair[0]], frame[frame["rank"] == pair[1]]
    sd = np.sqrt((lo["t"].var() + hi["t"].var()) / 2)
    d_game = (hi["t"].mean() - lo["t"].mean()) / sd
    stats = frame.groupby("player").agg(mean=("t", "mean"), n=("t", "size"), rank=("rank", "first"), var=("t", "var"))
    big = stats[stats["n"] >= MIN_GAMES]
    p_lo, p_hi = big[big["rank"] == pair[0]], big[big["rank"] == pair[1]]
    sd_p = np.sqrt((p_lo["mean"].var() + p_hi["mean"].var()) / 2)
    d_player = (p_hi["mean"].mean() - p_lo["mean"].mean()) / sd_p
    within = float((big["var"] * (big["n"] - 1)).sum() / (big["n"] - 1).sum())
    between = float(np.mean([g["mean"].var() for _, g in big.groupby("rank")]) - within * (1 / big["n"]).mean())
    return d_game, d_player, max(between, 0.0) / (max(between, 0.0) + within), int(keep.sum()), float(big["n"].median())


def phase(_args) -> None:
    import scripts.strength.rank_cvec_v2 as cv2
    games, _, _, _ = cv2.data()
    info = games[games["handicap"] <= 1].drop_duplicates("game_index").set_index("game_index")
    names, slices = [], []
    for side in (0, 1):
        for low, high in WINDOWS:
            names.append(f"{'own' if side == 0 else 'opp'} ply {low}-{high if high < 10_000 else 'end'}")
        names.append(f"{'own' if side == 0 else 'opp'} all")
    index, sums, counts = [], [], []
    for run in sorted(PERMOVE.glob("c_*_hd*")):
        if "test" in run.name:
            continue
        for shard in sorted(p for p in run.glob("shard_*") if (p / "meta.json").exists()):
            g = pd.read_parquet(shard / "games.parquet")
            done = np.load(shard / "done.npy")
            slot = np.load(shard / "move_slot.npy").astype(np.int64)
            ply = np.load(shard / "move_ply.npy").astype(np.int64)
            logp = np.load(shard / "move_logp.npy").astype(np.float64)
            window = np.digitize(ply, [w[1] for w in WINDOWS[:-1]])
            code = slot * len(WINDOWS) + window                      # (game, side, window)
            size = 2 * len(g) * len(WINDOWS)
            total = np.stack([np.bincount(code, weights=logp[:, r], minlength=size) for r in range(10)], axis=1)
            n = np.bincount(code, minlength=size)
            index.append(g["game_index"].to_numpy()[done])
            sums.append(total.reshape(len(g), 2, len(WINDOWS), 10)[done])
            counts.append(n.reshape(len(g), 2, len(WINDOWS))[done])
    index, sums, counts = np.concatenate(index), np.concatenate(sums), np.concatenate(counts)
    keep = np.isin(index, info.index)
    index, sums, counts = index[keep], sums[keep], counts[keep]
    meta = info.loc[index]
    rank, player = meta["rank_id"].to_numpy(), meta["player_id"].astype(str).to_numpy()
    color, hd = meta["player_color"].to_numpy(), meta["handicap"].to_numpy()
    print(f"games {len(index)}, players {len(set(player))}", flush=True)

    def stat(side, window, pair):
        if window is None:
            s, n = sums[:, side].sum(1), counts[:, side].sum(1)
        else:
            s, n = sums[:, side, window], counts[:, side, window]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n >= 5, (s[:, pair[1]] - s[:, pair[0]]) / n, np.nan)

    rows = []
    for side in (0, 1):
        for window in [*range(len(WINDOWS)), None]:
            label = f"{'own' if side == 0 else 'opp'} " + ("all" if window is None else
                                                             f"ply {WINDOWS[window][0]}-{WINDOWS[window][1] if WINDOWS[window][1] < 10_000 else 'end'}")
            moves = float((counts[:, side].sum(1) if window is None else counts[:, side, window]).mean())
            values = np.array([_separation(stat(side, window, pair), rank, player, pair)[:3] for pair in PAIRS])
            rows.append({"slice": label, "moves": round(moves, 1), "d_game": values[:, 0].mean(), "d_player": values[:, 1].mean(),
                         "bias": values[:, 2].mean(), **{f"d_game {RANKS[a]}/{RANKS[b]}": v for (a, b), v in zip(PAIRS, values[:, 0])},
                         **{f"d_player {RANKS[a]}/{RANKS[b]}": v for (a, b), v in zip(PAIRS, values[:, 1])}})
    for label, mask in (("black games", color == 0), ("white games", color == 1), ("HD0 games", hd == 0),
                        ("HD1 as black", (hd == 1) & (color == 0)), ("HD1 as white", (hd == 1) & (color == 1))):
        for side in (0, 1):
            values = np.array([_separation(np.where(mask, stat(side, None, pair), np.nan), rank, player, pair)[:3]
                               for pair in PAIRS])
            rows.append({"slice": f"{label}, {'own' if side == 0 else 'opp'}", "moves": float("nan"),
                         "d_game": values[:, 0].mean(), "d_player": values[:, 1].mean(), "bias": values[:, 2].mean(),
                         **{f"d_game {RANKS[a]}/{RANKS[b]}": v for (a, b), v in zip(PAIRS, values[:, 0])},
                         **{f"d_player {RANKS[a]}/{RANKS[b]}": v for (a, b), v in zip(PAIRS, values[:, 1])}})
    table = pd.DataFrame(rows).round(3)
    print(table.to_string())
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "phase.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("questions")
    p.add_argument("--jobs", type=int, default=5)
    p.set_defaults(fn=questions)
    sub.add_parser("phase").set_defaults(fn=phase)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
