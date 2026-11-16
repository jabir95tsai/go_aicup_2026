"""Does the opponent side of a game identify the player? (added set, bench; 2026-11-17)

Hypothesis: players often play several games against the same opponent, so the opponent's
moves in a query game may match the opponent moves in the true candidate's gallery games.
The current model only embeds the player's own moves.

Stage 1 (GPU): embed the opponent side of every bench game with the frozen 9,600-step encoder
(same settings as the saved own-side vectors: colour swap, the run's min_ply).
Stage 2 (CPU): on pool 0 / pool 1 draws (100-119 / 100-129), report
  * own-style only, opponent-style only (centroid cosine, as style_scores)
  * opponent "max" score: mean over query games of the best cosine to any single gallery game
    of the candidate (a repeated opponent should show up per game, not only in the centroid)
  * full fusion + lam_opp * w * opponent cosine, lam_opp in {0, 2, 4, 8, 16}
Rank similarity alone also makes opponent vectors informative (matchmaking pairs similar
strengths), so the opponent-only score is compared with a rank-only reference: candidates
scored by whether their rank equals the question's rank (uniform within rank).

usage: python -m scripts.identity.pid2_oppstyle [--embed]
"""
import argparse
import json

import numpy as np
import pandas as pd

from go_aicup.paths import ROOT
from go_aicup.data import cv
from go_aicup.tasks.identity import data, twocolor
from scripts.identity.pid2_diag_d123a import COLS, bucket_means, paired
from scripts.identity.pid2_paired import points
from scripts.identity.pid2_transductive import load

RUN = "pid_ge2e_v4_d4_hd_9600"
BENCH = ROOT / "outputs" / "pid" / "pid2_bench"
OUT = ROOT / "outputs" / "pid" / "pid2_oppstyle"
LAMBDAS = (0.0, 2.0, 4.0, 8.0, 16.0)


def embed() -> None:
    import torch
    from go_aicup.data.board_cache import BoardCache
    from go_aicup.tasks.identity.ge2e import MIN_PLY, IdentityEncoder
    run_dir = ROOT / "outputs" / "pid" / "ge2e" / RUN
    saved = np.load(BENCH / RUN / "embeddings.npz")
    games = saved["game_index"]
    meta = data.load_meta()
    own = meta["player_color"].to_numpy()[games].astype(np.int8)
    lengths = meta["length"].to_numpy()[games]
    config = json.loads((run_dir / "config.json").read_text())
    model = IdentityEncoder(**config["model"])
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)["model"])
    device = torch.device("cuda")
    model.to(device)
    min_ply = config.get("sampling", {}).get("min_ply", MIN_PLY)
    cache = BoardCache(data.CACHE_DIR)
    # integrity: the own side reproduces the saved vectors on a slice
    check = twocolor.embed_games(model, config, cache, games[:500], own[:500], lengths[:500], device,
                                 swap=True, min_ply=min_ply)
    cos = (twocolor.normalise(check) * twocolor.normalise(saved["swap"][:500])).sum(1)
    print(f"own side reproduce min cosine {cos.min():.6f}", flush=True)
    if cos.min() < 0.9999:
        raise SystemExit("own-side vectors do not reproduce the saved bench vectors")
    opp = twocolor.embed_games(model, config, cache, games, (1 - own).astype(np.int8), lengths, device,
                               swap=True, min_ply=min_ply)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "opp_embeddings.npz", game_index=games, opp=opp, min_ply=min_ply)
    print(f"embedded opponent side of {len(games)} games (min_ply {min_ply}); finite={np.isfinite(opp).all()}")


def max_scores(unit, gallery, query) -> np.ndarray:
    """mean over a question's games of max cosine to any single gallery game of the candidate."""
    sizes = [len(g) for g in gallery]
    flat = unit[np.concatenate(gallery)]
    owner = np.repeat(np.arange(len(gallery)), sizes)
    out = np.empty((len(query), len(gallery)))
    for i, q in enumerate(query):
        sim = unit[q] @ flat.T                                   # [k, gallery games]
        best = np.full((len(q), len(gallery)), -np.inf)
        np.maximum.at(best.T, owner, sim.T)
        out[i] = best.mean(axis=0)
    return out


def score() -> None:
    env = load(RUN)
    saved = np.load(BENCH / RUN / "embeddings.npz")
    opp_file = np.load(OUT / "opp_embeddings.npz")
    assert np.array_equal(opp_file["game_index"], saved["game_index"])
    own_vec, opp_vec = saved["swap"], opp_file["opp"]
    opp_unit = twocolor.normalise(opp_vec.astype(np.float64))
    w = env["summary"]["w"]
    meta = data.load_meta()
    rank_of = dict(zip(meta["player_id"].astype(str), meta["rank"].astype(str)))
    rows = []
    for pool, n in ((0, 20), (1, 30)):
        for draw in range(100, 100 + n):
            candidates, gallery, q = cv.draw_pid2(env["bank"], pool, draw)
            query = list(q["games"])
            ks = q["k"].to_numpy()
            s = env["summary"]
            rest = (twocolor.opening_scores(env["prefix"], env["hashes"], gallery, query, env["color"],
                                            env["populations"], s["opening"])
                    + s["mu"] * twocolor.category_scores(env["codes"], gallery, query, 4))
            pos = env["position"]
            g, qq = [pos[x] for x in gallery], [pos[x] for x in query]
            own = twocolor.style_scores(own_vec, g, qq)
            opp = twocolor.style_scores(opp_vec, g, qq)
            opp_max = max_scores(opp_unit, g, qq)
            cand_rank = np.array([rank_of[str(c)] for c in candidates])
            q_rank = np.array([rank_of[str(p)] for p in q["player_id"]])
            rng = np.random.default_rng(draw)
            rank_only = (q_rank[:, None] == cand_rank[None, :]) + 1e-6 * rng.random((len(q), len(candidates)))
            arms = {"own only": own, "opp only": opp, "opp max only": opp_max, "rank only (oracle)": rank_only}
            for lam in LAMBDAS:
                arms[f"full+opp{lam:g}"] = rest + s["lambda"] * w * own + lam * w * opp
            for lam in (4.0, 8.0):
                arms[f"full+oppmax{lam:g}"] = rest + s["lambda"] * w * own + lam * w * opp_max
            for arm, sc in arms.items():
                rows.append({"pool": pool, "draw": draw, "arm": arm, **bucket_means(points(sc, candidates, q), ks)})
            print(f"pool {pool} draw {draw}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "per_draw.csv", index=False)
    report = {}
    for pool in (0, 1):
        t = paired(frame[frame["pool"] == pool], "full+opp0")
        wide = frame[frame["pool"] == pool].pivot(index="draw", columns="arm", values="all")
        t["se_all"] = [float((wide[a] - wide["full+opp0"]).std(ddof=1) / np.sqrt(len(wide))) for a in t["arm"]]
        print(f"\n== pool {pool} (difference to full fusion, own only) ==\n{t.round(4).to_string(index=False)}", flush=True)
        report[f"pool{pool}"] = t.to_dict(orient="records")
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=float))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embed", action="store_true", help="stage 1 (GPU) before scoring")
    args = parser.parse_args()
    if args.embed:
        embed()
    score()


if __name__ == "__main__":
    main()
