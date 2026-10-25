"""C2: HumanSL vectors learned per Yugen rank (RESULTS A-055).

select  : learning games from the A-047 train-split draw (outputs/humansl_train20),
          players of the S-select half excluded. Per rank, 10% of players (seed
          2027) are monitor players; 1,350 learn games and 150 monitor games per
          rank -> outputs/humansl_rankvec/games.parquet (role learn / monitor).
check   : (GPU) with every vector at its rank_1d initial value, per-move log p
          must equal the unmodified network with the rank_1d metadata row
          (A-050 path, Board.would_be_legal mask), max |diff| <= 1e-3 on 2 games,
          in full fp32 (TF32 off).
analyze : B2 fitted on the Valid-S fitting games (q1000 S-fit + vs_rest S-fit /
          S-extra, 38,589 games, as A-044 (i)), scored on the 1,000 S-select
          questions; C = own_r* / opp_r* means from humansl_rankvec extract
          (outputs/humansl_rankvec_feat, games of outputs/emb_stage1).
            AH (reference, A-044 (i) 0.7101), AH_C, A_C, H_C, C
Pre-registered gate: AH_C - AH >= +0.01.

    python scripts/humansl_rankvec.py select
    python scripts/humansl_rankvec.py check
    python scripts/humansl_rankvec.py analyze
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from emb_stage1 import table
from humansl_pilot import a024_features
from humansl_questions import by_k, fit_players
from go_aicup.features import humansl_rankvec as rv
from go_aicup.metrics import score_rank
from go_aicup.paths import PROCESSED, ROOT
from go_aicup.tasks.strength.humansl import RankModel
from go_aicup.tasks.strength.v2.aggregate import official_scores

DATASET = PROCESSED / "strength_v2" / "seed2027"
OUT = ROOT / "outputs"
RUN, FEAT = OUT / "humansl_rankvec", OUT / "humansl_rankvec_feat"
LEARN, MONITOR, MONITOR_PLAYERS, MIN_GAIN = 1350, 150, 0.1, 0.01


def select_half() -> set[str]:
    seen = pd.read_parquet(DATASET / "strength_v2_valid_seen_questions.parquet")
    return set(seen["player_id"].astype(str)) - fit_players(seen)


def select() -> None:
    pool = pd.read_parquet(OUT / "humansl_train20" / "games.parquet")
    pool = pool[~pool["player_id"].astype(str).isin(select_half())]
    rng = np.random.default_rng(rv.SEED)
    picks = []
    for rank, games in pool.groupby("rank_id", sort=True):
        players = rng.permutation(np.sort(games["player_id"].unique()))
        held = set(players[:int(round(MONITOR_PLAYERS * len(players)))])
        monitor = games[games["player_id"].isin(held)]
        learn = games[~games["player_id"].isin(held)]
        picks.append(learn.iloc[np.sort(rng.permutation(len(learn))[:LEARN])].assign(role="learn"))
        picks.append(monitor.iloc[np.sort(rng.permutation(len(monitor))[:MONITOR])].assign(role="monitor"))
    out = pd.concat(picks).sort_values("game_index").reset_index(drop=True)
    RUN.mkdir(parents=True, exist_ok=True)
    out.to_parquet(RUN / "games.parquet")
    summary = out.groupby(["rank_id", "role"]).agg(games=("game_index", "size"), players=("player_id", "nunique"),
                                                   positions=("length", "sum"))
    print(summary.to_string())
    print("games", len(out), "positions", int(out["length"].sum()))


def check() -> None:
    import torch
    from humansl_torch_check import game_logp as reference_logp
    from katago.game.features import Features
    from katago.train.load_model import load_model
    from go_aicup.features.humansl import cache_games
    from go_aicup.features.humansl_torch import CKPT, encode_game, _init_worker, profile_metadata

    device = torch.device("cuda")
    model, init = rv.load_frozen(device, fp32=True)
    plain, _, _ = load_model(str(CKPT), use_swa=False, device=device)
    plain.eval()
    features = Features(plain.config, plain.pos_len)
    _init_worker(plain.config)
    vectors = init.repeat(rv.RANKS, 2, 1)
    games = pd.read_parquet(RUN / "games.parquet").iloc[:2]
    worst = 0.0
    for (moves, colors), color in zip(cache_games(games["game_index"]), games["player_color"]):
        bins, globals_, played = encode_game((moves, colors))
        mine = rv.game_logp(model, vectors, bins, globals_, played, colors == color).cpu().numpy()
        ref = reference_logp(plain, features, moves, colors, [profile_metadata("rank_1d")], device)[:, 0]
        worst = max(worst, float(np.abs(mine - ref[:, None]).max()))
    print(json.dumps({"max_abs_diff": worst, "pass": worst <= 1e-3}))


def analyze() -> None:
    fit = pd.concat([table(OUT / "humansl_q1000", ["S-fit"]), table(OUT / "humansl_vs_rest", ["S-fit", "S-extra"])],
                    ignore_index=True)
    test = table(OUT / "humansl_qsel", ["S-select"])
    questions = pd.read_parquet(OUT / "humansl_qsel" / "questions.parquet")
    c = rv.run_features(FEAT).set_index("game_index")
    for frame, name in ((fit, "fit"), (test, "S-select")):
        missing = set(frame["game_index"]) - set(c.index)
        if missing:
            raise ValueError(f"{name}: {len(missing)} games without C features")
    human = [col for col in fit.columns if col.startswith(("own_", "opp_"))]

    def blocks(frame):
        return {"A": a024_features(frame["game_index"]), "H": frame[human].to_numpy(),
                "C": c.loc[frame["game_index"]].to_numpy()}

    f, t = blocks(fit), blocks(test)
    views = {"AH": ("A", "H"), "AH_C": ("A", "H", "C"), "A_C": ("A", "C"), "H_C": ("H", "C"), "C": ("C",)}
    k = questions["k"].to_numpy()
    result = {"fit_games": len(fit), "S-select": {}}
    for name, parts in views.items():
        x_fit, x_test = np.hstack([f[p] for p in parts]), np.hstack([t[p] for p in parts])
        model = RankModel("B2").fit(fit, pd.DataFrame(), x_fit)
        pred = score_rank(model.probabilities(x_test, test["game_index"].to_numpy(), questions))
        result["S-select"][name] = by_k(official_scores(pred, questions["rank_id"].to_numpy()), k)
        print(name, result["S-select"][name], flush=True)
    gain = result["S-select"]["AH_C"]["equal_k"] - result["S-select"]["AH"]["equal_k"]
    result["gate"] = {"gain_AH_C_minus_AH": round(gain, 4), "min_gain": MIN_GAIN, "pass": bool(gain >= MIN_GAIN)}
    print(json.dumps(result["gate"]))
    (FEAT / "result.json").write_text(json.dumps(result, indent=1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("select", "check", "analyze"))
    {"select": select, "check": check, "analyze": analyze}[parser.parse_args().command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
