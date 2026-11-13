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
valid_u : (after the gate passed) B2 fitted on every Valid-S game (65,223, as
          A-044 (ii)), scored once on the 1,000 Valid-U questions: AH (reference,
          0.6833) vs AH_C, player-grouped bootstrap of the gain. C of the games
          outside emb_stage1 comes from outputs/humansl_rankvec_feat_vu.

    python -m scripts.strength.humansl_rankvec select
    python -m scripts.strength.humansl_rankvec check
    python -m scripts.strength.humansl_rankvec analyze
    python -m scripts.strength.humansl_rankvec valid_u
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from scripts.strength.emb_stage1 import table
from scripts.strength.humansl_pilot import a024_features
from scripts.strength.humansl_questions import by_k, equal_k, fit_players
from go_aicup.features import humansl_rankvec as rv
from go_aicup.metrics import score_rank
from go_aicup.paths import PROCESSED, ROOT
from go_aicup.tasks.strength.humansl import SEED, RankModel
from go_aicup.tasks.strength.v2.aggregate import official_scores

DATASET = PROCESSED / "strength_v2" / "seed2027"
OUT = ROOT / "outputs"
RUN, FEAT, FEAT_VU = OUT / "humansl_rankvec", OUT / "humansl_rankvec_feat", OUT / "humansl_rankvec_feat_vu"
LEARN, MONITOR, MONITOR_PLAYERS, MIN_GAIN, BOOTSTRAP = 1350, 150, 0.1, 0.01, 2000


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
    from scripts.strength.humansl_torch_check import game_logp as reference_logp
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


def valid_u() -> None:
    # Row order as A-044 (ii): B2 draws synthetic questions by row index.
    fit = pd.concat([table(OUT / "humansl_q1000", ["S-fit"]), table(OUT / "humansl_qsel", ["S-select"]),
                     table(OUT / "humansl_vs_rest", ["S-fit", "S-extra", "S-select"])], ignore_index=True)
    test = table(OUT / "humansl_q1000", ["U"])
    questions = pd.read_parquet(OUT / "humansl_q1000" / "questions.parquet")
    questions = questions[questions["role"] == "U"].reset_index(drop=True)
    c = pd.concat([rv.run_features(FEAT), rv.run_features(FEAT_VU)]).drop_duplicates("game_index") \
        .set_index("game_index")
    for frame, name in ((fit, "Valid-S"), (test, "Valid-U")):
        missing = set(frame["game_index"]) - set(c.index)
        if missing:
            raise ValueError(f"{name}: {len(missing)} games without C features")
    human = [col for col in fit.columns if col.startswith(("own_", "opp_"))]

    def features(frame, with_c):
        x = np.hstack([a024_features(frame["game_index"]), frame[human].to_numpy()])
        return np.hstack([x, c.loc[frame["game_index"]].to_numpy()]) if with_c else x

    k = questions["k"].to_numpy()
    scores = {}
    for name, with_c in (("AH", False), ("AH_C", True)):
        model = RankModel("B2").fit(fit, pd.DataFrame(), features(fit, with_c))
        pred = score_rank(model.probabilities(features(test, with_c), test["game_index"].to_numpy(), questions))
        scores[name] = official_scores(pred, questions["rank_id"].to_numpy())
        print(name, by_k(scores[name], k), flush=True)
    groups = pd.Series(np.arange(len(questions))).groupby(questions["player_id"].astype(str).to_numpy()) \
        .apply(np.asarray).to_list()
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(BOOTSTRAP):
        rows = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        draws.append(equal_k(scores["AH_C"][rows], k[rows]) - equal_k(scores["AH"][rows], k[rows]))
    result = {"fit_games": len(fit), "fit_players": int(fit["player_id"].nunique()),
              "Valid-U": {name: by_k(value, k) for name, value in scores.items()},
              "gain_AH_C_minus_AH": round(equal_k(scores["AH_C"], k) - equal_k(scores["AH"], k), 4),
              "bootstrap_95": [round(float(q), 4) for q in np.quantile(draws, [0.025, 0.975])]}
    print(json.dumps(result, indent=1))
    (FEAT_VU / "result.json").write_text(json.dumps(result, indent=1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("select", "check", "analyze", "valid_u"))
    {"select": select, "check": check, "analyze": analyze, "valid_u": valid_u}[parser.parse_args().command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
