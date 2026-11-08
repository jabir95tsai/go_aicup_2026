"""P3: HumanSL (H) and rank-vector (C) features of HD1 games at komi 0.5 (RESULTS 2026-11-08).

HD1 (Q&A Q13) is 讓先: no komi, a tie goes to White, about KataGo komi 0.5. Every H and
C feature so far was computed at komi 6.5, which HumanSL sees as a global input, so in
HD1 games (23% of the test games) its move likelihoods are conditioned on the wrong
score. P3 recomputes H and C for HD1 games only, at komi 0.5, and refits P1+P2.

    python scripts/humansl_p3_komi.py tables            # HD1 game tables (Valid-S/U, test)
    (GPU) humansl / humansl_rankvec extract --komi 0.5 on each table   # slurm/rank_p3_komi.slurm
    python scripts/humansl_p3_komi.py evaluate          # gate on E2, five seeds
    python scripts/humansl_p3_komi.py export --job ID   # only if the gate passes

Arms (all fitted on Valid-S HD<=1 with the HD1 flags, as P1+P2):
  all      P1+P2 (komi 6.5 features; the reference arm of compare)
  h        H of HD1 games at komi 0.5, C unchanged
  hc       H and C of HD1 games at komi 0.5
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys

import numpy as np
import pandas as pd

import go_aicup.tasks.strength.humansl as hs
from go_aicup.data.board_cache import RANKS
from go_aicup.features import humansl as hfeat
from go_aicup.features import humansl_rankvec as cfeat
from go_aicup.metrics import score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength import humansl_predict as hp
from go_aicup.tasks.strength.humansl import RankModel

import humansl_hd_filter as hf

OUT = ROOT / "outputs" / "rank" / "humansl" / "p3_komi"
WORK = ROOT / "outputs" / "rank_test_ahc_713770" / "rank"
RAW = ROOT / "data" / "raw" / "tests"
SAMPLE = RAW / "sample_submission_rank.csv"
H_COLS = slice(10, 40)    # x = [A (10) | H own 15, opp 15 | C own 10, opp 10]
C_COLS = slice(40, 60)


def tables() -> None:
    games, fit, eval_games, *_ = hf.setup()
    OUT.mkdir(parents=True, exist_ok=True)
    train = games.loc[games["handicap"] == 1, ["game_index", "player_color", "role"]].reset_index(drop=True)
    train.to_parquet(OUT / "train_hd1.parquet")
    report = {"train_hd1": {r: int((train["role"] == r).sum()) for r in ("S", "U")}}
    for part in ("public", "private"):
        work = WORK / f"rank_{part}"
        flags = hp.hd1_features(work, RAW / f"rank_prediction_test_{part}.csv")
        hd1 = flags.index[flags.sum(axis=1) > 0]
        cache = pd.read_parquet(work / "cache" / "games.parquet").set_index("game_index")
        table = cache.loc[hd1, ["player_color"]].rename_axis("game_index").reset_index()
        table.to_parquet(OUT / f"test_{part}_hd1.parquet")
        report[f"test_{part}_hd1"] = len(table)
    (OUT / "tables.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


def replaced(games: pd.DataFrame, x: np.ndarray, use_c: bool) -> tuple[np.ndarray, dict]:
    """x with H (and C) rows of HD1 games replaced by the komi-0.5 runs."""
    h = hfeat.run_features(OUT / "h_train").set_index("game_index")
    h = h[[c for c in h.columns if c.startswith(("own_", "opp_"))]]
    c = cfeat.run_features(OUT / "c_train").set_index("game_index")
    rows = pd.Series(np.arange(len(games)), index=games["game_index"].to_numpy())
    hd1 = games["game_index"][games["handicap"] == 1].to_numpy()
    have = np.intersect1d(np.intersect1d(hd1, h.index), c.index)
    x = x.copy()
    x[rows.loc[have].to_numpy(), H_COLS] = h.loc[have].to_numpy()
    if use_c:
        x[rows.loc[have].to_numpy(), C_COLS] = c.loc[have].to_numpy()
    return x, {"hd1_games": int(len(hd1)), "recomputed": int(len(have))}


def evaluate() -> None:
    games, fit, eval_games, x_fit, x_eval = hf.setup()
    all_x = np.vstack([x_fit, x_eval])
    both = pd.concat([fit, eval_games], ignore_index=True)
    x_h, cov = replaced(both, all_x, use_c=False)
    x_hc, _ = replaced(both, all_x, use_c=True)
    n = len(fit)
    sets = hf.question_sets(games, rebuild=False)
    hd01 = (fit["handicap"] <= 1).to_numpy()
    flags_fit, flags_eval = hf.hd1_columns(fit), hf.hd1_columns(eval_games)
    arms = {name: (hd01, np.hstack([x[:n], flags_fit]), np.hstack([x[n:], flags_eval]))
            for name, x in (("all", all_x), ("h", x_h), ("hc", x_hc))}
    # feature shift on HD1 games (komi 0.5 minus 6.5), a sanity check that the runs differ
    hd1 = (both["handicap"] == 1).to_numpy()
    shift = {"own_h_mean": float((x_h[hd1, 10:25] - all_x[hd1, 10:25]).mean()),
             "opp_h_mean": float((x_h[hd1, 25:40] - all_x[hd1, 25:40]).mean()),
             "c_mean": float((x_hc[hd1, C_COLS] - all_x[hd1, C_COLS]).mean())}
    result = {"coverage": cov, "shift_hd1": shift}
    print(json.dumps(result), flush=True)
    for treatment in ("h", "hc"):
        sub = {k: arms[k] for k in ("all", treatment)}
        hf.compare(fit, eval_games, sets, sub, treatment, result.setdefault(treatment, {"seeds": {}}),
                   f"result_p3_{treatment}.json")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=1))


def export(job: str, arm: str) -> None:
    """Fit the arm at seed 2027 on Valid-S HD<=1 and predict the test parts with HD1 games replaced."""
    games, fit, eval_games, x_fit, x_eval = hf.setup()
    both = pd.concat([fit, eval_games], ignore_index=True)
    x, _ = replaced(both, np.vstack([x_fit, x_eval]), use_c=arm == "hc")
    hd01 = (fit["handicap"] <= 1).to_numpy()
    with hf.PRODUCTION.open("rb") as handle:
        info = pickle.load(handle)["info"]
    hs.SEED = hf.SEED
    model = RankModel("B2").fit(fit[hd01].reset_index(drop=True), pd.DataFrame(),
                                np.hstack([x[:len(fit)], hf.hd1_columns(fit)])[hd01])
    columns = info["columns"] + hp.HD1_COLUMNS
    template = pd.read_csv(SAMPLE, dtype=str)
    parts, report = [], {}
    for part in ("public", "private"):
        work = WORK / f"rank_{part}"
        table = hp.features(hp.a024_per_game(np.load(work / "category.npy", allow_pickle=False)),
                            hp.release_features(work / "humansl"), hp.rankvec_features([work / "rankvec"]))
        table = table.join(hp.hd1_features(work, RAW / f"rank_prediction_test_{part}.csv"), how="inner")
        if list(table.columns) != columns:
            raise ValueError("test feature columns differ")
        h = hfeat.run_features(OUT / f"h_test_{part}").set_index("game_index")
        h = h[[c for c in h.columns if c.startswith(("own_", "opp_"))]]
        hd1 = table.index[table[hp.HD1_COLUMNS].sum(axis=1) > 0]
        if not set(hd1) <= set(h.index):
            raise ValueError(f"{part}: missing komi-0.5 H features")
        table.loc[hd1, [c for c in columns if c.startswith(("own_", "opp_"))]] = h.loc[hd1].to_numpy()
        if arm == "hc":
            c = cfeat.run_features(OUT / f"c_test_{part}").set_index("game_index").add_prefix("c_")
            if not set(hd1) <= set(c.index):
                raise ValueError(f"{part}: missing komi-0.5 C features")
            table.loc[hd1, list(c.columns)] = c.loc[hd1].to_numpy()
        questions = hp.question_table(work)
        if any(len(set(g) - set(table.index)) for g in questions["games"]):
            raise ValueError(f"{part}: a question has games without features")
        proba = model.probabilities(table.to_numpy(), table.index.to_numpy(), questions)
        pred = score_rank(proba)
        parts.append(pd.DataFrame({"question_id": questions["question_id"], "pred_rank": [RANKS[r] for r in pred]}))
        base = pd.read_parquet(work / "rankvec_p12_predictions.parquet").set_index("question_id")
        report[part] = {"hd1_games": int(len(hd1)),
                        "changed_vs_p12": int((base.loc[questions["question_id"], "rank_id"].to_numpy() != pred).sum())}
    result = pd.concat(parts, ignore_index=True).set_index("question_id").loc[template.question_id].reset_index()
    if len(result) != 800 or not result.pred_rank.isin(RANKS).all():
        raise ValueError("bad submission")
    output = ROOT / "submissions" / "rank" / f"ahc-p12-komi{arm}_{job}_rank.csv"
    result.to_csv(output, index=False, lineterminator="\r\n")
    print("wrote", output, json.dumps(report))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["tables", "evaluate", "export"])
    parser.add_argument("--job", default="local")
    parser.add_argument("--arm", default="hc", choices=["h", "hc"])
    args = parser.parse_args()
    {"tables": tables, "evaluate": evaluate}.get(args.command, lambda: export(args.job, args.arm))()
    return 0


if __name__ == "__main__":
    sys.exit(main())
