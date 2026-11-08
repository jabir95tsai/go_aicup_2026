"""Two cheap B2 extensions on the P1+P2 setup (RESULTS 2026-11-08), CPU only.

A  game outcome proxies: who moved last (in a resigned game usually the winner),
   length, end-of-game stone share and capture statistics from a full replay
   (features/game_stats.py), plus "moved last" x opponent HumanSL features, so a
   question can carry "beats opponents who play like rank r" (Elo-like evidence the
   per-move likelihoods do not hold).
B  question-level dispersion: the std over the question's games of the 60 AH+C
   features next to their mean; and a small MLP head on mean + std.

Every arm uses the same synthetic-question draws as B2 (humansl.RankModel._fit_synthetic,
reproduced here so all arms share them), fitted on Valid-S HD<=1 with the HD1 flags,
evaluated on E2 (and E1) over five seeds; gate as P1/P2: E2 Equal-k >= +0.005 and every
k >= -0.005 against the control arm.

    python scripts/rank_b2_extensions.py stats       # replay statistics for Valid-S/U and test games
    python scripts/rank_b2_extensions.py evaluate
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from go_aicup.data.store import GameStore
from go_aicup.features import game_stats as gs
from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.paths import PROCESSED, ROOT
from go_aicup.tasks.strength.v2.aggregate import official_scores

import humansl_hd_filter as hf

OUT = ROOT / "outputs" / "rank" / "humansl" / "b2_extensions"
CACHE = PROCESSED / "boards" / "strength_v1"
TEST_STORE = ROOT / "outputs" / "rank_test_ahc_713770" / "store"
TEST_WORK = ROOT / "outputs" / "rank_test_ahc_713770" / "rank"
SYNTHETIC = 20000
OUTCOME = ["me_last", "length", "stone_share_end", "me_lost_stones", "me_cap_stones", "op_cap_stones",
           "me_cap_big", "op_cap_big"]
OPP_H = slice(25, 40)   # x = [A 10 | own H 15 | opp H 15 | C 20]


class _Store:
    """The two attributes and len() that game_stats.to_frame reads."""

    def __init__(self, player_color, lengths):
        self.player_color, self.lengths = player_color, lengths

    def __len__(self) -> int:
        return len(self.lengths)


def replay_frame(moves, offsets, lengths, player_color) -> pd.DataFrame:
    out = np.zeros((len(offsets), 2, gs.NF), dtype=np.float32)
    gs._replay(np.ascontiguousarray(moves), offsets.astype(np.int64), lengths.astype(np.int64),
               gs.NB, gs.NB_CNT, gs.EARLY_PLIES, out)
    return gs.to_frame(_Store(player_color, lengths), out)


def stats() -> None:
    games, *_ = hf.setup()
    meta = pd.read_parquet(CACHE / "games.parquet").set_index("game_index").loc[games["game_index"]].reset_index()
    local, length = meta["local_offset"].to_numpy(), meta["length"].to_numpy()
    seq = [None] * len(meta)
    for shard, rows in meta.groupby("shard_id").indices.items():
        moves = np.load(CACHE / f"shard_{shard:05d}_moves.npy", mmap_mode="r")
        for i in rows:
            seq[i] = np.asarray(moves[local[i]:local[i] + length[i]])
    lengths = np.array([len(s) for s in seq])
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    frame = replay_frame(np.concatenate(seq).astype(np.int16), offsets, lengths,
                         games["player_color"].to_numpy().astype(np.int8))
    frame.insert(0, "game_index", games["game_index"].to_numpy())
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUT / "stats_train.parquet", index=False)
    print("train", len(frame), frame[OUTCOME].mean().round(4).to_dict(), flush=True)
    for part in ("public", "private"):
        store = GameStore(f"rank_{part}", TEST_STORE)
        sums = gs.replay(store)
        f = gs.to_frame(store, sums)
        cache = pd.read_parquet(TEST_WORK / f"rank_{part}" / "cache" / "games.parquet")
        f = f.iloc[cache["store_row"].to_numpy()].reset_index(drop=True)
        f.insert(0, "game_index", cache["game_index"].to_numpy())
        f.to_parquet(OUT / f"stats_test_{part}.parquet", index=False)
        print(part, len(f), f[OUTCOME].mean().round(4).to_dict(), flush=True)


def draws(fit: pd.DataFrame, seed: int) -> dict:
    """B2's synthetic draws: per k, (row index arrays [SYNTHETIC, k], labels)."""
    rng = np.random.default_rng(seed)
    by_player = list(fit.groupby(fit["player_id"].astype(str)).indices.values())
    y = fit["rank_id"].to_numpy()
    out = {}
    for k in K_BUCKETS:
        players = [rows for rows in by_player if len(rows) >= k]
        picks = rng.integers(0, len(players), SYNTHETIC)
        rows = np.stack([rng.choice(players[p], k, replace=False) for p in picks])
        out[k] = (rows, np.array([y[players[p][0]] for p in picks]))
    return out


def question_x(x: np.ndarray, rows: np.ndarray, std_cols: slice | None) -> np.ndarray:
    m = x[rows].mean(axis=1)
    if std_cols is None:
        return m
    return np.hstack([m, x[rows][:, :, std_cols].std(axis=1)])


def fit_arm(x, d, std_cols, head, seed):
    models = {}
    for k in K_BUCKETS:
        rows, labels = d[k]
        features = question_x(x, rows, std_cols)
        scaler = StandardScaler().fit(features)
        if head == "lr":
            model = LogisticRegression(C=1.0, max_iter=5000)
        else:
            model = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, early_stopping=True,
                                  max_iter=300, random_state=seed)
        models[k] = (scaler, model.fit(scaler.transform(features), labels))
    return models


def predict(models, x_eval, eval_index, questions, std_cols) -> np.ndarray:
    return official_scores(score_rank(predict_proba(models, x_eval, eval_index, questions, std_cols)),
                           questions["rank_id"].to_numpy())


def predict_proba(models, x_eval, eval_index, questions, std_cols) -> np.ndarray:
    pos = pd.Series(np.arange(len(eval_index)), index=eval_index)
    out = np.empty((len(questions), 10))
    k = questions["k"].to_numpy()
    for b in K_BUCKETS:
        mask = np.flatnonzero(k == b)
        if not len(mask):
            continue
        rows = np.stack([pos.loc[list(questions["games"].iloc[i])].to_numpy() for i in mask])
        scaler, model = models[b]
        out[mask] = model.predict_proba(scaler.transform(question_x(x_eval, rows, std_cols)))
    return out


def outcome_block(frame: pd.DataFrame, game_index, x: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """centre: opponent-H means of the fit games (the same shift for fit, eval and test)."""
    f = frame.set_index("game_index").loc[game_index]
    base = f[OUTCOME].to_numpy(float).copy()
    base[:, 1] /= 100.0
    last = f["me_last"].to_numpy(float)[:, None]
    return np.hstack([base, last * (x[:, OPP_H] - centre)])


def all_stats_block(frame: pd.DataFrame, games: pd.DataFrame) -> np.ndarray:
    f = frame.set_index("game_index").loc[games["game_index"]]
    v = f.to_numpy(float)
    return np.nan_to_num(v)


def evaluate() -> None:
    games, fit, eval_games, x_fit, x_eval = hf.setup()
    sets = {k: v for k, v in hf.question_sets(games, rebuild=False).items() if k in ("E1", "E2")}
    frame = pd.read_parquet(OUT / "stats_train.parquet")
    hd01 = (fit["handicap"] <= 1).to_numpy()
    fit_h = fit[hd01].reset_index(drop=True)
    base_fit = np.hstack([x_fit, hf.hd1_columns(fit)])[hd01]
    base_eval = np.hstack([x_eval, hf.hd1_columns(eval_games)])
    centre = x_fit[hd01][:, OPP_H].mean(axis=0)
    oc_fit = outcome_block(frame, fit_h["game_index"], x_fit[hd01], centre)
    oc_eval = outcome_block(frame, eval_games["game_index"], x_eval, centre)
    st_fit, st_eval = all_stats_block(frame, fit_h), all_stats_block(frame, eval_games)
    core = slice(0, 60)
    arms = {  # name: (x_fit, x_eval, std_cols, head)
        "control": (base_fit, base_eval, None, "lr"),
        "A_outcome": (np.hstack([base_fit, oc_fit]), np.hstack([base_eval, oc_eval]), None, "lr"),
        "A_allstats": (np.hstack([base_fit, oc_fit, st_fit]), np.hstack([base_eval, oc_eval, st_eval]), None, "lr"),
        "B_std": (base_fit, base_eval, core, "lr"),
        "B_mlp": (base_fit, base_eval, core, "mlp"),
    }
    eval_index = eval_games["game_index"].to_numpy()
    result = {"arms": list(arms), "seeds": {}}
    scores_2027 = {}
    for seed in hf.SEEDS:
        d = draws(fit_h, seed)
        entry = {}
        scores = {name: {} for name in arms}
        for name, (xf, xe, std_cols, head) in arms.items():
            models = fit_arm(xf, d, std_cols, head, seed)
            for set_name, q in sets.items():
                scores[name][set_name] = predict(models, xe, eval_index, q, std_cols)
        for set_name, q in sets.items():
            k = q["k"].to_numpy()
            ctrl = scores["control"][set_name]
            entry[set_name] = {"control": hf.by_k(ctrl, k)}
            for name in arms:
                if name == "control":
                    continue
                s = scores[name][set_name]
                entry[set_name][name] = hf.by_k(s, k) | {
                    "diff_equal_k": round(hf.equal_k(s, k) - hf.equal_k(ctrl, k), 4),
                    "diff_by_k": {str(b): round(float(s[k == b].mean() - ctrl[k == b].mean()), 4) for b in K_BUCKETS}}
                if seed == hf.SEED:
                    entry[set_name][name]["bootstrap_95"] = hf.bootstrap(ctrl, s, q)
        result["seeds"][str(seed)] = entry
        print(seed, json.dumps({n: {a: (v.get("diff_equal_k"), v.get("equal_k")) for a, v in e.items()}
                                for n, e in entry.items()}), flush=True)
    gate = {}
    for name in arms:
        if name == "control":
            continue
        e2 = result["seeds"][str(hf.SEED)]["E2"][name]
        mean = float(np.mean([result["seeds"][str(s)]["E2"][name]["diff_equal_k"] for s in hf.SEEDS]))
        gate[name] = {"diff_equal_k": e2["diff_equal_k"], "min_diff_by_k": min(e2["diff_by_k"].values()),
                      "bootstrap_95": e2["bootstrap_95"], "mean_over_seeds": round(mean, 4),
                      "pass": bool(e2["diff_equal_k"] >= hf.MIN_GAIN and min(e2["diff_by_k"].values()) >= -hf.MAX_DROP)}
    result["gate"] = gate
    print(json.dumps(gate), flush=True)
    (OUT / "result.json").write_text(json.dumps(result, indent=1))


def export(job: str) -> None:
    """LB-probe files (seed 2027 fit, as P1+P2): control must reproduce the P1+P2 test ranks."""
    import pickle
    from go_aicup.data.board_cache import RANKS
    from go_aicup.tasks.strength import humansl_predict as hp
    from go_aicup.tasks.strength.humansl import k_bucket
    games, fit, eval_games, x_fit, x_eval = hf.setup()
    frame = pd.read_parquet(OUT / "stats_train.parquet")
    hd01 = (fit["handicap"] <= 1).to_numpy()
    fit_h = fit[hd01].reset_index(drop=True)
    base_fit = np.hstack([x_fit, hf.hd1_columns(fit)])[hd01]
    centre = x_fit[hd01][:, OPP_H].mean(axis=0)
    oc_fit = outcome_block(frame, fit_h["game_index"], x_fit[hd01], centre)
    arms = {"control": (base_fit, False, None), "outcome": (np.hstack([base_fit, oc_fit]), True, None),
            "std": (base_fit, False, slice(0, 60))}
    d = draws(fit_h, hf.SEED)
    models = {name: fit_arm(xf, d, std_cols, "lr", hf.SEED) for name, (xf, _, std_cols) in arms.items()}
    with hf.PRODUCTION.open("rb") as handle:
        columns = pickle.load(handle)["info"]["columns"] + hp.HD1_COLUMNS
    template = pd.read_csv(ROOT / "data" / "raw" / "tests" / "sample_submission_rank.csv", dtype=str)
    rows = {name: [] for name in arms}
    report = {}
    for part in ("public", "private"):
        work = TEST_WORK / f"rank_{part}"
        table = hp.features(hp.a024_per_game(np.load(work / "category.npy", allow_pickle=False)),
                            hp.release_features(work / "humansl"), hp.rankvec_features([work / "rankvec"]))
        table = table.join(hp.hd1_features(work, ROOT / "data" / "raw" / "tests" / f"rank_prediction_test_{part}.csv"),
                           how="inner")
        if list(table.columns) != columns:
            raise ValueError("test feature columns differ from the P1+P2 model")
        x = table.to_numpy()
        stats_test = pd.read_parquet(OUT / f"stats_test_{part}.parquet")
        oc = outcome_block(stats_test, table.index, x[:, :60], centre)
        questions = hp.question_table(work)
        questions["k"] = questions["num_games"].map(k_bucket)
        if any(len(set(g) - set(table.index)) for g in questions["games"]):
            raise ValueError(f"{part}: a question has games without features")
        p12 = pd.read_parquet(work / "rankvec_p12_predictions.parquet").set_index("question_id") \
            .loc[questions["question_id"], "rank_id"].to_numpy()
        report[part] = {}
        for name, (_, with_outcome, std_cols) in arms.items():
            xe = np.hstack([x, oc]) if with_outcome else x
            pred = score_rank(predict_proba(models[name], xe, table.index.to_numpy(), questions, std_cols))
            report[part][f"changed_vs_p12_{name}"] = int((pred != p12).sum())
            rows[name].append(pd.DataFrame({"question_id": questions["question_id"],
                                            "pred_rank": [RANKS[r] for r in pred]}))
        if report[part]["changed_vs_p12_control"]:
            raise ValueError(f"{part}: control does not reproduce the P1+P2 submission")
    for name in ("outcome", "std"):
        result = pd.concat(rows[name], ignore_index=True).set_index("question_id").loc[template.question_id].reset_index()
        if len(result) != 800 or not result.pred_rank.isin(RANKS).all():
            raise ValueError(name)
        output = ROOT / "submissions" / "rank" / f"ahc-p12-{name}_{job}_rank.csv"
        result.to_csv(output, index=False, lineterminator="\r\n")
        print("wrote", output)
    print(json.dumps(report))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["stats", "evaluate", "export"])
    parser.add_argument("--job", default="local")
    args = parser.parse_args()
    {"stats": stats, "evaluate": evaluate}.get(args.command, lambda: export(args.job))()
    return 0


if __name__ == "__main__":
    sys.exit(main())
