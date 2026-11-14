"""Is the stored per-move evidence used badly, or is it simply all there is? (RESULTS 2026-11-14)

The base turns a game into the plain mean over moves of log p(played move | rank r) (C v2). Two
proposals keep HumanSL frozen and change only that step: learned per-move weights (A) and a
sequence model over the moves (B). Both read nothing but the per-move log p already stored by
S2 (outputs/rank/humansl/permove, HD0 games), so both are tested here without new extraction.

acf      serial dependence of the per-move rank evidence inside a game. u_t = slope of log p
         over the 10 rank hypotheses (positive = the move is more likely for strong players);
         same-side lags 1..10, demeaned per game and side, against a within-game shuffle.
         design effect = 1 + 2 * sum(rho - rho_shuffled). A sequence model with a persistent
         state (B, variant 3) only has something to model if this is clearly above 1.
weights  single-game rank readers on ONE side's moves, fitted on pool games (Valid-S + train20,
         70% of the players), direction / selection on the other 30%, scored on Valid-U games:
           mean          10 mean log p (the base's statistic)
           vote          mean over moves of softmax_r(log p)          (S2 variant, for the
                         in-pool vs Valid-U comparison)
           bin_spread    moves split in 4 by the spread of log p over ranks; mean per bin (40)
           bin_surprise  ... by the rank-average log p (40)
           bin_ply       ... by ply < 40 / < 100 / < 180 / later (40)
           bin_all       3 spread x 3 surprise bins (90)
           attn          alpha_t = softmax_t f(spread, surprise, ply); one weight per move
                         shared by the ranks (proposal A as written)
           deepsets      mean_t phi(log p_t, spread, surprise, ply) next to the plain mean
                         (any per-move transform: clipping, mixtures, error tolerance)
         The bin arms are linear supersets of rank-shared bin weights. The weight inputs are
         rank-agnostic scalars, no board, so a reader cannot identify the owner. The opponent
         side decides: its moves come from a different person in every game (about 85k
         independent fitting games) and carry almost no owner bias, which is where "the evidence
         is used badly" would show.
Pre-registered (opponent side): the arm with the highest hold-out mid d' (A/1D, 1D/2D, 2D/3D,
single game) is the chosen arm; if its Valid-U mid d' is >= 1.10 x mean's, it goes to the full
G1 (swap into the 52-dim B2 input, E1 / E2, five seeds). Otherwise A and B are closed for the
stored likelihoods. Design effect < 1.15 on both sides closes B's persistent-state variant.

    python -m scripts.strength.rank_move_weights acf
    python -m scripts.strength.rank_move_weights weights [--epochs 6]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from go_aicup.metrics import score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.v2.aggregate import official_scores

PERMOVE = ROOT / "outputs" / "rank" / "humansl" / "permove"
OUT = ROOT / "outputs" / "rank" / "humansl" / "move_weights"
RUNS = ("c_valid_hd0", "c_t20_hd0")
LAGS, SEED, HOLDOUT = 10, 2027, 0.3
MID = ((3, 4), (4, 5), (5, 6))
SIDES = {"opp": 1, "own": 0}
PLY_EDGES = (40, 100, 180)
GATE = 1.10


def shards():
    for run in RUNS:
        paths = sorted(p for p in (PERMOVE / run).glob("shard_*") if (p / "meta.json").exists())
        if len(paths) != 8:
            raise ValueError(f"{run}: {len(paths)} of 8 shards")
        for shard in paths:
            yield (pd.read_parquet(shard / "games.parquet"), np.load(shard / "done.npy"),
                   np.load(shard / "move_slot.npy").astype(np.int64), np.load(shard / "move_ply.npy"),
                   np.load(shard / "move_logp.npy").astype(np.float32))


# ------------------------------------------------------------------ acf
def acf() -> None:
    contrast = np.arange(10) - 4.5
    contrast /= np.linalg.norm(contrast)
    rng = np.random.default_rng(SEED)
    stats = {s: {n: {"num": np.zeros(LAGS), "cnt": np.zeros(LAGS), "num0": np.zeros(LAGS), "den": 0.0, "n": 0}
                 for n in ("slope", "surprise")} for s in SIDES}
    for _, _, slot, _, logp in shards():
        series = {"slope": logp.astype(np.float64) @ contrast, "surprise": logp.astype(np.float64).mean(axis=1)}
        for side, parity in SIDES.items():
            idx = np.flatnonzero(slot % 2 == parity)
            ss = slot[idx]
            shuffled = np.lexsort((rng.random(len(idx)), ss))      # within-game shuffle, slots stay grouped
            for name, values in series.items():
                u = values[idx]
                count = np.bincount(ss)
                u = u - (np.bincount(ss, weights=u) / np.maximum(count, 1))[ss]
                acc = stats[side][name]
                acc["den"] += float((u * u).sum())
                acc["n"] += len(u)
                u0 = u[shuffled]
                for lag in range(1, LAGS + 1):
                    same = ss[:-lag] == ss[lag:]
                    acc["num"][lag - 1] += float((u[:-lag] * u[lag:])[same].sum())
                    acc["num0"][lag - 1] += float((u0[:-lag] * u0[lag:])[same].sum())
                    acc["cnt"][lag - 1] += int(same.sum())
    result = {}
    for side in SIDES:
        result[side] = {}
        for name, acc in stats[side].items():
            var = acc["den"] / acc["n"]
            rho, rho0 = acc["num"] / acc["cnt"] / var, acc["num0"] / acc["cnt"] / var
            result[side][name] = {"moves": acc["n"], "rho": [round(float(v), 4) for v in rho],
                                  "rho_shuffled": [round(float(v), 4) for v in rho0],
                                  "design_effect": round(float(1 + 2 * (rho - rho0).sum()), 4)}
            print(side, name, json.dumps(result[side][name]), flush=True)
    result["drop_persistent_state"] = bool(max(result[s]["slope"]["design_effect"] for s in SIDES) < 1.15)
    print(json.dumps({"drop_persistent_state": result["drop_persistent_state"]}))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "acf.json").write_text(json.dumps(result, indent=1))


# -------------------------------------------------------------- weights
def load_side(parity: int, meta: pd.DataFrame) -> dict:
    """Moves of one side of every HD0 game in meta: g (game row, sorted), ply, logp; per game rank/player/set."""
    parts, games, offset = [], [], 0
    for table, done, slot, ply, logp in shards():
        gi = table["game_index"].to_numpy()
        keep = done & np.isin(gi, meta.index)
        local = np.cumsum(keep) - 1
        row = slot // 2
        take = keep[row] & (slot % 2 == parity)
        parts.append((local[row[take]] + offset, ply[take], logp[take]))
        games.append(gi[keep])
        offset += int(keep.sum())
    games = np.concatenate(games)
    if len(np.unique(games)) != len(games):
        raise ValueError("duplicate games")
    info = meta.loc[games]
    return {"g": np.concatenate([p[0] for p in parts]), "ply": np.concatenate([p[1] for p in parts]).astype(np.float32),
            "logp": np.concatenate([p[2] for p in parts]), "games": len(games),
            "rank": info["rank_id"].to_numpy(), "player": info["player_id"].astype(str).to_numpy(),
            "color": info["player_color"].to_numpy(), "set": info["set"].to_numpy()}


def split_players(meta: pd.DataFrame) -> pd.DataFrame:
    """set: 'train' / 'hold' for pool players (rank-stratified, HOLDOUT share held out), 'U' for Valid-U."""
    pool = meta[meta["pool"]]
    players = pool.groupby(pool["player_id"].astype(str))["rank_id"].first().sort_index()
    rng = np.random.default_rng(SEED)
    hold = set()
    for _, ids in players.groupby(players):
        order = rng.permutation(len(ids))
        hold |= set(ids.index[order[: int(round(HOLDOUT * len(ids)))]])
    which = np.where(meta["player_id"].astype(str).isin(hold), "hold", "train")
    return meta.assign(set=np.where(meta["pool"], which, "U"))


def game_mean(values: np.ndarray, g: np.ndarray, games: int, weight: np.ndarray | None = None) -> np.ndarray:
    """[games, C] sum over a game's moves of values (* weight) / the game's move count."""
    count = np.bincount(g, minlength=games)
    out = np.empty((games, values.shape[1]), np.float64)
    for c in range(values.shape[1]):
        column = values[:, c] if weight is None else values[:, c] * weight
        out[:, c] = np.bincount(g, weights=column, minlength=games)
    return out / count[:, None]


def binned(data: dict, bins: np.ndarray, count: int) -> np.ndarray:
    return np.hstack([game_mean(data["logp"], data["g"], data["games"], (bins == b).astype(np.float64))
                      for b in range(count)])


def linear_features(data: dict) -> dict:
    logp, train_move = data["logp"], (data["set"] == "train")[data["g"]]
    spread, surprise = logp.std(axis=1), logp.mean(axis=1)
    data["spread"], data["surprise"] = spread, surprise
    q = lambda v, n: np.quantile(v[train_move][::7], np.arange(1, n) / n)        # thresholds from fitting games
    z = logp - logp.max(axis=1, keepdims=True)
    post = np.exp(z)
    post /= post.sum(axis=1, keepdims=True)
    s3, m3 = np.searchsorted(q(spread, 3), spread), np.searchsorted(q(surprise, 3), surprise)
    return {"mean": game_mean(logp, data["g"], data["games"]),
            "vote": game_mean(post, data["g"], data["games"]),
            "bin_spread": binned(data, np.searchsorted(q(spread, 4), spread), 4),
            "bin_surprise": binned(data, np.searchsorted(q(surprise, 4), surprise), 4),
            "bin_ply": binned(data, np.searchsorted(PLY_EDGES, data["ply"], side="right"), 4),
            "bin_all": binned(data, s3 * 3 + m3, 9)}


def lr_logits(x: np.ndarray, data: dict) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    train = data["set"] == "train"
    scaler = StandardScaler().fit(x[train])
    model = LogisticRegression(C=1.0, max_iter=3000).fit(scaler.transform(x[train]), data["rank"][train])
    return model.decision_function(scaler.transform(x))


def net_logits(arm: str, data: dict, mean_x: np.ndarray, epochs: int) -> np.ndarray:
    """attn / deepsets trained on the fitting games; the epoch with the lowest hold-out NLL is kept."""
    import torch
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sets, rank, games = data["set"], data["rank"], data["games"]
    train_move = (sets == "train")[data["g"]]
    scalars = np.stack([data["spread"], data["surprise"], data["ply"] / 100.0], axis=1)
    centred = data["logp"] - data["surprise"][:, None]
    inputs = np.hstack([centred, scalars]).astype(np.float32)
    mu, sd = inputs[train_move].mean(0), inputs[train_move].std(0) + 1e-6
    inputs = torch.as_tensor((inputs - mu) / sd, device=device)
    logp = torch.as_tensor(data["logp"], device=device)
    g_mu, g_sd = mean_x[sets == "train"].mean(0), mean_x[sets == "train"].std(0)
    g_mu, g_sd = (torch.as_tensor(v, dtype=torch.float32, device=device) for v in (g_mu, g_sd))
    start = np.searchsorted(data["g"], np.arange(games + 1))
    target = torch.as_tensor(rank, device=device)

    if arm == "attn":
        body = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
        torch.nn.init.zeros_(body[2].weight)
        torch.nn.init.zeros_(body[2].bias)          # starts as the plain mean
        head = torch.nn.Linear(10, 10)
    else:
        body = torch.nn.Sequential(torch.nn.Linear(13, 32), torch.nn.ReLU(), torch.nn.Linear(32, 16), torch.nn.ReLU())
        head = torch.nn.Linear(26, 10)
    body, head = body.to(device), head.to(device)
    optimiser = torch.optim.Adam([*body.parameters(), *head.parameters()], lr=2e-3, weight_decay=1e-4)

    def forward(batch: np.ndarray) -> "torch.Tensor":
        lengths = start[batch + 1] - start[batch]
        moves = torch.as_tensor(np.concatenate([np.arange(start[b], start[b + 1]) for b in batch]), device=device)
        group = torch.as_tensor(np.repeat(np.arange(len(batch)), lengths), device=device)
        count = torch.as_tensor(lengths, dtype=torch.float32, device=device)[:, None]
        plain = torch.zeros(len(batch), 10, device=device).index_add_(0, group, logp[moves]) / count
        if arm == "attn":
            score = body(inputs[moves][:, 10:]).squeeze(1)
            top = torch.full((len(batch),), -1e9, device=device).scatter_reduce(0, group, score, "amax")
            weight = torch.exp(score - top[group])
            weight = weight / torch.zeros(len(batch), device=device).index_add_(0, group, weight)[group]
            pooled = torch.zeros(len(batch), 10, device=device).index_add_(0, group, weight[:, None] * logp[moves])
            return head((pooled - g_mu) / g_sd)
        pooled = torch.zeros(len(batch), 16, device=device).index_add_(0, group, body(inputs[moves])) / count
        return head(torch.cat([(plain - g_mu) / g_sd, pooled], dim=1))

    def predict(rows: np.ndarray) -> "torch.Tensor":
        with torch.no_grad():
            return torch.cat([forward(rows[i:i + 4096]) for i in range(0, len(rows), 4096)])

    train_rows, hold_rows = np.flatnonzero(sets == "train"), np.flatnonzero(sets == "hold")
    rng = np.random.default_rng(SEED)
    best, kept = np.inf, None
    for epoch in range(epochs):
        order = rng.permutation(train_rows)
        for i in range(0, len(order), 512):
            batch = np.sort(order[i:i + 512])
            loss = torch.nn.functional.cross_entropy(forward(batch), target[batch])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        nll = float(torch.nn.functional.cross_entropy(predict(hold_rows), target[hold_rows]))
        print(arm, "epoch", epoch, "hold_nll", round(nll, 4), flush=True)
        if nll < best:
            best, kept = nll, predict(np.arange(games)).cpu().numpy()
    return kept


def d_prime(z_ref, y_ref, z, y, pairs=MID) -> float:
    """Mean over rank pairs of d' on z along the Fisher direction estimated on (z_ref, y_ref)."""
    out = []
    for a, b in pairs:
        ra, rb = z_ref[y_ref == a], z_ref[y_ref == b]
        within = 0.5 * (np.cov(ra.T) + np.cov(rb.T)) + 1e-6 * np.eye(z.shape[1])
        w = np.linalg.solve(within, rb.mean(0) - ra.mean(0))
        pa, pb = z[y == a] @ w, z[y == b] @ w
        out.append((pb.mean() - pa.mean()) / np.sqrt(0.5 * (pa.var() + pb.var())))
    return float(np.mean(out))


def k_means(z: np.ndarray, data: dict, which: str, k: int, draws: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Question-like means of z over k games of one (player, colour) of a set; (means, rank, player)."""
    rows = np.flatnonzero(data["set"] == which)
    table = pd.DataFrame({"row": rows, "player": data["player"][rows], "color": data["color"][rows]})
    rng = np.random.default_rng(SEED + k)
    means, ranks, players = [], [], []
    for (player, _), group in table.groupby(["player", "color"]):
        r = group["row"].to_numpy()
        if len(r) < k:
            continue
        for _ in range(draws):
            means.append(z[rng.choice(r, k, replace=False)].mean(axis=0))
            ranks.append(data["rank"][r[0]])
            players.append(player)
    return np.stack(means), np.array(ranks), np.array(players)


def report(z: np.ndarray, data: dict) -> dict:
    sets, rank = data["set"], data["rank"]
    out = {}
    for which in ("hold", "U"):
        m = sets == which
        zz = z[m] - z[m].max(axis=1, keepdims=True)
        logq = zz - np.log(np.exp(zz).sum(axis=1, keepdims=True))
        out[which] = {"games": int(m.sum()),
                      "nll": round(float(-logq[np.arange(m.sum()), rank[m]].mean()), 4),
                      "official": round(float(official_scores(score_rank(np.exp(logq)), rank[m]).mean()), 4)}
    hold = sets == "hold"
    out["hold"]["mid_d"] = round(d_prime(z[sets == "train"], rank[sets == "train"], z[hold], rank[hold]), 4)
    out["U"]["mid_d"] = round(d_prime(z[hold], rank[hold], z[sets == "U"], rank[sets == "U"]), 4)
    out["U"]["all_d"] = round(d_prime(z[hold], rank[hold], z[sets == "U"], rank[sets == "U"],
                                      tuple((r, r + 1) for r in range(9))), 4)
    for k in (5, 20):
        ref, ref_y, _ = k_means(z, data, "hold", k)
        got, got_y, _ = k_means(z, data, "U", k)
        out["U"][f"mid_d_k{k}"] = round(d_prime(ref, ref_y, got, got_y), 4)
    return out


def bootstrap_ratio(z_arm, z_mean, data, draws: int = 300) -> list[float]:
    """Player bootstrap (Valid-U) of mid d'(arm) / mid d'(mean), single game; directions fixed on hold-out."""
    sets, rank = data["set"], data["rank"]
    hold, rows = sets == "hold", np.flatnonzero(sets == "U")
    groups = pd.Series(rows).groupby(data["player"][rows]).apply(np.asarray).to_list()
    rng = np.random.default_rng(SEED)
    ratios = []
    for _ in range(draws):
        pick = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        ratios.append(d_prime(z_arm[hold], rank[hold], z_arm[pick], rank[pick])
                      / d_prime(z_mean[hold], rank[hold], z_mean[pick], rank[pick]))
    return [round(float(v), 4) for v in np.quantile(ratios, [0.025, 0.975])]


def weights(epochs: int) -> None:
    import scripts.strength.rank_cvec_v2 as cv2
    _, fit, eval_games, _ = cv2.data()
    meta = pd.concat([fit.assign(pool=True), eval_games.assign(pool=False)], ignore_index=True)
    meta = split_players(meta[meta["handicap"] == 0]).set_index("game_index")
    if set(meta.loc[meta["set"] == "U", "player_id"]) & set(meta.loc[meta["set"] != "U", "player_id"]):
        raise ValueError("Valid-U players in the pool")
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    for side, parity in SIDES.items():
        data = load_side(parity, meta)
        counts = {s: int((data["set"] == s).sum()) for s in ("train", "hold", "U")}
        print(side, "games", counts, "moves", len(data["g"]), flush=True)
        arms = linear_features(data)
        logits = {name: lr_logits(x, data) for name, x in arms.items()}
        for arm in ("attn", "deepsets"):
            logits[arm] = net_logits(arm, data, arms["mean"], epochs)
        table = {name: report(z, data) for name, z in logits.items()}
        for name, row in table.items():
            row["hold_ratio"] = round(row["hold"]["mid_d"] / table["mean"]["hold"]["mid_d"], 4)
            row["U_ratio"] = round(row["U"]["mid_d"] / table["mean"]["U"]["mid_d"], 4)
            print(side, name, json.dumps(row), flush=True)
        chosen = max((n for n in table if n != "mean"), key=lambda n: table[n]["hold"]["mid_d"])
        result[side] = {"games": counts, "arms": table, "chosen": chosen,
                        "chosen_U_ratio": table[chosen]["U_ratio"],
                        "chosen_U_ratio_bootstrap_95": bootstrap_ratio(logits[chosen], logits["mean"], data),
                        "pass": bool(table[chosen]["U_ratio"] >= GATE)}
        print(side, json.dumps({k: result[side][k] for k in ("chosen", "chosen_U_ratio",
                                                              "chosen_U_ratio_bootstrap_95", "pass")}), flush=True)
        del data, arms, logits
    result["decision"] = "G1" if result["opp"]["pass"] else "close A and B for the stored likelihoods"
    print(json.dumps({"decision": result["decision"]}))
    (OUT / "weights.json").write_text(json.dumps(result, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["acf", "weights"])
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()
    if args.command == "acf":
        acf()
    else:
        weights(args.epochs)


if __name__ == "__main__":
    main()
