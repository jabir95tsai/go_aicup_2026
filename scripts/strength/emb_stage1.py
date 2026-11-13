"""C1 stage-1 gate (RESULTS A-051), CPU only.

B2 fitted on the Valid-S fitting games (q1000 S-fit + vs_rest S-fit / S-extra),
scored on the 1,000 S-select questions. HumanSL trunk embeddings (3,072-d) are
reduced with PCA fitted on the fitting games only.
  AH      A-024 + HumanSL mean (reference; A-044 (i) 0.7101)
  AH_Emb  AH + 128 PCA components
  H_Emb   HumanSL mean + embedding (diagnostic)
  Emb     embedding only (diagnostic)
Gate: AH_Emb - AH >= +0.01.

    python -m scripts.strength.emb_stage1
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from scripts.strength.humansl_pilot import a024_features
from scripts.strength.humansl_questions import by_k
from go_aicup.features.humansl import run_features
from go_aicup.features.humansl_torch import run_embeddings
from go_aicup.metrics import score_rank
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.humansl import SEED, RankModel
from go_aicup.tasks.strength.v2.aggregate import official_scores

OUT = ROOT / "outputs"
COMPONENTS, MIN_GAIN = 128, 0.01


def table(run, roles) -> pd.DataFrame:
    games = pd.read_parquet(run / "games.parquet")
    games = games[games["role"].isin(roles)]
    out = games.merge(run_features(run), on="game_index", validate="one_to_one")
    if len(out) != len(games):
        raise ValueError(f"{run}: HumanSL features missing")
    return out


def main() -> int:
    fit = pd.concat([table(OUT / "humansl_q1000", ["S-fit"]), table(OUT / "humansl_vs_rest", ["S-fit", "S-extra"])],
                    ignore_index=True)
    test = table(OUT / "humansl_qsel", ["S-select"])
    questions = pd.read_parquet(OUT / "humansl_qsel" / "questions.parquet")
    embedding = run_embeddings(OUT / "emb_stage1").set_index("game_index")
    for frame, name in ((fit, "fit"), (test, "S-select")):
        missing = set(frame["game_index"]) - set(embedding.index)
        if missing:
            raise ValueError(f"{name}: {len(missing)} games without embeddings")
    pca = PCA(COMPONENTS, random_state=SEED).fit(embedding.loc[fit["game_index"]].to_numpy())
    human = [c for c in fit.columns if c.startswith(("own_", "opp_"))]

    def blocks(frame):
        return {"A": a024_features(frame["game_index"]), "H": frame[human].to_numpy(),
                "Emb": pca.transform(embedding.loc[frame["game_index"]].to_numpy())}

    f, t = blocks(fit), blocks(test)
    views = {"AH": ("A", "H"), "AH_Emb": ("A", "H", "Emb"), "H_Emb": ("H", "Emb"), "Emb": ("Emb",)}
    k = questions["k"].to_numpy()
    result = {"pca_explained_variance": round(float(pca.explained_variance_ratio_.sum()), 4),
              "fit_games": len(fit), "S-select": {}}
    for name, parts in views.items():
        x_fit, x_test = np.hstack([f[p] for p in parts]), np.hstack([t[p] for p in parts])
        model = RankModel("B2").fit(fit, pd.DataFrame(), x_fit)
        pred = score_rank(model.probabilities(x_test, test["game_index"].to_numpy(), questions))
        result["S-select"][name] = by_k(official_scores(pred, questions["rank_id"].to_numpy()), k)
        print(name, result["S-select"][name], flush=True)
    gain = result["S-select"]["AH_Emb"]["equal_k"] - result["S-select"]["AH"]["equal_k"]
    result["gate"] = {"gain_AH_Emb_minus_AH": round(gain, 4), "min_gain": MIN_GAIN, "pass": bool(gain >= MIN_GAIN)}
    print(json.dumps(result["gate"]))
    (OUT / "emb_stage1" / "result.json").write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
