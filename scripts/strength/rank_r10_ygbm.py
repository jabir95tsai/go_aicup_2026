"""Add the 839169 yGBM features to the frozen 840005 R10-D features."""

import os
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd

from go_aicup.tasks.strength.cvec2_policy import full_pool, official
from scripts.strength import rank_ogs, rank_prod_a0_fusion


def main():
    started = monotonic()
    job = os.environ["SLURM_JOB_ID"]
    run = Path("outputs/rank/humansl/cvec2_policy/full_pool/840005")
    arm = "A/R10-D"
    spec = full_pool.read_json(run / "full_pool.json")["settings"]
    manifest = full_pool.read_json(run / "complete.json")
    full_pool.all_report.verify_files(run, {
        name: manifest["files"][name]
        for name in ("full_pool.json", "draws.npz", f"{arm}/fit.npy", f"{arm}/head.npz")
    })
    official_dir = run / "official"
    train_dir = Path(spec["source"]) / "cpu"
    games = pd.concat(
        [pd.read_parquet(train_dir / f"{group}.parquet") for group in full_pool.GROUPS],
        ignore_index=True,
    )
    _, questions, test_base, _ = official.dataset(Path(spec["planes"]))
    test_dir = official_dir / arm
    full_pool.all_report.verify_files(
        test_dir, {"fit.npy": full_pool.read_json(test_dir / "features.json")["files"]["fit.npy"]},
    )
    x_train = np.load(run / arm / "fit.npy")
    x_test = np.load(test_dir / "fit.npy")
    assert x_train.shape == (len(games), 52)
    assert x_test.shape == test_base.shape == (10000, 52)

    stored = pd.read_parquet(official_dir / "r10d-l10.parquet").set_index("question_id")
    with np.load(run / arm / "head.npz") as original:
        baseline = official.predict(questions, x_test, original).set_index("question_id")
    assert baseline.pred_rank.equals(stored.loc[baseline.index, "pred_before_l10"].rename("pred_rank"))
    np.testing.assert_allclose(
        np.stack(baseline.probabilities), np.stack(stored.loc[baseline.index, "probabilities"]),
        rtol=1e-7, atol=1e-9,
    )
    print("Original R10-D probabilities and ranks reproduced.", flush=True)

    fit, valid_u, h_fit, h_valid_u = rank_prod_a0_fusion.train_data()
    eligible = valid_u.handicap.le(1).to_numpy()
    pool = pd.concat([fit, valid_u.loc[eligible]], ignore_index=True)
    h_pool = np.vstack([h_fit, h_valid_u[eligible]])
    index = pd.Index(pool.game_index)
    game_order = pd.Index(games.game_index)
    order = index.get_indexer(game_order)
    assert index.is_unique and game_order.is_unique and len(index) == len(game_order) and (order >= 0).all()
    for col in ("player_id", "rank_id", "handicap", "role"):
        np.testing.assert_array_equal(pool[col].to_numpy()[order], games[col].to_numpy())
    y_train, y_test, _ = rank_ogs.yugen_oof(pool, h_pool, test_base)
    correlation = float(np.corrcoef(y_train[:, 0], pool.rank_id)[0, 1])
    assert abs(correlation - 0.9037541607682416) < 1e-6
    assert y_train.shape == (len(games), 2) and y_test.shape == (len(x_test), 2)
    assert np.isfinite(y_train).all() and np.isfinite(y_test).all()
    print(f"yGBM OOF correlation {correlation:.9f}, elapsed {monotonic() - started:.1f}s.", flush=True)

    draws = []
    with np.load(run / "draws.npz") as saved:
        for k in full_pool.K_BUCKETS:
            ids = saved[f"games_{k}"]
            rows = game_order.get_indexer(ids.ravel()).reshape(ids.shape)
            labels = saved["labels"][saved["k"] == k]
            assert rows.shape == (20000, k) and (rows >= 0).all()
            np.testing.assert_array_equal(games.rank_id.to_numpy()[rows[:, 0]], labels)
            draws.append((k, rows, labels))

    out = run / "r10_ygbm" / job
    out.mkdir(parents=True, exist_ok=True)
    full_pool.save_head(out / "ygbm.npz", {
        "game_index": games.game_index.to_numpy(), "fit": y_train[order], "test": y_test,
    })
    head = full_pool.fit_head(np.column_stack([x_train, y_train[order]]), draws)
    full_pool.save_head(out / "head.npz", head)
    prediction = official.predict(questions, np.column_stack([x_test, y_test]), head)
    name = "R10-D+yGBM"
    official.TAGS[name] = "r10y-l10"
    official.REFERENCES = ("840005-cva-l10.csv", "839169-ygbm-l10.csv")
    official.publish({name: [prediction]}, out, job)
    print(f"Total elapsed {monotonic() - started:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
