"""Official added-set style term of another encoder + B head (2026-11-17).

For each part, the per-move vectors of --run on the official board caches of the predict2 directory
are aggregated by --head and turned into the style term w * cosine (w, lambda, mu, opening and category
of the hung-seqB directory). Check first: the same code path with the 9,600-step encoder and head B
s2026 must reproduce the seqB style term (max abs diff <= 1e-4, identical top-1); otherwise stop.
Output: outputs/pid/day0_1112/{part}2_d4_9600_{tag}/ in the format combine.py reads.

    python -m scripts.identity.pid2_style_export --run pid_e2e_scratch32_9600/step9600 \\
        --head outputs/pid/ge2e/pid_e2e_scratch32_9600/step9600/best.pt --tag e2e9600
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from go_aicup.data.board_cache import BoardCache
from go_aicup.data.preprocess import load_store
from go_aicup.paths import PROCESSED, ROOT
from go_aicup.tasks.identity import twocolor
from go_aicup.tasks.identity.assign import top5
from go_aicup.tasks.identity.seqagg import MoveStore, game_vectors, load_head, move_vectors
from scripts.identity.pid2_seqagg import load_encoder

STORE = PROCESSED / "identity" / "artifacts" / "day0_1112" / "store"
DAY0 = ROOT / "outputs" / "pid" / "day0_1112"
REF_RUN = "pid_ge2e_v4_d4_hd_9600"
REF_HEAD = ROOT / "outputs" / "pid" / "pid2_seqagg" / "head_B" / "best.pt"


def head_vectors(run: str, head_file: Path, part: str) -> np.ndarray:
    """[pool games + question games, 256] head game vectors in store order."""
    model, config, device = load_encoder(run)
    if config.get("sampling", {}).get("min_ply", 20) != 0:
        raise ValueError("ply positions assume min_ply 0")
    head, step = load_head(head_file, device)
    src = DAY0 / f"{part}2_d4_9600"
    out = []
    for name in (f"pool2_{part}", f"pid2_{part}"):
        _, _, meta_s = load_store(name, STORE, mmap=False)
        cache_dir = src / name / "cache"
        games = pd.read_parquet(cache_dir / "games.parquet")
        if len(games) != len(meta_s):
            raise ValueError(f"{name}: board cache lost games")
        rows = games["store_row"].to_numpy()
        colour = (meta_s["color"].astype(str).to_numpy()[rows] == "W").astype(np.int8)
        index = games["game_index"].to_numpy()
        vec, off = move_vectors(model, config, BoardCache(cache_dir), index, colour, games["length"].to_numpy(),
                                device, 0)
        store = MoveStore.__new__(MoveStore)
        store.vectors, store.offsets, store.games = vec.astype(np.float16), off, index
        store.first_ply = colour.astype(np.int64)
        h = np.empty((len(meta_s), 256), np.float32)
        h[rows] = game_vectors(head, store, np.arange(len(index)), device)
        out.append(h)
        print(f"{run} {part} {name}: {len(meta_s)} games (head step {step})", flush=True)
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    for part in ("public", "private"):
        src = DAY0 / f"{part}2_d4_9600_seqB"
        dst = DAY0 / f"{part}2_d4_9600_{args.tag}"
        if (dst / "parts.npz").exists():
            print(f"{dst} exists, skipping", flush=True)
            continue
        info = json.loads((src / "info.json").read_text())
        s = info["summary"]
        parts = dict(np.load(src / "parts.npz"))
        _, _, pool_meta = load_store(f"pool2_{part}", STORE, mmap=False)
        _, _, q_meta = load_store(f"pid2_{part}", STORE, mmap=False)
        players = pool_meta["player_id"].astype(str).to_numpy()
        candidates = np.array(sorted(set(players)))
        assert np.array_equal(candidates, np.load(src / "candidates.npy", allow_pickle=True).astype(str))
        gallery = [np.flatnonzero(players == c) for c in candidates]
        qids = list(dict.fromkeys(q_meta["question_id"]))
        rows_by_q = q_meta.groupby("question_id", sort=False).indices
        query = [len(pool_meta) + np.sort(rows_by_q[q]) for q in qids]
        ks = q_meta.groupby("question_id", sort=False)["num_games"].first().loc[qids].to_numpy()
        pred = pd.read_parquet(src / "predictions.parquet")
        assert pred.question_id.astype(str).tolist() == qids
        rest = parts["opening"] + s["mu"] * parts["category"]

        ref = s["w"] * twocolor.style_scores(head_vectors(REF_RUN, REF_HEAD, part), gallery, query)
        diff = float(np.abs(ref - parts["style"]).max())
        same = bool(np.array_equal(top5(rest + s["lambda"] * ref)[:, 0], top5(np.load(src / "scores.npy"))[:, 0]))
        print(f"{part}: reference reproduces seqB style, max abs diff {diff:.2e}, top-1 identical {same}", flush=True)
        if diff > 1e-4 or not same:
            raise SystemExit("reference path does not reproduce the seqB style term")

        style = s["w"] * twocolor.style_scores(head_vectors(args.run, args.head, part), gallery, query)
        scores = rest + s["lambda"] * style
        ranked = candidates[top5(scores)]
        dst.mkdir(parents=True, exist_ok=False)
        np.savez(dst / "parts.npz", opening=parts["opening"], style=style, category=parts["category"])
        np.save(dst / "scores.npy", scores)
        np.save(dst / "candidates.npy", candidates)
        table = pd.DataFrame({"question_id": qids, "num_games": ks})
        for j in range(5):
            table[f"top{j + 1}"] = ranked[:, j]
        table.to_parquet(dst / "predictions.parquet", index=False)
        changed = ranked[:, 0] != pred["top1"].to_numpy()
        (dst / "info.json").write_text(json.dumps({
            "run": args.run, "head": str(args.head), "reference_max_abs_diff": diff,
            "top1_changed_vs_seqB": int(changed.sum()),
            "by_k": {int(k): int(changed[ks == k].sum()) for k in sorted(set(ks))}, "summary": s}, indent=2))
        print(f"{part}: written {dst}, top-1 changed vs seqB {int(changed.sum())}", flush=True)


if __name__ == "__main__":
    main()
