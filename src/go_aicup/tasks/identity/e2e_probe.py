"""H1 zero-LR controls and fixed pool-0 monitoring; never reads sealed draws."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from go_aicup.data import cv
from go_aicup.data.board_cache import BoardCache
from go_aicup.paths import ROOT
from go_aicup.tasks.identity import data, twocolor
from go_aicup.tasks.identity.ge2e import IdentityEncoder, ge2e_loss
from go_aicup.tasks.identity.seqagg import MoveStore, game_vectors, load_head, move_vectors
from go_aicup.tasks.identity.train_ge2e import read_config, training_players
from go_aicup.tasks.strength.shared.runtime import autocast_context
from go_aicup.tasks.strength.shared.training import seed_process

BASE_RUN = "pid_ge2e_v4_d4_hd_9600"


class BenchMonitor:
    """All valid plies, fixed development draws, and frozen fusion weights."""

    def __init__(self, config, device, draws=1):
        from scripts.identity.pid2_transductive import load
        self.config, self.device = config, device
        env = load(BASE_RUN)
        self.draws = []
        for draw in range(100, 100 + draws):
            candidates, gallery, q = cv.draw_pid2(env["bank"], 0, draw)
            query = list(q["games"])
            other = twocolor.opening_scores(env["prefix"], env["hashes"], gallery, query,
                env["color"], env["populations"], env["summary"]["opening"])
            other += twocolor.category_scores(env["codes"], gallery, query, 4)
            self.draws.append((candidates, gallery, q, other))
        self.games = np.unique(np.concatenate([
            np.concatenate(g + list(q["games"])) for _, g, q, _ in self.draws]))
        meta = data.load_meta()
        self.colors = meta["player_color"].to_numpy().astype(np.int8)[self.games]
        self.lengths = meta["length"].to_numpy()[self.games]
        self.cache = BoardCache(config["paths"]["cache_dir"])
        saved = np.load(ROOT / "outputs/pid/pid2_bench" / BASE_RUN / "embeddings.npz")
        self.w = float(saved["w"])
        self.scores = {}
        self.vectors = {}
        self.per_draw = []

    def score_vectors(self, vectors):
        from scripts.identity.pid2_paired import points
        self.vectors, self.scores, self.per_draw = vectors, {}, []
        out = {}
        for name, vec in vectors.items():
            totals = {"style": [], "fused": []}
            for draw, (candidates, gallery, q, other) in enumerate(self.draws, 100):
                g = [np.searchsorted(self.games, x) for x in gallery]
                qq = [np.searchsorted(self.games, x) for x in q["games"]]
                cosine = twocolor.style_scores(vec, g, qq)
                for what, score in (("style", cosine), ("fused", other + 16 * self.w * cosine)):
                    self.scores.setdefault(f"{name}_{what}", []).append(score)
                    pts = points(score, candidates, q)
                    ks = q["k"].to_numpy()
                    row = {str(k): float(pts[ks == k].mean()) for k in (1, 2, 3, 5)}
                    row["all"] = float(pts.mean())
                    totals[what].append(row)
                    self.per_draw.append({"draw": draw, "arm": name, "what": what, **row})
            for what, rows in totals.items():
                for key in rows[0]:
                    out[f"{name}_{what}_{key}"] = float(np.mean([r[key] for r in rows]))
        self.scores = {k: np.stack(v) for k, v in self.scores.items()}
        return out

    def __call__(self, encoder, heads):
        from go_aicup.tasks.identity.train_e2e import training_mode
        vec, off = move_vectors(encoder, self.config, self.cache, self.games, self.colors,
                               self.lengths, self.device, 0)
        store = SimpleNamespace(vectors=vec.astype(np.float16), offsets=off, games=self.games,
                                first_ply=self.colors.astype(np.int64))
        store.batch = MoveStore.batch.__get__(store)
        rows = np.arange(len(self.games))
        vectors = {name: game_vectors(head, store, rows, self.device)
                   for name, head in {"A": None, **heads}.items()}
        training_mode(encoder, self.config["e2e"].get("freeze_bn", False))
        return self.score_vectors(vectors)

    def cached(self, head):
        colors = data.load_meta()["player_color"].to_numpy().astype(np.int8)
        store = MoveStore(Path(self.config["paths"]["bench_cache"]) / "bench", colors)
        pos = {int(g): i for i, g in enumerate(store.games)}
        rows = np.array([pos[int(g)] for g in self.games])
        return self.score_vectors({"A": game_vectors(None, store, rows, self.device),
                                   "B": game_vectors(head, store, rows, self.device)})


def main():
    from go_aicup.tasks.identity.train_e2e import (
        PlyBatches, joint_games, restore_encoder_optimizer, split_players, training_mode)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    out = Path(config["paths"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "report.json").exists():
        raise ValueError("probe already completed; use its report or a new output directory")
    (out / "config.json").write_text(json.dumps(config, indent=2))
    device = torch.device("cuda")
    seed_process(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    source = torch.load(config["paths"]["init_model"], map_location="cpu", weights_only=False)
    saved = torch.load(config["paths"]["init_optimizer"], map_location="cpu", weights_only=False)
    encoder = IdentityEncoder(**config["model"]).to(device)
    encoder.load_state_dict(source["model"])
    head, _ = load_head(config["paths"]["init_head"], device)
    head_source = copy.deepcopy(head.state_dict())
    monitor = BenchMonitor(config, device, draws=20)
    player_games, lengths, _, colors, _ = training_players(config)
    train_games, _, _, _, _ = split_players(config, player_games)
    if np.isin(monitor.games, np.concatenate(player_games)).any():
        raise ValueError("monitor overlaps original encoder training games")
    ref = monitor.cached(head)
    ref_vectors = {k: v.copy() for k, v in monitor.vectors.items()}
    ref_scores = {k: v.copy() for k, v in monitor.scores.items()}
    baseline = monitor(encoder, {"B": head})
    baseline_scores = {k: v.copy() for k, v in monitor.scores.items()}
    baseline_rows = list(monitor.per_draw)
    min_cos = {k: float(np.min(np.sum(twocolor.normalise(v) * twocolor.normalise(ref_vectors[k]), axis=1)))
               for k, v in monitor.vectors.items()}
    score_error = {k: float(np.max(np.abs(v - ref_scores[k]))) for k, v in baseline_scores.items()}
    # Different inference chunk boundaries can change BF16 rounding, but not the baseline materially.
    parity = min(min_cos.values()) > 0.9999 and max(
        abs(baseline[f"B_fused_{k}"] - ref[f"B_fused_{k}"]) for k in ("1", "2", "3", "5", "all")) <= 0.002
    report = {"cached": ref, "step0": baseline, "min_vector_cosine": min_cos,
              "cached_max_score_difference": score_error, "baseline_pass": parity, "controls": {}}
    print(json.dumps(report), flush=True)
    if not parity:
        (out / "report.json").write_text(json.dumps({**report, "pass": False}, indent=2))
        raise SystemExit("step 0 differs from cached baseline")
    np.savez_compressed(out / "step0_scores.npz", **baseline_scores)
    shape = tuple(config["batch"][k] for k in ("players", "games", "moves"))
    bn_buffers = {f"{n}.{k}" if n else k for n, m in encoder.named_modules()
                  if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                  for k, _ in m.named_buffers(recurse=False)}
    for freeze in (False, True):
        seed_process(config["seed"])
        encoder.load_state_dict(source["model"])
        head.load_state_dict(head_source)
        config["e2e"]["freeze_bn"] = freeze
        training_mode(encoder, freeze)
        head.train()
        params = list(encoder.parameters()) + list(head.parameters())
        optimizer = torch.optim.AdamW(params, **{**config["optimizer"], "lr": 0.0})
        restore_encoder_optimizer(optimizer, encoder, saved)
        dataset = PlyBatches(config["paths"]["cache_dir"], train_games, lengths,
            players=shape[0], games=shape[1], moves=shape[2], steps=30,
            seed=config["seed"], rank=0, start_step=0,
            max_open_shards=config["loader"]["max_open_shards"], player_color=colors,
            min_ply=0, player_d4=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=4,
            pin_memory=True, prefetch_factor=2, multiprocessing_context="spawn")
        for step, (x, plies) in enumerate(loader, 1):
            with autocast_context(config, device):
                moves = encoder(x.to(device, dtype=torch.float32, non_blocking=True))
            games = joint_games(head, moves, plies.to(device), shape)
            loss, _ = ge2e_loss(games, head.w, head.b)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite LR-zero loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, config["training"]["gradient_clip_norm"])
            optimizer.step()
        if step != 30:
            raise ValueError("incomplete LR-zero control")
        changed = [k for k, v in encoder.state_dict().items() if not torch.equal(v.cpu(), source["model"][k])]
        head_unchanged = all(torch.equal(v, head_source[k]) for k, v in head.state_dict().items())
        weights_unchanged = not (set(changed) - bn_buffers) and head_unchanged
        del loader, optimizer, moves, games, loss, x, plies
        torch.cuda.empty_cache()
        scores = monitor(encoder, {"B": head})
        errors = {k: float(np.max(np.abs(v - baseline_scores[k]))) for k, v in monitor.scores.items()}
        label = "frozen_bn" if freeze else "updating_bn"
        report["controls"][label] = {"scores": scores, "delta": {k: scores[k] - baseline[k] for k in scores},
            "weights_unchanged": weights_unchanged, "changed_buffers": changed, "max_score_difference": errors}
        np.savez_compressed(out / f"{label}_scores.npz", **monitor.scores)
        (out / f"{label}_per_draw.json").write_text(json.dumps(monitor.per_draw))
        print(json.dumps({label: report["controls"][label]}), flush=True)
    frozen = report["controls"]["frozen_bn"]
    normal = report["controls"]["updating_bn"]
    report["pass"] = bool(frozen["weights_unchanged"] and normal["weights_unchanged"]
                          and not frozen["changed_buffers"] and max(frozen["max_score_difference"].values()) <= 1e-6)
    delta = normal["delta"]
    report["freeze_bn"] = bool(delta["B_fused_all"] < -0.002 or delta["B_style_all"] < -0.002
                              or min(delta[f"B_fused_{k}"] for k in (1, 2, 3, 5)) < -0.005)
    (out / "step0_per_draw.json").write_text(json.dumps(baseline_rows))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"pass": report["pass"], "freeze_bn": report["freeze_bn"]}), flush=True)
    if not report["pass"]:
        raise SystemExit("zero-update parity failed; do not train")


if __name__ == "__main__":
    main()
