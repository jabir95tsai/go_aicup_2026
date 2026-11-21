"""Train paired R10 and CNN-R10 residuals with a frozen cvec2-style question head."""

import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from go_aicup.artifacts import file_hash, write_json
from go_aicup.metrics import K_BUCKETS
from go_aicup.tasks.strength.policy_large.run import parallel
from go_aicup.tasks.strength.policy_residual.epoch import epoch_batches
from go_aicup.tasks.strength.policy_residual.train import atomic_save
from scripts.strength import rank_prod_a0_fusion as prod
from scripts.strength.rank_more_players import PriorMatched, player_prior
from . import encoder, full_cache, run
from .played import H_COLUMNS


SEED = 2027
QUESTIONS_PER_K_BATCH = 8
POSITION_CHUNK = 4096
LR, WEIGHT_DECAY = .0003, .0001
CASES = (("R10-C", "R10", False), ("R10-D", "R10", True),
         ("CNN-R10-C", "CNN-R10", False), ("CNN-R10-D", "CNN-R10", True))


class B2(nn.Module):
    def __init__(self, initial, trainable=False, *, exact_dot=False):
        super().__init__()
        self.exact_dot = exact_dot
        width = torch.as_tensor(initial["means"]).shape[-1]
        buckets = len(K_BUCKETS)
        for name, shape in (("means", (buckets, width)), ("scales", (buckets, width)),
                            ("weights", (buckets, 10, width)), ("biases", (buckets, 10))):
            tensor = torch.as_tensor(initial[name], dtype=torch.float32).clone()
            if tensor.shape != shape or not torch.isfinite(tensor).all():
                raise ValueError(f"invalid B2 {name}")
            if name == "scales" and not (tensor > 0).all():
                raise ValueError("nonpositive B2 scale")
            if trainable and name in ("weights", "biases"):
                self.register_parameter(name, nn.Parameter(tensor))
            else:
                self.register_buffer(name, tensor)

    def forward(self, features, k):
        slots = torch.as_tensor(np.searchsorted(K_BUCKETS, k), device=features.device)
        x = (features - self.means[slots]) / self.scales[slots]
        if self.exact_dot:
            # The overlapping teacher enables TF32 globally. Keep this small FP32 head independent of that flag.
            return (self.weights[slots] * x.unsqueeze(1)).sum(-1) + self.biases[slots]
        return torch.bmm(self.weights[slots], x.unsqueeze(-1)).squeeze(-1) + self.biases[slots]


class FrozenB2(B2):
    def __init__(self, fitted):
        values = [fitted.per_k[k] for k in K_BUCKETS]
        for _, model in values:
            np.testing.assert_array_equal(model.classes_, np.arange(10))
        super().__init__({
            "means": np.stack([scaler.mean_ for scaler, _ in values]),
            "scales": np.stack([scaler.scale_ for scaler, _ in values]),
            "weights": np.stack([model.coef_ for _, model in values]),
            "biases": np.stack([model.intercept_ for _, model in values]),
        })


def train_arm(case, cache, raw, indices, labels, k, batches, fitted, device, output):
    name, architecture, full = case
    view = cache.for_device(device)
    out = output / name
    out.mkdir(exist_ok=True)
    with torch.cuda.device(device):
        model = encoder.build(architecture, full, SEED).to(device)
        final = out / "model.pt"
        if final.is_file():
            saved = torch.load(final, map_location=device, weights_only=True)
            model.load_state_dict(saved["model"])
            return model
        head = FrozenB2(fitted).to(device).eval()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        lp, target, q, _ = next(view.chunks(np.arange(1), POSITION_CHUNK))
        with torch.no_grad():
            delta = model(lp, target, q)
            torch.testing.assert_close(delta, torch.zeros_like(delta), rtol=0, atol=0)
        del lp, target, q, delta
        checkpoint = out / "checkpoint.pt"
        history = []
        if checkpoint.is_file():
            saved = torch.load(checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            history = saved["losses"]
        started = time.monotonic()
        for step in range(len(history) + 1, len(batches) + 1):
            batch = batches[step - 1]
            optimizer.zero_grad(set_to_none=True)
            loss = encoder.backward_questions(model, view, raw, [indices[i] for i in batch],
                                              labels[batch], k[batch], head, POSITION_CHUNK)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            optimizer.step()
            history.append(loss)
            if step % 500 == 0 or step == len(batches):
                atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                             "losses": history}, checkpoint)
                print(dict(arm=name, step=step, loss=float(np.mean(history[-500:])),
                           seconds=time.monotonic() - started), flush=True)
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError(f"nonfinite parameters: {name}")
        write_json(out / "training.json", {"loss": history, "seconds": time.monotonic() - started})
        atomic_save({"model": model.state_dict(), "architecture": architecture, "full_policy": full,
                     "steps": len(batches), "seed": SEED}, final)
    return model


def frozen_features(model, cache, raw, device):
    model.eval()
    with torch.cuda.device(device):
        view = cache.for_device(device)
        values = [encoder.game_features(model, view, raw, np.arange(start, min(start + 64, len(raw))),
                                        POSITION_CHUNK).cpu().numpy() for start in range(0, len(raw), 64)]
    values = np.vstack(values)
    if values.shape != raw.shape or not np.isfinite(values).all():
        raise ValueError("invalid frozen per-game features")
    return values


def stage2(args):
    started = time.monotonic()
    root = args.stage1_out
    stage1 = json.loads((root / "complete.json").read_text())
    required = {"games.parquet", "E1.parquet", "E2.parquet", "features.npz", "result.json", "question_scores.parquet"}
    if stage1["stage"] != 1 or required != set(stage1["files"]) or any(
        file_hash(root / filename) != digest for filename, digest in stage1["files"].items()
    ):
        raise ValueError("first-stage inputs failed integrity verification")
    if not json.loads((root / "result.json").read_text())["parity"]["pass"]:
        raise ValueError("first-stage cvec2 parity did not pass")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "complete.json").exists():
        raise FileExistsError(args.out / "complete.json")
    games = pd.read_parquet(root / "games.parquet")
    sets = {split: pd.read_parquet(root / f"{split}.parquet") for split in ("E1", "E2")}
    with np.load(root / "features.npz", allow_pickle=False) as features:
        baseline, raw = features["A"], features[args.anchor]
    if raw.shape != (len(games), 40) or not np.isfinite(raw).all():
        raise ValueError("invalid second-stage anchor features")
    fit = games[games.role.ne("U")].reset_index(drop=True)
    if not games.iloc[:len(fit)].game_index.equals(fit.game_index):
        raise ValueError("stage 1 fitting games no longer form the expected prefix")
    prior = player_prior(fit[fit.role.eq("S")])
    fitted = prod._fit(fit, raw[:len(fit)], prior, SEED)
    draws = list(PriorMatched(prior).synthetic_rows(fit, fit.rank_id.to_numpy(), SEED))
    indices = [row for _, rows, _ in draws for row in rows]
    labels = np.concatenate([y for _, _, y in draws])
    k = np.concatenate([np.full(len(rows), bucket) for bucket, rows, _ in draws])
    batches = epoch_batches(pd.DataFrame({"k": k}), SEED, 1, QUESTIONS_PER_K_BATCH)
    settings = {"anchor": args.anchor, "stage1": str(root), "stage1_files": stage1["files"],
                "seed": SEED, "batch": 4 * QUESTIONS_PER_K_BATCH, "updates": len(batches),
                "lr": LR, "weight_decay": WEIGHT_DECAY, "position_chunk": POSITION_CHUNK,
                "code": {p.name: file_hash(p) for p in (encoder.__file__, full_cache.__file__, __file__)}}
    settings_path = args.out / "settings.json"
    if settings_path.is_file() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("second-stage settings changed since checkpoint")
    write_json(settings_path, settings)
    cache = full_cache.infer(games, args.out)
    expected = np.hstack((baseline[:, H_COLUMNS], baseline[:, 30:50]))
    error = np.abs(cache.arrays["c"].numpy() - expected)
    audit = {"max": float(error.max()), "mean": float(error.mean())}
    write_json(args.out / "teacher_parity.json", audit)
    if audit["max"] > .05 or audit["mean"] > .005:
        raise ValueError(f"played-move teacher parity failed: {audit}")
    devices = {case[0]: torch.device("cuda", i) for i, case in enumerate(CASES)}
    models = parallel([(case[0], train_arm, (case, cache, raw, indices, labels, k, batches, fitted,
                                              devices[case[0]], args.out)) for case in CASES], 4)
    arms = parallel([(name, frozen_features, (model, cache, raw, devices[name]))
                     for name, model in models.items()], 4)
    del cache, models
    comparisons = {
        pair + "-D_minus_C": (pair + "-C", pair + "-D") for pair in ("R10", "CNN-R10")
    } | {name + "_minus_A": ("A", name) for name in ("R10-D", "CNN-R10-D")}
    report, rows = run.evaluate(games, {"A": baseline, args.anchor: raw, **arms},
                                sets, comparisons, args.workers)
    for name in ("R10", "CNN-R10"):
        e1, e2 = (report["comparisons"][split][name + "-D_minus_A"] for split in ("E1", "E2"))
        gain = (e1["equal_k_delta"], e2["equal_k_delta"])
        report.setdefault("candidate_gate", {})[name] = (
            report["comparisons"]["E2"][name + "-D_minus_C"]["equal_k_delta"] > 0
            and max(gain) >= .005 and min(gain) >= 0 and e2["min_k_delta"] >= -.005
        )
    for split, q in sets.items():
        group = rows[rows.split.eq(split)]
        r10 = (group["R10-D"] - group["R10-C"]).to_numpy().reshape(len(prod.SEEDS), len(q)).mean(0)
        cnn = (group["CNN-R10-D"] - group["CNN-R10-C"]).to_numpy().reshape(len(prod.SEEDS), len(q)).mean(0)
        report["comparisons"][split]["CNN_gain_minus_R10_gain"] = {
            "delta": run.hf.by_k(cnn - r10, q.k.to_numpy()), "player_bootstrap_95": run.hf.bootstrap(r10, cnn, q)
        }
    rows.to_parquet(args.out / "question_scores.parquet", index=False)
    np.savez(args.out / "features.npz", **arms)
    write_json(args.out / "result.json", report | {"settings": settings, "seconds": time.monotonic() - started})
    write_json(args.out / "complete.json", {"stage": 2, "result_sha256": file_hash(args.out / "result.json")})
