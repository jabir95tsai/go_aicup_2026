"""Score the eight A-based arms on the fixed E2 questions."""
import json
from time import monotonic

import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits
import torch

from go_aicup.artifacts import file_hash, write_json
from go_aicup.metrics import K_BUCKETS, score_rank
from go_aicup.tasks.strength.v2.aggregate import official_scores
from scripts.strength.humansl_hd_filter import BOOTSTRAP
from . import all_data

C_NORMAL = "A/C10-logsoftmax"
HC_NORMAL = "A/H15-C10-logsoftmax"
CONTROLS = (C_NORMAL, HC_NORMAL)
ARCHITECTURES = ("A/R10", "A/CNN-R10")
MODEL_ARMS = tuple(f"{arch}-{case}" for arch in ARCHITECTURES for case in ("C", "D"))
MOTHER = "A/CNN-R10-D"
WEIGHTED = MOTHER + "-weighted"
WEIGHTED_ARMS = (WEIGHTED,)
LEARNED_ARMS = MODEL_ARMS + WEIGHTED_ARMS
CANDIDATES = CONTROLS + LEARNED_ARMS
ARM_FILES = ("model.pt", "training.json", "fit.npy", "fit_40.npy", "e2.npy", "features.json")


def fingerprint(root, names):
    return {str(name): file_hash(root / name) for name in names}


def verify_files(root, files):
    if not files or files != fingerprint(root, files):
        raise ValueError(f"artifact integrity check failed: {root}")


def write_table(path, frame):
    temporary = path.with_suffix(".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def by_k(values, k):
    buckets = {str(b): float(values[k == b].mean()) for b in K_BUCKETS}
    return buckets | {"equal_k": float(np.mean(list(buckets.values())))}


def bootstrap(questions, values):
    """Use the same resampled players across arms and k buckets."""
    players, slots = np.unique(questions.player_id.to_numpy(), return_inverse=True)
    k = questions.k.to_numpy()
    rng = np.random.default_rng(2027)
    counts = rng.multinomial(len(players), np.full(len(players), 1 / len(players)), size=BOOTSTRAP)
    totals = np.zeros((len(players), len(K_BUCKETS), len(values)), dtype=np.float64)
    sizes = np.zeros((len(players), len(K_BUCKETS)), dtype=np.int64)
    for col, bucket in enumerate(K_BUCKETS):
        rows = np.flatnonzero(k == bucket)
        np.add.at(totals[:, col], slots[rows], values[:, rows].T)
        np.add.at(sizes[:, col], slots[rows], 1)
    denominators = counts @ sizes.astype(np.float64)
    keep = (denominators > 0).all(axis=1)
    counts = counts[keep].astype(np.float64)
    if not len(counts):
        raise ValueError("no bootstrap replicate included all k buckets")
    means = (counts @ totals.reshape(len(players), -1)).reshape(len(counts), len(K_BUCKETS), len(values))
    means /= denominators[keep, :, None]
    means = np.concatenate((means, means.mean(axis=1, keepdims=True)), axis=1)
    limits = np.quantile(means, [.025, .975], axis=0)
    keys = [*(str(k) for k in K_BUCKETS), "equal_k"]
    return [{key: limits[:, i, j].tolist() for i, key in enumerate(keys)} for j in range(len(values))]


def comparisons(scores):
    deltas = {f"{arm} minus A": scores[arm] - scores["A"] for arm in CANDIDATES}
    for arm in (HC_NORMAL, *LEARNED_ARMS):
        deltas[f"{arm} minus {C_NORMAL}"] = scores[arm] - scores[C_NORMAL]
    for arch in ARCHITECTURES:
        deltas[f"{arch}-D minus {arch}-C"] = scores[arch + "-D"] - scores[arch + "-C"]
    deltas["CNN-R10 gain minus R10 gain"] = (
        deltas["A/CNN-R10-D minus A/CNN-R10-C"] - deltas["A/R10-D minus A/R10-C"]
    )
    deltas[f"{WEIGHTED} minus {MOTHER}"] = scores[WEIGHTED] - scores[MOTHER]
    return deltas


def candidate_gate(arm, deltas, effective=None):
    failures = []

    def minimum(label):
        if deltas[label]["equal_k"] < .005:
            failures.append(f"{label}: Equal-k below +0.005")
        for bucket in K_BUCKETS:
            if deltas[label][str(bucket)] < -.005:
                failures.append(f"{label}: k={bucket} below -0.005")

    minimum(f"{arm} minus A")
    if arm in LEARNED_ARMS and deltas[f"{arm} minus {C_NORMAL}"]["equal_k"] <= 0:
        failures.append("model Equal-k does not exceed the C10 normalization control")
    if arm.endswith("-D") and deltas[f"{arm} minus {arm[:-1]}C"]["equal_k"] <= 0:
        failures.append("full-policy minus played-move Equal-k is not positive")
    if arm == WEIGHTED:
        minimum(f"{arm} minus {MOTHER}")
        for side in ("own", "opp"):
            if effective[side]["p05"] < .50:
                failures.append(f"{side} effective-move ratio p05 below 0.50")
    return failures


def artifact_array(path, values):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values.astype(np.float32, copy=False), allow_pickle=False)
    temporary.replace(path)


def normalized_arms(run, data, gpu):
    offsets = {family: np.load(run / f"offset_{family}.npy", allow_pickle=False)
               for family in ("H", "C")}
    shapes = {"fit.npy": (len(data.groups["fit"]), 52), "e2.npy": (len(data.groups["e2"]), 52)}
    files = [*shapes, "fit_40.npy", "head.npz"]
    status = run / "normalization.json"
    identity = {f"offset_{fam}.npy": gpu["files"][f"offset_{fam}.npy"] for fam in offsets}
    manifest = json.loads(status.read_text()) if status.exists() else None
    if manifest is not None:
        if manifest["offsets"] != identity or manifest["identity"] != gpu["identity"]:
            raise ValueError("normalized B2 inputs changed")
        verify_files(run, manifest["files"])
        return manifest

    started = monotonic()
    arms = all_data.fit_normalized(data, offsets)
    if tuple(arms) != CONTROLS:
        raise ValueError("normalization did not produce both planned controls")
    n_fit = shapes["fit.npy"][0]
    for arm, (features, head) in arms.items():
        path = run / arm
        path.mkdir(parents=True, exist_ok=True)
        for name, values in zip(shapes, (features[:n_fit], features[n_fit:]), strict=True):
            if values.shape != shapes[name] or not np.isfinite(values).all():
                raise ValueError(f"invalid normalized features: {arm}/{name}")
            artifact_array(path / name, values)
        artifact_array(path / "fit_40.npy", features[:n_fit, all_data.PROJECTION])
        temporary = path / "head.tmp"
        with temporary.open("wb") as handle:
            np.savez(handle, **head)
        temporary.replace(path / "head.npz")
    manifest = {"offsets": identity, "identity": gpu["identity"], "seconds": monotonic() - started,
                "files": fingerprint(run, (f"{arm}/{name}" for arm in CONTROLS for name in files))}
    write_json(status, manifest)
    return manifest


def head_probabilities(head, features, positions, k):
    p = np.empty((len(k), 10), dtype=np.float64)
    for i, bucket in enumerate(K_BUCKETS):
        mask, indices = k == bucket, positions[bucket]
        x = features[indices].mean(axis=1)
        logits = ((x - head["means"][i]) / head["scales"][i]) @ head["weights"][i].T + head["biases"][i]
        p[mask] = softmax(logits, axis=1)
    if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("invalid scoring probabilities")
    return p


def checked_features(path, anchor, n_fit):
    fit = np.load(path / "fit.npy", mmap_mode="r", allow_pickle=False)
    e2 = np.load(path / "e2.npy", mmap_mode="r", allow_pickle=False)
    fit_40 = np.load(path / "fit_40.npy", mmap_mode="r", allow_pickle=False)
    for values, base in ((fit, anchor[:n_fit]), (e2, anchor[n_fit:])):
        if values.shape != base.shape or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError(f"invalid game feature shape or dtype in {path}")
        if not np.array_equal(values[:, -2:], base[:, -2:]):
            raise ValueError(f"HD1 flags differ from the A anchor in {path}")
    if fit_40.dtype != np.float32 or not np.array_equal(fit_40, fit[:, all_data.PROJECTION]):
        raise ValueError(f"invalid 40-column projection in {path}")
    return e2


def score(run, data, config, gpu):
    started = monotonic()
    normalization = normalized_arms(run, data, gpu)
    questions = data.questions.reset_index(drop=True)
    k = questions.k.to_numpy()
    y = questions.rank_id.to_numpy()
    index = pd.Index(data.groups["e2"].game_index)
    positions = {}
    for bucket in K_BUCKETS:
        games = np.stack(questions.loc[k == bucket, "games"])
        positions[bucket] = index.get_indexer(games.ravel()).reshape(games.shape)
        if (positions[bucket] < 0).any():
            raise ValueError("an E2 question references an uncached game")
    keys = ["question_id", "player_id", "rank_id", "k"]
    if not data.scores[keys].reset_index(drop=True).equals(questions[keys]):
        raise ValueError("baseline scores differ from the prepared E2 question order")
    score_rows = questions[keys].assign(seed=2027)
    scores = {"A": data.scores["A"].to_numpy()}
    predictions, parameters, effective = [], {}, {}
    n_fit = len(data.groups["fit"])
    for arm in CANDIDATES:
        path = run / arm
        values = checked_features(path, data.features["A"], n_fit)
        if arm in CONTROLS:
            with np.load(path / "head.npz", allow_pickle=False) as archive:
                head = dict(archive)
            parameters[arm] = 0
        else:
            model = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
            if model["steps"] != len(data.batches) or model["arm"] != arm or model["identity"] != gpu["identity"]:
                raise ValueError(f"invalid completed model: {arm}")
            head = {name: value.numpy() for name, value in model["head"].items()}
            parameters[arm] = model["parameters"]
        p = head_probabilities(head, values, positions, k)
        pred = score_rank(p)
        scores[arm] = official_scores(pred, y)
        predictions.append(score_rows.assign(arm=arm, pred_rank_id=pred, probabilities=p.tolist(), score=scores[arm]))
        if arm == WEIGHTED:
            effective[arm] = json.loads((path / "features.json").read_text())["effective_moves"]["e2"]

    deltas = comparisons(scores)
    means = {arm: by_k(value, k) for arm, value in scores.items()}
    difference = {label: by_k(value, k) for label, value in deltas.items()}
    smoke = config["smoke_steps"] is not None
    failures = {arm: candidate_gate(arm, difference, effective.get(arm)) for arm in CANDIDATES}
    passed = [arm for arm in CANDIDATES if not failures[arm]]
    selected = min(passed, key=lambda arm: (-means[arm]["equal_k"], parameters[arm], arm), default=None)
    all_values = scores | deltas
    with threadpool_limits(limits=1):
        intervals = dict(zip(all_values, bootstrap(questions, np.stack(list(all_values.values()))), strict=True))
    for arm in scores:
        score_rows[arm] = scores[arm]
    write_table(run / "question_scores.parquet", score_rows)
    write_table(run / "question_predictions.parquet", pd.concat(predictions, ignore_index=True))
    report = {
        "smoke": smoke, "seed": 2027, "questions": len(questions),
        "scores": {arm: {"score": means[arm], "player_bootstrap_95": intervals[arm]} for arm in means},
        "comparisons": {label: {"delta": difference[label], "player_bootstrap_95": intervals[label]}
                        for label in difference},
        "gate": None if smoke else {"winner": selected, "failures": failures}, "extra_parameters": parameters,
        "effective_moves": effective, "seconds": monotonic() - started,
    }
    write_json(run / "result.json", report)
    files = ("question_scores.parquet", "question_predictions.parquet", "result.json",
             "normalization.json", "config.json", "gpu_complete.json", "cpu/complete.json", "prepare_timing.json")
    cpu = {f"cpu/{name}": digest for name, digest in data.identity["files"].items()}
    write_json(run / "complete.json", {"stage": "all", "smoke": smoke,
               "files": fingerprint(run, files) | gpu["files"] | normalization["files"] | cpu})
    return report
