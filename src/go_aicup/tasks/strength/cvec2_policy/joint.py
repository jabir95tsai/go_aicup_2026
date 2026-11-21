"""Joint B2 and residual updates, followed by frozen-mother move weighting."""

from contextlib import nullcontext
import json
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from go_aicup.artifacts import file_hash, write_json
from go_aicup.tasks.strength.policy_residual.train import atomic_save
from . import encoder
from .played import H_COLUMNS
from .train import B2, CASES, LR, POSITION_CHUNK, SEED, WEIGHT_DECAY

FIT_PROJECTION = np.r_[H_COLUMNS, 30:52]
WEIGHTED = "CNN-R10-D-weighted"
MOTHER = "CNN-R10-D"
CONCENTRATION_WEIGHT = .01
QUESTIONS_PER_K = 20_000
CHECKPOINT_EVERY = 500


class MoveWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(29, 16, 3), nn.ReLU(),
                                  nn.Conv2d(16, 16, 3), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def board_inputs(self, p, packed, target, progress):
        shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=packed.device)
        stones = ((packed[:, :, None] >> shifts) & 1).flatten(1)[:, :722].reshape(-1, 2, 361)
        marker = F.one_hot(target, 362)[:, None, :361]
        ply = progress[:, None, None].expand(-1, 1, 361)
        return torch.cat((p[..., :361], stones, marker, ply), dim=1).reshape(-1, 29, 19, 19)

    def forward(self, p, packed, target, progress):
        x = self.board_inputs(p, packed, target, progress)
        board = self.conv(x.contiguous(memory_format=torch.channels_last)).mean((-2, -1))
        return self.head(board).float().squeeze(-1)


def move_chunks(cache, rows, chunk, counts):
    # Both teachers emit chronological moves, keeping the same side slots across chunks.
    lengths = counts.reshape(-1, 2).sum(1).long()
    starts = lengths.cumsum(0) - lengths
    offset = 0
    for policy, target, q, slot, packed in cache.chunks(rows, chunk, board=True):
        game = slot // 2
        ply = torch.arange(offset + 1, offset + 1 + len(slot), device=slot.device) - starts[game]
        yield policy, target, q, slot, packed, ply.float() / lengths[game]
        offset += len(slot)


@torch.no_grad()
def weighted_moves(mother, weight, cache, rows, chunk, counts):
    outputs = []
    for policy, target, q, slot, packed, progress in move_chunks(cache, rows, chunk, counts):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cache.device.type == "cuda"):
            inputs = mother.policy_inputs(policy, target, q)
            delta = mother(policy, target, q, inputs)
            logit = weight(inputs[0], packed, target, progress)
        outputs.append((logit, delta, q, slot))
    return tuple(torch.cat(arrays) for arrays in zip(*outputs, strict=True))


def weighted_reduce(base, logits, delta, q, slots, counts):
    """Stable FP32 softmax over all moves of each game side."""
    max_by_side = counts.new_full(counts.shape, -torch.inf)
    max_by_side.scatter_reduce_(0, slots, logits.detach(), reduce="amax")
    exp = (logits - max_by_side[slots]).exp()
    total = counts.new_zeros(counts.shape).index_add(0, slots, exp)
    w = exp / total[slots]
    # Cancellation-free form of weighted(q + delta) minus mean(q).
    deviation = w - counts[slots].reciprocal()
    updates = w[:, None] * delta + deviation[:, None] * q
    side = q.new_zeros((len(counts), q.shape[1])).index_add(0, slots, updates)
    concentration = 1 + counts * counts.new_zeros(counts.shape).index_add(0, slots, deviation.square())
    return encoder.add_side_features(base, side), concentration


def backward_weighted(mother, weight, cache, raw, questions, labels, k, head, chunk):
    unique, local = encoder.question_indices(questions)
    counts = cache.counts(unique)
    logits, delta, q, slots = weighted_moves(mother, weight, cache, unique, chunk, counts)
    base = torch.as_tensor(raw[unique], dtype=torch.float32, device=cache.device)
    logits.requires_grad_()
    features, concentration = weighted_reduce(base, logits, delta, q, slots, counts)
    ce = encoder.question_ce(features, local, labels, k, head)
    penalty = CONCENTRATION_WEIGHT * (concentration - 1).mean()
    loss = ce + penalty
    if not torch.isfinite(loss):
        raise ValueError("nonfinite weighted policy question loss")
    (grad,) = torch.autograd.grad(loss, logits)
    start = 0
    for policy, target, q, _, packed, progress in move_chunks(cache, unique, chunk, counts):
        with torch.no_grad():
            p = mother.policy_inputs(policy, target, q)[0]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cache.device.type == "cuda"):
            logit = weight(p, packed, target, progress)
        stop = start + len(logit)
        logit.backward(grad[start:stop])
        start = stop
    return float(ce.detach()), float(penalty.detach())


def head_penalty(head):
    return head.weights.square().sum((1, 2)).mean() / (2 * QUESTIONS_PER_K)


def parse_arm(arm):
    anchor, name = arm.split("/")
    if anchor != "A":
        raise ValueError(f"invalid anchor: {anchor}")
    architecture, full = {name: (architecture, full) for name, architecture, full in CASES}[
        MOTHER if name == WEIGHTED else name
    ]
    return anchor, name == WEIGHTED, architecture, full


def load_checkpoint(path, arm, identity, mother):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (saved["arm"], saved["identity"], saved["mother_sha256"]) != (arm, identity, mother):
        raise ValueError(f"checkpoint identity differs: {path}")
    return saved


def arm_modules(arm, out, identity, initial):
    anchor, weighted, architecture, full = parse_arm(arm)
    model = encoder.build(architecture, full, SEED, raw_a=True)
    if not weighted:
        return model, None, B2(initial, trainable=True, exact_dot=True), None
    parent = out / anchor / MOTHER / "model.pt"
    digest = file_hash(parent)
    saved = load_checkpoint(parent, f"{anchor}/{MOTHER}", identity, None)
    model.load_state_dict(saved["model"])
    model.requires_grad_(False).eval()
    with encoder.INITIALIZATION_LOCK, torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(SEED)
        weight = MoveWeights()
    return model, weight, B2(saved["head"], exact_dot=True), digest


def device_scope(device):
    return torch.cuda.device(device) if device.type == "cuda" else nullcontext()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_stats(device):
    if device.type != "cuda":
        return {}
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def check_abort(abort):
    if abort is not None and abort.is_set():
        raise RuntimeError("another experiment worker failed")


def train_one(arm, cache, raw, indices, labels, k, batches, initial, device, out, identity,
              abort=None, *, on_first_update=None):
    """Resume one arm at its last completed update and publish the final model atomically."""
    device = torch.device(device)
    _, weighted, _, _ = parse_arm(arm)
    path = out / arm
    path.mkdir(parents=True, exist_ok=True)
    final = path / "model.pt"
    checkpoint = path / "checkpoint.pt"
    if not len(batches) or raw.shape != (len(cache.games), 52) or raw.dtype != np.float32:
        raise ValueError("training requires batches and aligned FP32 game features")
    history, regularizers = [], []
    with device_scope(device):
        view = cache.for_device(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model, weight, head, mother = arm_modules(arm, out, identity, initial)
        model, head = model.to(device), head.to(device)
        trainable = weight.to(device) if weighted else model
        groups = [{"params": trainable.parameters()}]
        if not weighted:
            groups.append({"params": head.parameters(), "weight_decay": 0})
        optimizer = torch.optim.AdamW(groups, lr=LR, weight_decay=WEIGHT_DECAY)
        parameters = [param for group in optimizer.param_groups for param in group["params"]]
        payload = {"arm": arm, "identity": identity, "mother_sha256": mother}
        if final.is_file():
            saved = load_checkpoint(final, arm, identity, mother)
            if saved["steps"] != len(batches):
                raise ValueError(f"final step differs: {final}")
            write_json(path / "training.json", saved["diagnostics"])
            return saved["diagnostics"]
        initial_time = time.monotonic()
        weights_key = "weight" if weighted else "model"
        first_update_seconds = None
        first_update_peer_bytes = None
        remaining_update_seconds = 0.
        remaining_update_peer_bytes = 0
        if checkpoint.is_file():
            saved = load_checkpoint(checkpoint, arm, identity, mother)
            trainable.load_state_dict(saved[weights_key])
            head.load_state_dict(saved["head"])
            optimizer.load_state_dict(saved["optimizer"])
            history, regularizers = saved["ce"], saved["regularizer"]
            if not 0 < saved["steps"] == len(history) == len(regularizers) <= len(batches):
                raise ValueError(f"checkpoint step differs: {checkpoint}")
            first_update_seconds = saved["first_update_seconds"]
            first_update_peer_bytes = saved["first_update_peer_bytes"]
            remaining_update_seconds = saved["remaining_update_seconds"]
            remaining_update_peer_bytes = saved["remaining_update_peer_bytes"]
        else:
            rows = np.array([0])
            with torch.no_grad():
                if weighted:
                    counts = view.counts(rows)
                    logits, delta, q, slots = weighted_moves(model, weight, view, rows, POSITION_CHUNK, counts)
                    base = torch.as_tensor(raw[rows], device=device)
                    got, concentration = weighted_reduce(base, logits, delta, q, slots, counts)
                    expected = encoder.game_features(model, view, raw, rows, POSITION_CHUNK)
                    torch.testing.assert_close(concentration, torch.ones_like(concentration))
                    torch.testing.assert_close(got, expected, rtol=1e-5, atol=2e-6)
                else:
                    got = encoder.game_features(model, view, raw, rows, POSITION_CHUNK)
                    torch.testing.assert_close(got, torch.as_tensor(raw[rows], device=device), rtol=0, atol=0)
        synchronize(device)
        init_seconds = time.monotonic() - initial_time
        started = time.monotonic()
        first_step = len(history) + 1
        for step in range(first_step, len(batches) + 1):
            check_abort(abort)
            if step == first_step and on_first_update is not None:
                on_first_update()
            tick = time.monotonic()
            before = getattr(view, "peer_bytes", 0)
            batch = batches[step - 1]
            questions = [indices[i] for i in batch]
            optimizer.zero_grad(set_to_none=True)
            if weighted:
                ce, penalty = backward_weighted(model, weight, view, raw, questions, labels[batch],
                                                k[batch], head, POSITION_CHUNK)
            else:
                reg = head_penalty(head)
                ce = encoder.backward_questions(model, view, raw, questions, labels[batch],
                                                k[batch], head, POSITION_CHUNK, reg)
                penalty = float(reg.detach())
            torch.nn.utils.clip_grad_norm_(parameters, float("inf"), error_if_nonfinite=True)
            optimizer.step()
            synchronize(device)
            elapsed = time.monotonic() - tick
            peer_bytes = getattr(view, "peer_bytes", 0) - before
            if step == 1:
                first_update_seconds = elapsed
                first_update_peer_bytes = peer_bytes
            else:
                remaining_update_seconds += elapsed
                remaining_update_peer_bytes += peer_bytes
            history.append(ce)
            regularizers.append(penalty)
            if step % CHECKPOINT_EVERY == 0 or step == len(batches):
                timing = {"first_update_seconds": first_update_seconds,
                          "first_update_peer_bytes": first_update_peer_bytes,
                          "remaining_update_seconds": remaining_update_seconds,
                          "remaining_update_peer_bytes": remaining_update_peer_bytes}
                atomic_save(payload | {weights_key: trainable.state_dict(), "head": head.state_dict(),
                            "optimizer": optimizer.state_dict(), "steps": step, "ce": history,
                            "regularizer": regularizers} | timing, checkpoint)
                print(json.dumps({"arm": arm, "step": step, "ce": float(np.mean(history[-500:])),
                                  "seconds": time.monotonic() - started}), flush=True)
        if any(not torch.isfinite(p).all() for p in parameters):
            raise ValueError(f"nonfinite parameters: {arm}")
        diagnostics = {"ce": history, "regularizer": regularizers, "steps": len(batches),
                       "init_seconds": init_seconds, "seconds": time.monotonic() - started,
                       "first_update_seconds": first_update_seconds,
                       "remaining_update_seconds": remaining_update_seconds,
                       "first_update_peer_bytes": first_update_peer_bytes,
                       "remaining_update_peer_bytes": remaining_update_peer_bytes} | memory_stats(device)
        params = sum(p.numel() for p in model.parameters())
        params += sum(p.numel() for p in weight.parameters()) if weighted else 0
        atomic_save(payload | {weights_key: trainable.state_dict(), "head": head.state_dict(),
                    "steps": len(batches), "parameters": params, "diagnostics": diagnostics}, final)
        write_json(path / "training.json", diagnostics)
    return diagnostics


def moves_summary(ratio):
    return {side: {"median": float(np.median(ratio[:, i])),
                   "p05": float(np.percentile(ratio[:, i], 5))}
            for i, side in enumerate(("own", "opp"))}


@torch.no_grad()
def extract_one(arm, cache, raw, device, out, identity, n_fit, abort=None):
    """Publish checked, ordered fit and E2 feature arrays under the arm's final checkpoint."""
    device = torch.device(device)
    path = out / arm
    final = path / "model.pt"
    manifest = path / "features.json"
    digest = file_hash(final)
    if not 0 < n_fit < len(raw):
        raise ValueError("fit and E2 feature ranges must be nonempty")
    anchor, weighted, architecture, full = parse_arm(arm)
    parent_digest = file_hash(out / anchor / MOTHER / "model.pt") if weighted else None
    saved = load_checkpoint(final, arm, identity, parent_digest)
    shapes = {"fit.npy": (n_fit, 52), "fit_40.npy": (n_fit, 40), "e2.npy": (len(raw) - n_fit, 52)}
    if manifest.is_file():
        record = json.loads(manifest.read_text())
        if (record["identity"] == identity and record["model_sha256"] == digest
                and set(record["files"]) == set(shapes)):
            for name, shape in shapes.items():
                file = path / name
                if not file.is_file() or file_hash(file) != record["files"][name]:
                    break
                values = np.load(file, allow_pickle=False, mmap_mode="r")
                if values.shape != shape or values.dtype != np.float32:
                    break
            else:
                return record
    with device_scope(device):
        view = cache.for_device(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if weighted:
            model, weight, _, _ = arm_modules(arm, out, identity, None)
            weight.load_state_dict(saved["weight"])
            weight = weight.eval().to(device)
        else:
            model = encoder.build(architecture, full, SEED, raw_a=True)
            model.load_state_dict(saved["model"])
        model = model.eval().to(device)
        features = np.empty_like(raw, dtype=np.float32)
        ratio = np.empty((len(raw), 2), dtype=np.float32) if weighted else None
        started = time.monotonic()
        for start in range(0, len(raw), 64):
            check_abort(abort)
            rows = np.arange(start, min(start + 64, len(raw)))
            if weighted:
                counts = view.counts(rows)
                logits, delta, q, slots = weighted_moves(model, weight, view, rows, POSITION_CHUNK, counts)
                base = torch.as_tensor(raw[rows], dtype=torch.float32, device=device)
                values, concentration = weighted_reduce(base, logits, delta, q, slots, counts)
                ratio[rows] = concentration.reciprocal().reshape(-1, 2).cpu().numpy()
            else:
                values = encoder.game_features(model, view, raw, rows, POSITION_CHUNK)
            features[rows] = values.cpu().numpy()
        synchronize(device)
        record = {"identity": identity, "model_sha256": digest, "seconds": time.monotonic() - started,
                  "positions": int(cache.games.length.sum()),
                  "peer_bytes": getattr(view, "peer_bytes", 0)} | memory_stats(device)
    if not np.isfinite(features).all() or features.shape != (len(raw), 52):
        raise ValueError(f"invalid features: {arm}")
    if weighted:
        if not np.isfinite(ratio).all() or (ratio <= 0).any() or (ratio > 1 + 1e-5).any():
            raise ValueError(f"invalid effective move ratio: {arm}")
        record["effective_moves"] = {"fit": moves_summary(ratio[:n_fit]), "e2": moves_summary(ratio[n_fit:])}
    files = {}
    for name, values in zip(shapes, (features[:n_fit], features[:n_fit, FIT_PROJECTION], features[n_fit:]), strict=True):
        temporary = path / (name + ".tmp")
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
        temporary.replace(path / name)
        files[name] = file_hash(path / name)
    record["files"] = files
    write_json(manifest, record)
    return record
