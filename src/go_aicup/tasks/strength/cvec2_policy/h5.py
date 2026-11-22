"""Train a five-block HumanSL policy correction behind a frozen R10-D and B2."""

from concurrent.futures import ThreadPoolExecutor
from time import monotonic

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from go_aicup.features.humansl import PROFILES
from go_aicup.features.humansl_rankvec import unpack
from go_aicup.features.humansl_torch import CKPT, _katago
from go_aicup.metrics import K_BUCKETS
from go_aicup.tasks.strength.policy_h.extract import profile_rows
from go_aicup.tasks.strength.policy_residual.model import question_mean
from go_aicup.tasks.strength.policy_residual.train import atomic_save

from . import encoder
from .train import B2


ARM = "H5-R10-D"
WEIGHTS = (1, 1, 2, 2)
LR = 3e-4
DECAY = 1e-4
ALPHA = .25
PER_SIDE = 8
CHUNK = 32
DEVICES = ("cuda:4", "cuda:5", "cuda:6", "cuda:7")


class Policy(nn.Module):
    def __init__(self, config):
        super().__init__()
        _katago()
        from katago.train.model_pytorch import Model

        # Only the primary policy head contributes to this loss.
        self.net = Model(config | {"block_kind": config["block_kind"][:5],
                                   "has_intermediate_head": False}, 19)
        self.net.initialize()
        self.net.value_head.requires_grad_(False)
        with torch.no_grad():
            self.net.policy_head.conv2p.weight.zero_()
            self.net.policy_head.linear_pass.weight.zero_()
        self.register_buffer("profiles", torch.from_numpy(profile_rows(PROFILES)), persistent=False)

    def forward(self, x, g, color):
        width = len(PROFILES)
        inputs = x.repeat_interleave(width, dim=0)
        globals_ = g.repeat_interleave(width, dim=0)
        meta = self.profiles[color].flatten(0, 1)
        # Reuse the checkpoint's stem, blocks, final norm and primary policy head.
        mask = inputs[:, :1].contiguous()
        sizes = mask.sum(dim=(2, 3), keepdim=True)
        size = sizes.sum()
        z = self.net.conv_spatial(inputs)
        z = z + self.net.linear_global(globals_)[:, :, None, None]
        z = z + self.net.metadata_encoder(meta, None)[:, :, None, None]
        for block in self.net.blocks:
            z = z + block(z, mask=mask, mask_sum_hw=sizes, mask_sum=size, extra_outputs=None)
        z = self.net.act_trunkfinal(
            self.net.norm_trunkfinal(z, mask=mask, mask_sum_hw=sizes, mask_sum=size))
        with torch.autocast("cuda", enabled=False):
            policy = self.net.policy_head(z.float(), mask.float(), sizes.float(), size.float(), None)
        return policy[:, 0, :].reshape(len(x), width, 362)


def model_config():
    config = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)["config"]
    if (config["version"], config["trunk_num_channels"], len(config["block_kind"])) != (14, 384, 18):
        raise ValueError("unexpected HumanSL model config")
    return config


def policy_shift(logp, q, target, d):
    """Apply p + delta p, keeping the exact played logp and the legal mask."""
    target_idx = target[:, None, None].expand(-1, len(PROFILES), 1)
    h = logp[:, :len(PROFILES)].float()
    hq = q[:, :len(PROFILES)]
    one_hot = F.one_hot(target, 362).bool()[:, None, :]
    # Cache logp is FP16. Centre with exact played mass and normalized remaining mass.
    rest = h.masked_fill(one_hot, -torch.inf)
    rest_norm = rest.logsumexp(-1, keepdim=True)
    rest_norm = torch.where(rest_norm.isfinite(), rest_norm, 0)
    played_mass = hq.exp().clamp(max=1)
    p = (rest - rest_norm).exp() * (1 - played_mass[..., None])
    p = p.scatter(2, target_idx, played_mass[..., None])
    r = d.float().tanh()
    factor = torch.log1p(ALPHA * (r - (p * r).sum(-1, keepdim=True)))
    played = factor.gather(2, target_idx).squeeze(-1)
    return (torch.cat((h + factor, logp[:, len(PROFILES):].float()), dim=1),
            torch.cat((hq + played, q[:, len(PROFILES):]), dim=1), played)


def correction(net, mother, bits, glob, color, logp, target, q):
    device = bits.device.type
    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        logits = net(unpack(bits), glob, color)
        policy, modified, played = policy_shift(logp, q, target, logits)
        with torch.no_grad():
            baseline = mother(logp, target, q)
        diff = mother(policy, target, modified) - baseline
    return diff + F.pad(played, (0, 10))


class Moves:
    """Packed canonical inputs for the sampled positions, indexed into the resident H/C cache."""
    def __init__(self, reader, games, per_side, seed):
        rng = np.random.default_rng(seed)
        lengths = games.length.to_numpy(dtype=np.int64)
        full_offsets = np.r_[0, np.cumsum(lengths)]
        maximum = int(lengths.sum()) if per_side is None else int(np.minimum(lengths, 2 * per_side).sum())
        self.bits = np.empty((maximum, 993), dtype=np.uint8)
        self.glob = np.empty((maximum, 19), dtype=np.float32)
        self.side = np.empty(maximum, dtype=np.int64)
        self.target = np.empty(maximum, dtype=np.int64)
        self.positions = np.empty(maximum, dtype=np.int64)
        self.counts = np.zeros((len(games), 2), dtype=np.int64)
        self.offsets = np.zeros(len(games) + 1, dtype=np.int64)
        self.colors = games.player_color.to_numpy(dtype=np.int64)
        offset = 0
        for rows, entries in reader.batches(0, len(games), 64):
            for row, (bits, glob, target, side) in zip(rows, entries, strict=True):
                selected = np.arange(len(side))
                if per_side is not None:
                    ids = [np.flatnonzero(side == color) for color in (0, 1)]
                    selected = np.sort(np.concatenate(
                        [rng.choice(a, min(per_side, len(a)), replace=False) for a in ids]))
                stop = offset + len(selected)
                self.bits[offset:stop] = bits[selected]
                self.glob[offset:stop] = glob[selected]
                self.target[offset:stop] = target[selected]
                self.side[offset:stop] = side[selected]
                self.positions[offset:stop] = full_offsets[row] + selected
                self.counts[row] = np.bincount(side[selected], minlength=2)
                self.offsets[row + 1] = stop
                offset = stop
        if (self.counts == 0).any() or offset != self.counts.sum():
            raise ValueError("incomplete canonical policy correction inputs")

    def select(self, rows):
        counts = np.diff(self.offsets)[rows]
        ids = np.concatenate([np.arange(self.offsets[row], self.offsets[row + 1]) for row in rows])
        game = np.repeat(np.arange(len(rows)), counts)
        slots = 2 * game + self.side[ids]
        return ids, slots, self.side[ids] ^ self.colors[rows[game]]


def teacher_rows(cache, moves, ids, device):
    positions = moves.positions[ids]
    logp = torch.empty((len(ids), 25, 362), device=device, dtype=torch.float16)
    q = torch.empty((len(ids), 25), device=device)
    assigned = np.zeros(len(ids), dtype=bool)
    for h, c in zip(cache.h.shards, cache.c.shards, strict=True):
        start, stop = cache.h.bounds[[h.first, h.last]]
        local = np.flatnonzero((positions >= start) & (positions < stop))
        if not len(local):
            continue
        destination = torch.as_tensor(local, device=device)
        rows = positions[local] - start
        ih = torch.as_tensor(rows, device=h.q.device)
        ic = torch.as_tensor(rows, device=c.q.device)
        logp[destination, :15] = h.logp[ih].to(device)
        logp[destination, 15:] = c.logp[ic].to(device)
        q[destination, :15] = h.q[ih].to(device)
        q[destination, 15:] = c.q[ic].to(device)
        assigned[local] = True
    if not assigned.all():
        raise ValueError("policy cache did not cover sampled moves")
    return logp, q


class Lane:
    def __init__(self, config, state, r10, device):
        self.device = torch.device(device)
        self.net = Policy(config).to(device, memory_format=torch.channels_last)
        self.net.load_state_dict(state)
        self.mother = encoder.build("R10", True, 2027, raw_a=True).to(device).eval().requires_grad_(False)
        self.mother.load_state_dict(r10)

    def inputs(self, cache, moves, ids, slots, colors):
        device = self.device
        logp, q = teacher_rows(cache, moves, ids, device)
        values = [moves.bits[ids], moves.glob[ids], colors, moves.target[ids], slots]
        bits, glob, color, target, slots = (torch.as_tensor(a, device=device) for a in values)
        return bits, glob, color, logp, target, q, slots

    def chunks(self, inputs):
        for start in range(0, len(inputs[0]), CHUNK):
            *values, slots = (a[start:start + CHUNK] for a in inputs)
            yield correction(self.net, self.mother, *values), slots

    @torch.no_grad()
    def forward(self, inputs, n):
        with torch.cuda.device(self.device):
            acc = torch.zeros((n, 25), device=self.device)
            for diff, slots in self.chunks(inputs):
                acc.index_add_(0, slots, diff)
            return acc

    def backward(self, inputs, gradient):
        with torch.cuda.device(self.device):
            self.net.zero_grad(set_to_none=True)
            gradient = gradient.to(self.device)
            for diff, slots in self.chunks(inputs):
                diff.backward(gradient[slots])


def balanced_loss(logits, labels, k):
    losses = F.cross_entropy(logits, labels, reduction="none")
    per_k = torch.stack([losses[torch.as_tensor(k == size, device=logits.device)].mean() for size in K_BUCKETS])
    weights = per_k.new_tensor(WEIGHTS)
    return (per_k * weights).sum() / weights.sum(), per_k


def _states(config, saved, seed):
    if saved is not None:
        return saved["model"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return Policy(config).state_dict()


def train(cache, reader, games, baseline, r10, head, indices, labels, k, batches, out, identity, seed):
    checkpoint = out / "h5_checkpoint.pt"
    final = out / "h5.pt"
    source = final if final.exists() else checkpoint
    saved = torch.load(source, map_location="cpu", weights_only=True) if source.exists() else None
    if saved is not None and saved["identity"] != identity:
        raise ValueError("H5 checkpoint sources changed")
    if final.exists():
        if saved["steps"] != len(batches):
            raise ValueError("H5 final checkpoint has different training budget")
        return saved
    config = model_config()
    state = _states(config, saved, seed)
    moves = Moves(reader, games, PER_SIDE, seed)
    lanes = [Lane(config, state, r10, d) for d in DEVICES]
    master = lanes[0]
    b2 = B2(head, exact_dot=True).to(master.device).eval()
    params = tuple(master.net.parameters())
    optimizer = torch.optim.AdamW((p for p in params if p.requires_grad), lr=LR, weight_decay=DECAY)
    history = [] if saved is None else saved["history"]
    if saved is not None:
        if not 0 < saved["steps"] == len(history) <= len(batches):
            raise ValueError("H5 checkpoint step or history differs")
        optimizer.load_state_dict(saved["optimizer"])
    started = monotonic()
    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        for step in range(len(history) + 1, len(batches) + 1):
            batch = batches[step - 1]
            rows, local = encoder.question_indices([indices[i] for i in batch])
            ids, slots, color = moves.select(rows)
            chunks = np.array_split(np.arange(len(ids)), len(lanes))
            blocks = list(pool.map(
                lambda task: task[0].inputs(cache, moves, ids[task[1]], slots[task[1]], color[task[1]]),
                zip(lanes, chunks, strict=True)))
            n = 2 * len(rows)
            sums = sum(t.to(master.device) for t in pool.map(
                lambda task: task[0].forward(task[1], n), zip(lanes, blocks, strict=True)))
            counts = torch.as_tensor(moves.counts[rows].ravel(), device=master.device)[:, None]
            if step == 1:
                torch.testing.assert_close(sums, torch.zeros_like(sums), atol=1e-5, rtol=0)
            features = encoder.add_side_features(
                torch.as_tensor(baseline[rows], device=master.device), sums / counts).requires_grad_()
            label = torch.as_tensor(labels[batch], device=master.device)
            loss, by_k = balanced_loss(b2(question_mean(features, local), k[batch]), label, k[batch])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite H5 question loss")
            (grad,) = torch.autograd.grad(loss, features)
            side = torch.cat((grad[:, :30].reshape(-1, 15), grad[:, 30:50].reshape(-1, 10)), 1) / counts
            list(pool.map(lambda task: task[0].backward(task[1], side), zip(lanes, blocks, strict=True)))
            for other in lanes[1:]:
                for p, v in zip(params, other.net.parameters(), strict=True):
                    if v.grad is not None:
                        p.grad = v.grad.to(master.device) if p.grad is None else p.grad.add_(v.grad.to(master.device))
            torch.nn.utils.clip_grad_norm_(params, float("inf"), error_if_nonfinite=True)
            optimizer.step()
            weights = master.net.state_dict()
            for other in lanes[1:]:
                other.net.load_state_dict(weights)
            history.append(by_k.detach().cpu().tolist())
            if step % 100 == 0 or step == len(batches):
                print({"arm": ARM, "step": step, "ce_by_k": np.mean(history[-100:], axis=0).tolist(),
                       "seconds": monotonic() - started}, flush=True)
                atomic_save({"identity": identity, "steps": step, "model": weights,
                             "history": history, "optimizer": optimizer.state_dict()}, checkpoint)
    payload = {"identity": identity, "steps": len(batches), "model": master.net.state_dict(), "history": history}
    atomic_save(payload, final)
    return payload


def extract(cache, reader, games, baseline, r10, state, seed):
    """Apply the trained correction to every move of a test part."""
    config = model_config()
    moves = Moves(reader, games, None, seed)
    lanes = [Lane(config, state, r10, d) for d in DEVICES]
    features = np.empty_like(baseline)

    def part(task):
        lane, rows = task
        for start in range(0, len(rows), 16):
            chunk = rows[start:start + 16]
            ids, slots, colors = moves.select(chunk)
            values = lane.inputs(cache, moves, ids, slots, colors)
            sums = lane.forward(values, 2 * len(chunk))
            counts = torch.as_tensor(moves.counts[chunk].ravel(), device=lane.device)[:, None]
            base = torch.as_tensor(baseline[chunk], device=lane.device)
            features[chunk] = encoder.add_side_features(base, sums / counts).cpu().numpy()

    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        list(pool.map(part, zip(lanes, np.array_split(np.arange(len(games)), len(lanes)), strict=True)))
    if not np.isfinite(features).all():
        raise ValueError("nonfinite official H5 features")
    return features
