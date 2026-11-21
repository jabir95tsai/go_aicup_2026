"""Read aligned cvec2 policy shards and overlap H15 inference with training."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import json
from threading import Event, Lock
import time

import numpy as np
import pandas as pd
import torch

from go_aicup.artifacts import write_json
from go_aicup.features.humansl import PROFILES
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.policy_large import cache
from go_aicup.tasks.strength.policy_large.run import parallel

from .planes import GAMES_PER_SHARD, PlaneReader


def a_config():
    """A experiment settings; historical experiments use the original JSON."""
    config = json.loads((ROOT / "configs/strength/policy_large_200k.json").read_text())
    return config | {"profiles_H": list(PROFILES), "inference_positions": {"H": 128, "C": 256},
                     "cache_fraction": .85}


class JointCache:
    def __init__(self, h, c):
        self.games = h.games
        if not h.games.equals(c.games) or not np.array_equal(h.bounds, c.bounds):
            raise ValueError("H/C game rows or position offsets differ")
        h_cols = h.arrays["c"].shape[1]
        if h_cols not in (18, 30) or c.arrays["c"].shape != (len(self.games), 20):
            raise ValueError("expected H9 or H15 and C10 cache profiles")
        if not torch.equal(h.arrays["counts"], c.arrays["counts"]):
            raise ValueError("H/C per-side move counts differ")
        for hs, cs in zip(h.shards, c.shards, strict=True):
            if (hs.first, hs.last) != (cs.first, cs.last):
                raise ValueError("H/C shard game ranges differ")
            if not torch.equal(hs.target, cs.target.to(hs.target.device)):
                raise ValueError("H/C played moves differ")
            if not torch.equal(hs.side, cs.side.to(hs.side.device)):
                raise ValueError("H/C move sides differ")
            if h_cols == 30:
                if hs.board is None or cs.board is not None:
                    raise ValueError("expected mover-stone planes on H15 shards only")
                for start in range(0, len(hs.target), 65536):
                    end = start + 65536
                    h_legal = torch.isfinite(hs.logp[start:end, 0])
                    c_legal = torch.isfinite(cs.logp[start:end, 0]).to(h_legal.device)
                    if (h_legal & ~c_legal).any():
                        raise ValueError("H15 has a legal move outside the C10 mask")
        self.arrays = {
            "counts": h.arrays["counts"],
            "c": torch.cat((h.arrays["c"], c.arrays["c"]), dim=1),
        }
        self.offsets = {"H": h.offsets, "C": c.offsets}
        if h_cols == 30 and any(values is None for values in self.offsets.values()):
            raise ValueError("H15 and C10 must provide both game-side profile offsets")
        for family, values in self.offsets.items():
            if values is not None:
                self.arrays[f"offset_{family}"] = torch.from_numpy(values)
        self.h, self.c = h, c

    def for_device(self, device):
        return JointView(self.h.for_device(device), self.c.for_device(device), self.arrays, self.games)


class JointView:
    def __init__(self, h, c, arrays, games):
        self.h, self.c, self.arrays = h, c, arrays
        self.device, self.games = h.device, games

    def counts(self, rows):
        return self.h.counts(rows)

    @property
    def peer_bytes(self):
        return self.h.peer_bytes + self.c.peer_bytes

    def chunks(self, rows, chunk, *, board=False):
        # Whole-game ranges and targets/sides were compared once at cache construction.
        for left, right in zip(self.h.chunks(rows, chunk, board=board), self.c.chunks(rows, chunk), strict=True):
            h, target, qh, slot, *stones = left
            c, _, qc, _ = right
            yield torch.cat((h, c), dim=1), target, torch.cat((qh, qc), dim=1), slot, *stones


def _check_devices_and_games(games):
    if torch.cuda.device_count() != 8:
        raise ValueError("cvec2 full-policy training requires eight GPUs")
    for device in range(8):
        for other in range(8):
            if device != other and not torch.cuda.can_device_access_peer(device, other):
                raise ValueError("cross-family GPU peer access is required for policy shards")
    if not games.game_index.is_unique or not games.player_color.isin([0, 1]).all():
        raise ValueError("invalid cvec2 game ordering or player colors")
    if not games.handicap.isin(range(10)).all():
        raise ValueError("invalid cvec2 handicap metadata")


def infer(games, out, planes_dir=None, config=None, nominal_lengths=None):
    """Stream each teacher into four GPU shards and join them as batches are read."""
    _check_devices_and_games(games)
    plane_batches = None
    if planes_dir is None:
        board = pd.read_parquet(ROOT / "data/processed/boards/strength_v1/games.parquet",
                                columns=["game_index", "length"]).set_index("game_index")
        length = board.loc[games.game_index, "length"].to_numpy(dtype=np.int64)
        if "length" in games and not np.array_equal(games.length, length):
            raise ValueError("cvec2 positions differ from the replay cache")
        games = games.assign(length=length)
        # Keep the original E1 replay for HD>=2; only HD1 receives komi 0.5.
        teacher_games = games.assign(handicap=games.handicap.eq(1).astype(np.int64))
    else:
        teacher_games = games
        plane_batches = PlaneReader(planes_dir, teacher_games).batches
    if config is None:
        config = a_config() if planes_dir is not None else json.loads(
            (ROOT / "configs/strength/policy_large_200k.json").read_text()
        )
    if planes_dir is not None and tuple(config["profiles_H"]) != PROFILES:
        raise ValueError("H15 teacher profiles must match the original A column order")
    matmul, cudnn = torch.backends.cuda.matmul, torch.backends.cudnn
    flags = matmul.allow_tf32, cudnn.allow_tf32
    try:
        matmul.allow_tf32 = cudnn.allow_tf32 = True
        result = parallel([
            (family, cache.infer_family, (family, teacher_games, config, out, cache.ABORT, plane_batches, nominal_lengths))
            for family in ("H", "C")
        ], 2)
    finally:
        matmul.allow_tf32, cudnn.allow_tf32 = flags
    alignment_started = time.monotonic()
    joint = JointCache(result["H"][0], result["C"][0])
    if planes_dir is not None:
        write_json(out / "teacher_alignment.json", {"games": len(games), "positions": int(games.length.sum()),
                   "seconds": time.monotonic() - alignment_started, "h15_legal_subset_of_c10": True})
    joint.games = games
    return joint


class FitWork:
    """Reserve disjoint H15 ranges while preferring the resident device."""
    def __init__(self, store, helper_smoke_games=0):
        self.bounds = store.bounds
        self.queues = []
        self.smoke = {}
        self.lock = Lock()
        for i, (first, last) in enumerate(store.ranges):
            if helper_smoke_games:
                if last - first <= helper_smoke_games:
                    raise ValueError("the smoke needs H15 games for both owner and helper")
                last -= helper_smoke_games
                self.smoke[i] = (i, last, last + helper_smoke_games)
            # Match disk shard cuts to minimize repeated decompression when sharing work.
            cuts = [first, *range((first // GAMES_PER_SHARD + 1) * GAMES_PER_SHARD,
                                  last, GAMES_PER_SHARD), last]
            self.queues.append(deque((i, a, b) for a, b in zip(cuts[:-1], cuts[1:])))

    def take(self, preferred, helper=False):
        with self.lock:
            if helper and preferred in self.smoke:
                return self.smoke.pop(preferred)
            if not helper and self.queues[preferred]:
                return self.queues[preferred].popleft()
            pending = [queue for queue in self.queues if queue]
            if pending:
                largest = max(pending, key=lambda q: self.bounds[q[-1][2]] - self.bounds[q[0][1]])
                return largest.pop()
        return None


def _wait(futures, abort):
    try:
        for future in as_completed(futures):
            yield futures[future], future.result()
    except BaseException:
        abort.set()
        for future in futures:
            future.cancel()
        raise


def _merge(fit, e2):
    n = len(fit.games)
    return cache.PolicyCache(pd.concat([fit.games, e2.games], ignore_index=True),
        [*fit.shards, *(replace(shard, first=n + shard.first, last=n + shard.last) for shard in e2.shards)],
        torch.cat([fit.arrays["c"], e2.arrays["c"]]), torch.cat([fit.arrays["counts"], e2.arrays["counts"]]),
        np.concatenate([fit.offsets, e2.offsets]))


def overlap(games, n_fit, out, planes_dir, config, nominal_fit, nominal_e2, on_fit, *,
            phase_started, smoke=False, abort=None):
    """Run both teachers, let C10 devices finish H15 fit work, then publish fit before H15 E2."""
    _check_devices_and_games(games)
    if tuple(config["profiles_H"]) != PROFILES:
        raise ValueError("H15 teacher profiles must match the original A column order")
    if not 0 < n_fit < len(games):
        raise ValueError("fit and E2 require distinct nonempty game ranges")
    if abort is None:
        abort = Event()
    fit, e2 = games.iloc[:n_fit].reset_index(drop=True), games.iloc[n_fit:].reset_index(drop=True)
    readers = [PlaneReader(planes_dir, rows, groups=(group,)).batches
               for group, rows in (("fit", fit), ("e2", e2))]
    stores = {}
    for family in ("H", "C"):
        fit_store = cache.TeacherStore(family, fit, config, readers[0], nominal_fit)
        e2_store = cache.TeacherStore(family, e2, config, readers[1], nominal_e2,
                                      resident_bytes=fit_store.nominal_card_bytes, reset_peaks=False)
        stores[family] = (fit_store, e2_store)
    h_fit, h_e2 = stores["H"]
    c_fit, c_e2 = stores["C"]
    work = FitWork(h_fit, config["games_per_batch"] if smoke else 0)

    def fit_h(worker, preferred, helper=False):
        reports = []
        while not abort.is_set() and (job := work.take(preferred, helper)) is not None:
            reports.append(h_fit.process(worker, *job, abort))
        return reports

    def start_h(index):
        worker = cache.TeacherWorker("H", config, h_fit.shards[index].q.device)
        return worker, fit_h(worker, index)

    def start_c(index):
        device = c_fit.shards[index].q.device
        worker = cache.TeacherWorker("C", config, device)
        c_reports = [store.process(worker, index, *store.ranges[index], abort) for store in stores["C"]]
        del worker
        c_ready = time.monotonic() - phase_started
        helpers = []
        task = work.take(index, helper=True) if not abort.is_set() else None
        if task is not None:
            worker = cache.TeacherWorker("H", config, device)
            helpers.append(h_fit.process(worker, *task, abort))
            helpers.extend(fit_h(worker, index, helper=True))
        return c_reports, helpers, c_ready

    matmul, cudnn = torch.backends.cuda.matmul, torch.backends.cudnn
    flags = matmul.allow_tf32, cudnn.allow_tf32
    try:
        matmul.allow_tf32 = cudnn.allow_tf32 = True
        with ThreadPoolExecutor(max_workers=8) as pool:
            h_results, c_results = {}, {}
            futures = {pool.submit(fn, i): (family, i) for family, fn in (("H", start_h), ("C", start_c))
                       for i in range(4)}
            for (family, i), result in _wait(futures, abort):
                (h_results if family == "H" else c_results)[i] = result
            aligned = time.monotonic()
            fit_joint = JointCache(h_fit.finish(), c_fit.finish())
            fit_alignment_seconds = time.monotonic() - aligned
            fit_ready = time.monotonic() - phase_started
            on_fit(fit_joint)
            e2_start = time.monotonic() - phase_started
            futures = {pool.submit(h_e2.process, h_results[i][0], i, *h_e2.ranges[i], abort): i
                       for i in range(4)}
            e2_results = dict(_wait(futures, abort))
            e2_end = time.monotonic() - phase_started
    except BaseException:
        abort.set()
        raise
    finally:
        matmul.allow_tf32, cudnn.allow_tf32 = flags

    aligned = time.monotonic()
    h_cache = _merge(fit_joint.h, h_e2.finish())
    c_cache = _merge(fit_joint.c, c_e2.finish())
    result = JointCache(h_cache, c_cache)
    e2_alignment_seconds = time.monotonic() - aligned
    helper_reports = [report for i in range(4) for report in c_results[i][1]]
    h_workers = [h_results[i][1] + [e2_results[i]] for i in range(4)]
    c_workers = [c_results[i][0] for i in range(4)]
    _publish_overlap(out, stores, result, h_workers, c_workers, helper_reports,
                     max(row[2] for row in c_results.values()), e2_end)
    write_json(out / "teacher_alignment.json", {
        "games": len(games), "positions": int(games.length.sum()), "h15_legal_subset_of_c10": True,
        "seconds": fit_alignment_seconds + e2_alignment_seconds,
        "fit_alignment_seconds": fit_alignment_seconds, "e2_alignment_seconds": e2_alignment_seconds,
    })
    helpers = _worker_reports(helper_reports)
    write_json(out / "teacher_overlap.json", {
        "fit_ready_seconds": fit_ready, "e2_start_seconds": e2_start, "e2_end_seconds": e2_end,
        "elapsed_seconds": time.monotonic() - phase_started,
        "c10_ready_seconds": max(row[2] for row in c_results.values()),
        "h_helper_positions": sum(row["positions"] for row in helpers),
        "h_helper_games": sum(row["games"] for row in helpers),
        "h_helper_peer_bytes": sum(row["peer_bytes"] for row in helpers),
        "h_helpers": helpers, "forced_helper_games_per_card": config["games_per_batch"] if smoke else 0,
    })
    return result


def _worker_reports(chunks):
    grouped = {}
    for row in chunks:
        grouped.setdefault(row["device"], []).append(row)
    reports = []
    for device, rows in sorted(grouped.items()):
        values = {name: sum(row[name] for row in rows) for name in
                  ("positions", "games", "inference_seconds", "input_wait_seconds",
                   "compiled_batches", "eager_batches", "peer_bytes")}
        values["first_compiled_call_seconds"] = next(
            (r["first_compiled_call_seconds"] for r in rows if r["first_compiled_call_seconds"] is not None), None)
        values.update({name: max(row[name] for row in rows) for name in
                       ("peak_allocated_bytes", "peak_reserved_bytes", "setup_seconds")})
        values["errors"] = np.max([row["errors"] for row in rows], axis=0).tolist()
        reports.append(values | {"device": device})
    return reports


def _publish_overlap(out, stores, joint, h_workers, c_workers, helper_reports, c10_ready, e2_end):
    for family, policy, reports, done in (("H", joint.h, h_workers, e2_end), ("C", joint.c, c_workers, c10_ready)):
        fit_store, e2_store = stores[family]
        native = _worker_reports([row for lane in reports for row in lane])
        helpers = _worker_reports(helper_reports) if family == "H" else []
        nominal = [a + b for a, b in zip(fit_store.nominal_card_bytes, e2_store.nominal_card_bytes, strict=True)]
        positions = int(policy.bounds[-1])
        if sum(row["positions"] for row in (*native, *helpers)) != positions:
            raise ValueError(f"{family} teacher work accounting differs from the cache")
        for i, worker in enumerate(native):
            worker["nominal_cache_bytes"] = nominal[i]
        cache.publish_teacher(out, family, policy, {
            "family": family, "games": len(policy.games), "positions": positions, "seconds": done,
            "teacher_passes_per_game": 1, "workers": native, "helpers": helpers,
            "cache_bytes": positions * fit_store.bytes_per_position, "profiles": list(fit_store.profiles),
            "microbatch_positions": fit_store.size, "nominal_cache_bytes": sum(nominal),
            "max_nominal_card_cache_bytes": max(nominal), "input": "zstd_planes",
        })
