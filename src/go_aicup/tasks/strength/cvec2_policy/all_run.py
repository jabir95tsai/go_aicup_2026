"""Prepare, train, and score the fixed A/H15 eight-arm cvec2 policy experiment."""
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from functools import partial
import json
from pathlib import Path
import platform
import subprocess
from threading import Event
from time import monotonic

from filelock import FileLock
import numpy as np
import torch

from go_aicup.artifacts import file_hash, object_hash, write_json
from go_aicup.paths import ROOT
from go_aicup.tasks.strength.policy_large.run import code_sources
from . import all_data, all_report

SEED = 2027
MODELS = all_report.MODEL_ARMS
WEIGHTED = all_report.WEIGHTED_ARMS
TRAIN_DEVICES = (4, 5, 6, 7)


def read_json(path):
    return json.loads(path.read_text())


def sources():
    directory = Path(__file__).parent
    files = [directory / name for name in (
        "__init__.py", "all_data.py", "all_report.py", "all_run.py", "all.slurm",
        "encoder.py", "full_cache.py", "joint.py", "planes.py", "run.py", "train.py", "PLAN.md",
    )]
    files += [ROOT / path for path in (
        "configs/strength/policy_large_200k.json", "src/go_aicup/tasks/strength/humansl.py",
    )]
    files += [ROOT / "scripts/strength" / name for name in (
        "rank_more_players.py", "rank_prod_a0_fusion.py", "humansl_hd_filter.py",
    )]
    return code_sources() | {str(path.relative_to(ROOT)): file_hash(path) for path in files}


def prepare(args):
    started = monotonic()
    out = args.out
    existing = out / "config.json"
    if existing.exists():
        config = read_json(existing)
        check_config(out, config)
        requested = (str(args.stage1_out.resolve()), str(args.folds.resolve()),
                     str(args.planes.resolve()), args.smoke_steps)
        stored = tuple(config[key] for key in ("stage1", "folds", "planes_root", "smoke_steps"))
        if requested != stored:
            raise ValueError("prepared experiment settings differ")
        all_data.load(out / "cpu")
        return config
    from . import full_cache, planes, train

    data = all_data.prepare(out / "cpu", args.stage1_out, args.folds, smoke_steps=args.smoke_steps)
    planes_started = monotonic()
    packed = planes.build(data.groups, args.planes, args.workers, smoke=args.smoke_steps is not None)
    config = {
        "stage1": str(args.stage1_out.resolve()), "folds": str(args.folds.resolve()),
        "seed": SEED, "smoke_steps": args.smoke_steps, "updates": len(data.batches),
        "train_devices": list(TRAIN_DEVICES), "plane_workers": args.workers,
        "fit_projection": all_data.PROJECTION.tolist(),
        "planes_root": str(args.planes.resolve()), "planes": str(packed.resolve()),
        "plane_files": all_report.fingerprint(packed, ("manifest.json", "complete.json")),
        "cpu_manifest": file_hash(out / "cpu/complete.json"), "sources": sources(),
        "batch_per_k": train.QUESTIONS_PER_K_BATCH, "lr": train.LR,
        "weight_decay": train.WEIGHT_DECAY, "position_chunk": train.POSITION_CHUNK,
        "head_c": 1, "head_weight_decay": 0, "weight_concentration_penalty": .01,
        "checkpoint_every": 500, "teacher_config": full_cache.a_config(),
        "schedule": "c10-assists-h15-fit-first",
        "teacher_sources": teacher_sources(),
    }
    write_json(out / "prepare_timing.json", {"seconds": monotonic() - started,
                                             "planes_seconds": monotonic() - planes_started})
    write_json(existing, config)
    return config


def teacher_sources():
    from go_aicup.features.humansl_torch import CKPT

    vectors = ROOT / "outputs/rank/humansl/cvec_v2/learn/vectors.pt"
    return {str(path.relative_to(ROOT)): file_hash(path) for path in (Path(CKPT), vectors)}


def check_config(out, config):
    if (config["sources"] != sources() or config["fit_projection"] != all_data.PROJECTION.tolist()
            or config["cpu_manifest"] != file_hash(out / "cpu/complete.json")):
        raise ValueError("prepared experiment code or CPU manifest changed")
    all_report.verify_files(Path(config["planes"]), config["plane_files"])


def environment(out):
    version = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version,name,memory.total,pci.bus_id", "--format=csv,noheader"], text=True
    )
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
              "gpus": version.splitlines()}
    write_json(out / "environment.json", result)


def run_tasks(out, data, fit_cache, full_ready, identity, abort, phase_started):
    """Train with the aligned fit cache, then extract when both splits are available."""
    from . import joint

    started = monotonic()
    devices = list(TRAIN_DEVICES)
    pending = list(MODELS)
    completed = set()
    tasks, diagnostics = {}, {"train": {}, "extract": {}}
    extracts = []
    update_starts = {}
    full_cache = None
    n_fit = len(data.groups["fit"])

    def mark_first_update(arm):
        update_starts[arm] = monotonic() - phase_started

    with ThreadPoolExecutor(max_workers=len(TRAIN_DEVICES)) as pool:
        try:
            while pending or tasks or extracts:
                if full_cache is None and full_ready.done():
                    full_cache = full_ready.result()
                while devices and (pending or (full_cache is not None and extracts)):
                    phase = "train" if pending else "extract"
                    arm = pending.pop(0) if pending else extracts.pop(0)
                    raw = data.features[arm[0]]
                    slot = devices.pop(0)
                    device = torch.device("cuda", slot)
                    if phase == "train":
                        diagnostics.setdefault("first_train_submit_seconds", monotonic() - phase_started)
                        future = pool.submit(joint.train_one, arm, fit_cache, raw[:n_fit], data.indices,
                                             data.labels, data.k, data.batches, data.heads["A"],
                                             device, out, identity, abort,
                                             on_first_update=partial(mark_first_update, arm))
                    else:
                        diagnostics.setdefault("first_extract_submit_seconds", monotonic() - phase_started)
                        future = pool.submit(joint.extract_one, arm, full_cache, raw, device, out, identity,
                                             n_fit, abort)
                    tasks[future] = (phase, arm, slot)
                waiting = list(tasks)
                if full_cache is None:
                    waiting.append(full_ready)
                ready, _ = wait(waiting, return_when=FIRST_COMPLETED)
                if full_ready in ready:
                    full_cache = full_ready.result()
                    ready.remove(full_ready)
                for future in ready:
                    phase, arm, slot = tasks.pop(future)
                    diagnostics[phase][arm] = future.result() | {"gpu": slot}
                    devices.append(slot)
                    if phase == "train":
                        completed.add(arm)
                        extracts.append(arm)
                        child = f"{arm}-weighted"
                        if child in WEIGHTED:
                            pending.insert(0, child)
        except BaseException:
            abort.set()
            for future in tasks:
                future.cancel()
            raise
    if completed != set(all_report.LEARNED_ARMS):
        raise ValueError("arm scheduler did not complete all five learned arms")
    diagnostics["wall_seconds"] = monotonic() - started
    diagnostics["parallel_arms"] = len(TRAIN_DEVICES)
    diagnostics["first_update_seconds_from_gpu_start"] = update_starts
    return diagnostics


def teacher_parity(cache, expected, path):
    errors = np.abs(cache.arrays["c"].numpy() - expected)
    parity = {"max": float(errors.max()), "mean": float(errors.mean()), "games": len(errors)}
    for family, values in (("H", errors[:, :30]), ("C", errors[:, 30:])):
        parity[family] = {"max": float(values.max()), "mean": float(values.mean())}
    write_json(path, parity)
    if not np.isfinite(errors).all() or any(parity[f]["max"] > .05 or parity[f]["mean"] > .005
                                           for f in ("H", "C")):
        raise ValueError(f"played-move teacher parity failed: {parity}")
    return parity


def check_smoke_teachers(out, overlap):
    if overlap["h_helper_positions"] <= 0 or overlap["h_helper_peer_bytes"] <= 0:
        raise ValueError("smoke did not exercise writing H15 policy from a C10 GPU")
    for family in ("H", "C"):
        workers = read_json(out / f"teacher_{family}.json")["workers"]
        if len(workers) != 4 or any(w["compiled_batches"] < 2 for w in workers):
            raise ValueError(f"smoke did not exercise successive compiled {family} teacher batches")


def gpu_phase(out, data, config):
    from . import full_cache

    marker = out / "gpu_complete.json"
    identity = object_hash(config)
    if marker.exists():
        summary = read_json(marker)
        if summary["identity"] != identity:
            raise ValueError("GPU inputs changed")
        all_report.verify_files(out, summary["files"])
        return summary
    started = monotonic()
    environment(out)
    if config["teacher_sources"] != teacher_sources():
        raise ValueError("teacher weights changed since preparation")
    smoke = config["smoke_steps"] is not None
    n_fit = len(data.groups["fit"])
    nominal_fit = nominal_e2 = None
    if smoke:
        split = all_data.COUNTS["fit"][0]
        nominal_fit, nominal_e2 = data.nominal_lengths[:split], data.nominal_lengths[split:]

    abort = Event()
    fit_ready = Future()
    summary = {}

    def on_fit(cache):
        summary["fit_teacher_parity"] = teacher_parity(cache, data.expected[:n_fit], out / "teacher_fit_parity.json")
        summary["fit_ready_seconds"] = monotonic() - started
        fit_ready.set_result(cache)

    def run_teachers():
        try:
            cache = full_cache.overlap(data.games, n_fit, out, Path(config["planes"]), config["teacher_config"],
                                       nominal_fit, nominal_e2, on_fit, phase_started=started,
                                       smoke=smoke, abort=abort)
            summary["teacher_parity"] = teacher_parity(cache, data.expected, out / "teacher_parity.json")
            summary["teacher_seconds"] = monotonic() - started
            overlap = read_json(out / "teacher_overlap.json")
            summary["e2_start_seconds"] = overlap["e2_start_seconds"]
            summary["e2_end_seconds"] = overlap["e2_end_seconds"]
            if not summary["fit_ready_seconds"] <= summary["e2_start_seconds"] <= summary["e2_end_seconds"]:
                raise ValueError("E2 timestamps differ from the fitting-cache release timeline")
            if smoke:
                check_smoke_teachers(out, overlap)
            return cache
        except BaseException as error:
            abort.set()
            if not fit_ready.done():
                fit_ready.set_exception(error)
            raise

    with ThreadPoolExecutor(max_workers=1) as pool:
        full_ready = pool.submit(run_teachers)
        try:
            summary["arms"] = run_tasks(out, data, fit_ready.result(), full_ready, identity, abort, started)
        except BaseException:
            abort.set()
            raise
    updates = summary["arms"]["first_update_seconds_from_gpu_start"]
    summary["updates_started_before_e2_complete"] = sorted(
        arm for arm, when in updates.items() if when < summary["e2_end_seconds"]
    )
    if smoke and updates and not summary["updates_started_before_e2_complete"]:
        raise ValueError("smoke training did not overlap the E2 teacher stage")
    summary["seconds"] = monotonic() - started
    files = [f"{arm}/{name}" for arm in all_report.LEARNED_ARMS for name in all_report.ARM_FILES]
    files += ["teacher_parity.json", "environment.json", "teacher_H.json", "teacher_C.json",
              "teacher_alignment.json", "offset_H.npy", "offset_C.npy", "teacher_fit_parity.json",
              "teacher_overlap.json"]
    gpu = {"identity": identity, "files": all_report.fingerprint(out, files), "diagnostics": summary}
    write_json(marker, gpu)
    return gpu


def execute(args):
    if args.phase == "preflight":
        data = all_data.preflight(args.stage1_out, args.folds)
        print(json.dumps(data.identity, indent=2), flush=True)
        return
    if args.out is None or args.workers < 1:
        raise SystemExit("--out and positive --workers are required")
    args.out.mkdir(parents=True, exist_ok=True)
    with FileLock(args.out / "run.lock", timeout=0):
        if args.phase == "prepare":
            config = prepare(args)
            print(json.dumps({"planes": config["planes"], "updates": config["updates"]}), flush=True)
            return
        config = read_json(args.out / "config.json")
        check_config(args.out, config)
        data = all_data.load(args.out / "cpu")
        if args.phase == "train":
            gpu_phase(args.out, data, config)
            return
        gpu = read_json(args.out / "gpu_complete.json")
        if gpu["identity"] != object_hash(config):
            raise ValueError("GPU outputs do not match the evaluation inputs")
        all_report.verify_files(args.out, gpu["files"])
        report = all_report.score(args.out, data, config, gpu)
        print(json.dumps({"scores": {arm: result["score"] for arm, result in report["scores"].items()},
                          "gate": report["gate"]}, indent=2), flush=True)
