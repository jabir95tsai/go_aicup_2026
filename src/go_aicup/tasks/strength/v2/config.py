"""V2 configuration defaults; no data access or device initialization."""
import json
from pathlib import Path

from go_aicup.paths import ROOT

DEFAULT_CONFIG = ROOT / "configs" / "strength" / "rank_cnn_v2_seed2027.json"


def read_config(path: Path = DEFAULT_CONFIG) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if (config["format_version"] != 1 or config.get("strength_version") != 2
            or config["cache_codec"] != "packed"):
        raise ValueError("unsupported V2 configuration")
    for key, value in config["paths"].items():
        target = Path(value)
        config["paths"][key] = str((target if target.is_absolute() else ROOT / target).resolve())
    if config["precision"] not in ("bf16", "fp16", "fp32"):
        raise ValueError("precision must be bf16, fp16 or fp32")
    if config["device"] not in ("cuda", "cpu"):
        raise ValueError("device must be cuda or cpu")
    if config["device"] == "cpu" and config["precision"] != "fp32":
        raise ValueError("CPU development requires explicit fp32 precision")
    if config["training"]["epochs"] != 1:
        raise ValueError("V2 first run requires exactly one epoch")
    for batch in (config["training"]["global_batch_size"], config["evaluation"]["global_batch_size"]):
        if type(batch) is not int or batch < 1:
            raise ValueError("global batch sizes must be positive integers")
    if config["training"]["evaluate_every_steps"] < 1:
        raise ValueError("evaluate_every_steps must be positive")
    split = config["split"]
    if (split.get("policy") != "player_disjoint" or set(split) != {"policy", "seen_holdout_fraction"}
            or not 0 < split["seen_holdout_fraction"] < 1):
        raise ValueError("V2 requires the player-disjoint split with a seen-player holdout fraction")
    monitoring = config["monitoring"]
    count = monitoring["games_per_rank"]
    if type(count) is not int or count < 1:
        raise ValueError("monitoring games_per_rank must be a positive integer")
    if monitoring["score_unit"] != "game" or monitoring["decoder"] != "argmax":
        raise ValueError("V2 monitoring requires game-level argmax scoring")
    if config["evaluation"]["question_decoder"] != "score_rank":
        raise ValueError("V2 official question decoder must be score_rank")
    return config
