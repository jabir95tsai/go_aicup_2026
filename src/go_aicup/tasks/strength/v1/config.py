"""V1-only configuration defaults and validation."""
import json
from pathlib import Path

from go_aicup.paths import ROOT

DEFAULT_CONFIG = ROOT / "configs" / "strength" / "rank_cnn_v1.json"


def read_config(path: Path = DEFAULT_CONFIG) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["format_version"] != 1 or config["cache_codec"] != "packed":
        raise ValueError("unsupported V1 configuration")
    for key, value in config["paths"].items():
        target = Path(value)
        config["paths"][key] = str((target if target.is_absolute() else ROOT / target).resolve())
    if config["precision"] not in ("bf16", "fp16", "fp32"):
        raise ValueError("precision must be bf16, fp16 or fp32")
    if config["device"] not in ("cuda", "cpu"):
        raise ValueError("device must be cuda or cpu")
    if config["training"]["epochs"] != 1:
        raise ValueError("compact V1 training requires exactly one epoch")
    if config["training"].get("evaluate_every_steps", 100) < 1:
        raise ValueError("evaluate_every_steps must be positive")
    if config["training"]["global_batch_size"] < 10:
        raise ValueError("global batch must fit all ten ranks")
    if config["device"] == "cpu" and config["precision"] != "fp32":
        raise ValueError("CPU development requires explicit fp32 precision")
    return config


