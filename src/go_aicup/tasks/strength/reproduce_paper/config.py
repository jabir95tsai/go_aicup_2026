"""SE competition-port configuration; importing performs no I/O."""
import json
from pathlib import Path
from go_aicup.paths import ROOT

SPLIT_SEED = 2026
DEFAULT_TRAINING_SEED = 2026

DEFAULT_CONFIG = ROOT / "configs" / "strength" / "se_seed2026.json"


def read_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    for name, value in config["paths"].items():
        target = Path(value)
        config["paths"][name] = str((target if target.is_absolute() else ROOT / target).resolve())
    return config


def validate_config(config):
    if config["format_version"] != 1:
        raise ValueError("unsupported SE config")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(config["split_seed"]) is not int or config["split_seed"] != SPLIT_SEED:
        raise ValueError("reproduce_paper split_seed must be fixed at 2026")
    for key in ("expected_games_per_rank", "train_games_per_rank"):
        if type(config["split"][key]) is not int or config["split"][key] < 1:
            raise ValueError("split counts must be positive integers")
    if not config["split"]["train_games_per_rank"] < config["split"]["expected_games_per_rank"]:
        raise ValueError("training count must be smaller than total count")
    positions = config["positions"]
    if type(positions["min_ply"]) is not int or positions["min_ply"] < 0:
        raise ValueError("min_ply must be a nonnegative integer")
    if positions["positions_per_rank"] != 7:
        raise ValueError("SE requires seven distinct positions per rank")
    if type(config["loader"]["max_open_shards"]) is not int or config["loader"]["max_open_shards"] < 1:
        raise ValueError("max_open_shards must be positive")
    if config["model"]["input_encoding"] != "pre_action_own_opponent" or config["model"]["policy_size"] != 362:
        raise ValueError("SE requires pre-action own/opponent inputs and 362 policy classes")

    count = config["questions"]["validation"]
    if type(count) is not int or count < 40 or count % 40:
        raise ValueError("validation questions must be a positive multiple of 40")
