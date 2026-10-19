"""Prepare V2 splits, fixed questions and whole-game monitoring indices."""
import argparse
from pathlib import Path

from go_aicup.tasks.strength.shared.prepare import prepare as prepare_shared
from go_aicup.tasks.strength.v2.config import DEFAULT_CONFIG, read_config


def prepare(config: dict) -> Path:
    if config.get("strength_version") != 2 or "monitoring" not in config:
        raise ValueError("V2 preparation requires a V2 monitoring configuration")
    return prepare_shared(config)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    print(f"Dataset manifest: {prepare(read_config(args.config))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
