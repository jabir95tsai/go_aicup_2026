"""V1 CLI for the shared split and player-ply index preparation."""
import argparse
from pathlib import Path

from go_aicup.tasks.strength.shared.prepare import prepare, write_sample_indices
from go_aicup.tasks.strength.v1.config import DEFAULT_CONFIG, read_config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    print(f"Dataset manifest: {prepare(read_config(args.config))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
