"""Publish independent experiment indices; never regenerate original folds."""
import argparse
import json
from .config import read_config, DEFAULT_CONFIG
from .data import prepare


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--release', action='store_true', help='Only after development settings and U evaluation are locked')
    args = parser.parse_args(argv)
    print(json.dumps(prepare(read_config(args.config), release=args.release), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
