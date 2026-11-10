"""Publish independent experiment indices; never regenerate original folds."""
import argparse
import json
from .config import read_config, DEFAULT_CONFIG
from .data import prepare, prepare_extension


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--extension', action='store_true',
                        help='Build additive capped pools (fit_all; with --release also M+U release pools)')
    parser.add_argument('--release', action='store_true', help='Only after development settings and U evaluation are locked')
    args = parser.parse_args(argv)
    config = read_config(args.config)
    if args.extension:
        result = prepare_extension(config, release=args.release)
    elif args.release:
        raise SystemExit('release retraining was removed; use --extension --release after the U lock')
    else:
        result = prepare(config)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
