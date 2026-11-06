"""Explicit experiment steps; importing never prepares data or trains models."""
import argparse
import json
from pathlib import Path

from .common import DEFAULT_CONFIG, read_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare', help='Freeze metadata, fitting questions and U_newgames; CPU only')
    extract = sub.add_parser('extract', help='Fill selected frozen per-game features; needs allocation')
    extract.add_argument('--component', required=True, choices=('se3', 'h', 'c'))
    extract.add_argument('--scope', choices=('dev', 'fit', 'newgames', 'old_u'), default='fit')
    extract.add_argument('--work', type=Path, help='Prepared release work directory, replaces training scope')
    extract.add_argument('--shard', type=int, default=0); extract.add_argument('--shards', type=int, default=1)
    extract.add_argument('--limit', type=int, help='Stratified global game cap for a benchmark')
    extract.add_argument('--device', choices=('cpu', 'cuda'), default=None)
    train = sub.add_parser('train', help='CPU cached-feature head training; does not run upstream models')
    train.add_argument('--stage', required=True, choices=('dev', 'final'))
    train.add_argument('--arm', required=True, choices=('a', 'se3'))
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('--stage', required=True, choices=('dev', 'final'))
    evaluate.add_argument('--role', required=True, choices=('select', 'newgames', 'old_u'))
    release = sub.add_parser('prepare-release')
    release.add_argument('--store', required=True, type=Path)
    release.add_argument('--name', required=True, choices=('rank_public', 'rank_private'))
    release.add_argument('--work', required=True, type=Path)
    aggregate = sub.add_parser('aggregate')
    aggregate.add_argument('--work', required=True, type=Path)
    aggregate.add_argument('--fallback', type=Path, help='Complete AH+C rankvec_predictions.parquet')
    export = sub.add_parser('export')
    export.add_argument('--work-root', required=True, type=Path)
    export.add_argument('--sample', required=True, type=Path)
    export.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    config = read_config(args.config)
    if args.command == 'prepare':
        from .data import prepare
        result = prepare(config)
    elif args.command == 'extract':
        from .features import extract
        result = extract(config, args.component, args.scope, release=args.work, shard=args.shard,
                         shards=args.shards, limit=args.limit, device=args.device)
    elif args.command == 'train':
        from .train import train
        result = train(config, args.stage, args.arm)
    elif args.command == 'evaluate':
        from .evaluate import evaluate
        result = evaluate(config, args.stage, args.role)
    elif args.command == 'prepare-release':
        from .predict import prepare_release
        q = prepare_release(config, args.store, args.name, args.work)
        result = dict(questions=len(q), work=str(args.work))
    elif args.command == 'aggregate':
        from .predict import aggregate
        result = aggregate(config, args.work, args.fallback)
    else:
        from .predict import export
        result = export(args.work_root, args.sample, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
