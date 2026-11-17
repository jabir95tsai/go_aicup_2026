#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/pid/e2e_adam/train-%j.out
#SBATCH --exclude=25a-hgpn006,25a-hgpn007
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
P="$HOME/.conda/envs/go/bin/python"
ARM="${1:?expected lr3e5 or lr1e5}"
case "$ARM" in lr3e5|lr1e5) ;; *) exit 2 ;; esac
"$P" - "$ARM" <<'PY'
import json
import sys
from pathlib import Path
arm = sys.argv[1]
report = json.loads(Path('outputs/pid/ge2e/pid_e2e_adam_probe/report.json').read_text())
if report.get('pass') is not True:
    raise SystemExit('zero-LR probe did not pass')
config = json.loads(Path(f'configs/identity/pid_e2e_adam_{arm}.json').read_text())
config['e2e']['freeze_bn'] = report['freeze_bn']
out = Path(f'outputs/pid/e2e_adam/{arm}.json')
if out.exists() and json.loads(out.read_text()) != config:
    raise SystemExit('refusing to change the resolved training config')
out.write_text(json.dumps(config, indent=2) + '\n')
print(json.dumps({'arm': arm, 'lr': config['optimizer']['lr'], 'freeze_bn': report['freeze_bn']}), flush=True)
PY
"$P" -m go_aicup.tasks.identity.train_e2e --config "outputs/pid/e2e_adam/$ARM.json"
RUN="pid_e2e_adam_$ARM"
CACHE="data/processed/identity/e2e_adam/$ARM"
"$P" -m scripts.identity.pid2_seqagg extract --run "$RUN" --cache "$CACHE" --parts bench --bench-run pid_ge2e_v4_d4_hd_9600
"$P" -m scripts.identity.pid2_seqagg compare \
    --arms "base=/home/jason1122/pid2_seqagg_cache:outputs/pid/pid2_seqagg/head_B/best.pt" \
           "$ARM=$CACHE:outputs/pid/ge2e/$RUN/head.pt" \
    --base base --pools 0 --first 100 --out "outputs/pid/e2e_adam/dev_$ARM"
