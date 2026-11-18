#!/bin/bash
#SBATCH --job-name=rank-policy-h-zip
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/slurm-%x-%j.out
# Move the finished H cache into <dir>.zip: 11 worker processes deflate/hash/verify shards in parallel, one writer appends.
set -euo pipefail
export OMP_NUM_THREADS=1
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1
SRC=data/processed/strength_features/humansl/policy_cnn_h/cached_AB_mondev_seed2027_h9
PYTHON_BIN="${HOME}/.conda/envs/go/bin/python"
"$PYTHON_BIN" - "$SRC" <<'PY'
import json, sys
from pathlib import Path
marker = json.loads((Path(sys.argv[1]) / 'policy/complete.json').read_text())
if marker['status'] != 'complete' or marker['positions'] != 5343778:
    raise SystemExit(f'policy cache incomplete: {marker}')
PY
"$PYTHON_BIN" -m go_aicup.tasks.strength.policy_h.zip_parallel "$SRC" --level 1 --workers 11 --batch 32
