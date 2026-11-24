#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=rank-opp-rich
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=800G
#SBATCH --time=02:45:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/rank/opp_llr/rich-%j.out
#SBATCH --exclude=25a-hgpn006
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
P=/home/jason1122/.conda/envs/go/bin/python
run=outputs/rank/opp_llr
IFS=, read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
N=${#GPUS[@]}
if (( $(df --output=avail -B1 /dev/shm | tail -n 1) < 16 * 1024**3 )); then
  echo 'Need at least 16 GiB of node shared memory for g40 policy shards' >&2
  exit 1
fi
scratch=$(mktemp -d /dev/shm/go-opp-rich.XXXXXX)
trap 'rm -rf -- "$scratch"' EXIT
"$P" - "$scratch" <<'PY'
import sys
from pathlib import Path
import pandas as pd

out = Path('outputs/rank/opp_llr')
jobs = pd.concat([pd.read_parquet(out / name) for name in ('jobs.parquet', 'jobs_g40.parquet')], ignore_index=True)
if jobs['id'].duplicated().any():
    raise ValueError('duplicate jobs in g40 policy')
jobs.to_parquet(Path(sys.argv[1]) / 'jobs.parquet', index=False)
PY
pids=()
for ((i=0; i<N; i++)); do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} "$P" -m scripts.strength.rank_opp_llr policy \
    --jobs "$scratch/jobs.parquet" --prefix "$scratch/shard_" --shard "$i" --shards "$N" --workers 10 \
    > "$run/rich-policy-${SLURM_JOB_ID}-${i}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
tail -n 1 "$run"/rich-policy-"${SLURM_JOB_ID}"-*.log
[[ $status -eq 0 ]] || exit 1
CUDA_VISIBLE_DEVICES=${GPUS[0]} OMP_NUM_THREADS=32 NUMBA_NUM_THREADS=32 "$P" \
  -m scripts.strength.rank_opp_llr llr --shards "$N" --g40 --rich-dir "$scratch"
OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 "$P" \
  -m scripts.strength.rank_opp_next rich --job "$SLURM_JOB_ID"
