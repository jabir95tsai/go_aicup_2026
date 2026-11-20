#!/bin/bash
#SBATCH --job-name=rank-cvec2-policy-all
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --nodes=1
#SBATCH --no-requeue
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --exclude=25a-hgpn007,25a-hgpn069,25a-hgpn083
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/slurm-%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/go/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
TASK_CACHE=$(mktemp -d /dev/shm/cvec2-policy-all-XXXXXX)
GPU_MONITOR=
cleanup() {
  if [[ -n "$GPU_MONITOR" ]]; then
    kill "$GPU_MONITOR" 2>/dev/null || true
    wait "$GPU_MONITOR" 2>/dev/null || true
  fi
  rm -rf -- "$TASK_CACHE"
}
trap cleanup EXIT
export TORCHINDUCTOR_CACHE_DIR="$TASK_CACHE/inductor" TRITON_CACHE_DIR="$TASK_CACHE/triton"
OUT="${OUT:-outputs/rank/humansl/cvec2_policy/all/$SLURM_JOB_ID}"
MODULE=go_aicup.tasks.strength.cvec2_policy.run
TIMEFORMAT="%R wall seconds"
mkdir -p "$OUT"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader,nounits \
  --loop=2 --filename="$OUT/gpu_samples.csv" &
GPU_MONITOR=$!
if [[ -n "${SMOKE_STEPS:-}" ]]; then
  time python -m "$MODULE" all --phase prepare --out "$OUT" --smoke-steps "$SMOKE_STEPS" \
    --workers "${PLANE_WORKERS:-24}"
fi
time python -m "$MODULE" all --phase train --out "$OUT"
if [[ -n "${SMOKE_STEPS:-}" ]]; then
  time python -m "$MODULE" all --phase score --out "$OUT"
fi
