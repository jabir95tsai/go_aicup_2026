#!/bin/bash
#SBATCH --job-name=rank-cvec2-policy-s2
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
: "${STAGE1_OUT:?completed stage 1 directory required}"
: "${ANCHOR:?select B or C explicitly}"
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
TASK_CACHE=$(mktemp -d /dev/shm/cvec2-policy-XXXXXX)
trap 'rm -rf -- "$TASK_CACHE"' EXIT
export TORCHINDUCTOR_CACHE_DIR="$TASK_CACHE/inductor" TRITON_CACHE_DIR="$TASK_CACHE/triton"
OUT="${OUT:-outputs/rank/humansl/cvec2_policy/stage2/$SLURM_JOB_ID}"
"${HOME}/.conda/envs/go/bin/python" -m go_aicup.tasks.strength.cvec2_policy.run stage2 \
  --out "$OUT" --stage1-out "$STAGE1_OUT" --anchor "$ANCHOR" --workers 8
