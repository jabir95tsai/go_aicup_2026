#!/bin/bash
#SBATCH --job-name=rank-policy-export
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --nodes=1
#SBATCH --no-requeue
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --exclude=25a-hgpn007,25a-hgpn069,25a-hgpn083
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/slurm-%x-%j.out
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
TASK_CACHE=$(mktemp -d /dev/shm/policy-export-XXXXXX)
trap 'rm -rf -- "$TASK_CACHE"' EXIT
export TORCHINDUCTOR_CACHE_DIR="$TASK_CACHE/inductor" TRITON_CACHE_DIR="$TASK_CACHE/triton"
RUN=outputs/rank/humansl/policy_large/200k_deep_seed2027/806639
"${HOME}/.conda/envs/go/bin/python" -m go_aicup.tasks.strength.policy_large.export \
    --run "$RUN" --out "$RUN/test_export"
