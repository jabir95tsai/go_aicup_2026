#!/bin/bash
#SBATCH --job-name=rank-cvec2-full-pool
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --nodes=1
#SBATCH --no-requeue
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=200G
#SBATCH --time=16:00:00
#SBATCH --exclude=25a-hgpn007,25a-hgpn069,25a-hgpn083
#SBATCH --output=outputs/slurm-%x-%j.out
set -euo pipefail
export PATH="$HOME/.conda/envs/go/bin:$PATH"
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
scratch=$(mktemp -d /dev/shm/cvec2-full-pool-XXXXXX)
trap 'rm -rf -- "$scratch"' EXIT
export TORCHINDUCTOR_CACHE_DIR="$scratch/inductor" TRITON_CACHE_DIR="$scratch/triton"
source_run="${SOURCE_RUN:-outputs/rank/humansl/cvec2_policy/all/821286-mask-reuse}"
out="${OUT:-outputs/rank/humansl/cvec2_policy/full_pool/$SLURM_JOB_ID}"
module=go_aicup.tasks.strength.cvec2_policy.full_pool
python -m "$module" train --source "$source_run" --out "$out"
python -m "$module" export --out "$out"
