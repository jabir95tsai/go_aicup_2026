#!/bin/bash
#SBATCH --job-name=rank-policy-branches
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:5
#SBATCH --cpus-per-task=60
#SBATCH --mem=500G
#SBATCH --time=02:00:00
#SBATCH --exclude=25a-hgpn007,25a-hgpn069,25a-hgpn083
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/slurm-%x-%j.out
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON_BIN="${HOME}/.conda/envs/go/bin/python"
"$PYTHON_BIN" -m go_aicup.tasks.strength.policy_residual.branches
