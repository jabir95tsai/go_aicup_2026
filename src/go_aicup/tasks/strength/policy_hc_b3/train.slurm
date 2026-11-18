#!/bin/bash
#SBATCH --job-name=rank-policy-hc-b3
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=400G
#SBATCH --time=01:00:00
#SBATCH --exclude=25a-hgpn007,25a-hgpn069,25a-hgpn083
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/slurm-%x-%j.out
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CONFIG="${CONFIG:-configs/strength/policy_hc_b3.json}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"${HOME}/.conda/envs/go/bin/python" -m go_aicup.tasks.strength.policy_hc_b3.train --config "$CONFIG"
