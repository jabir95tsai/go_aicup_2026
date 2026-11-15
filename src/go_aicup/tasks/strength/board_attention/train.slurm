#!/bin/bash
#SBATCH --job-name=rank-board-attn
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/slurm-%x-%j.out
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHON_BIN="${HOME}/.conda/envs/go/bin/python"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
"$PYTHON_BIN" -m go_aicup.tasks.strength.board_attention.train "${MODE:-train}"
