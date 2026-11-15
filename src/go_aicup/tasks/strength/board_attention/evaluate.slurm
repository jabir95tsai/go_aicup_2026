#!/bin/bash
#SBATCH --job-name=rank-board-b2
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=00:20:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/slurm-%x-%j.out
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"${HOME}/.conda/envs/go/bin/python" -m go_aicup.tasks.strength.board_attention.evaluate --jobs 5
