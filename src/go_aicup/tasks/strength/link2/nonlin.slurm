#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=rank-nonlin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=02:30:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/rank/nonlin/nonlin-%j.out
#SBATCH --exclude=25a-hgpn006
# Nonlinear per-game feature extension (scripts/strength/rank_nonlin.py): local five seeds, export if both sets improve. CPU.
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
P=/home/jason1122/.conda/envs/go/bin/python
start=$(date +%s)
$P -m scripts.strength.rank_nonlin --job "${SLURM_JOB_ID}" ${NONLIN_ARGS:-}
echo "seconds=$(( $(date +%s) - start ))"
