#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=rank-qmerge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=01:30:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/rank/qmerge/qmerge-%j.out
#SBATCH --exclude=25a-hgpn006
# Same-player question merge (RANK_QMERGE_PLAN.md): export (must reproduce 751959-cvec2) then Valid-U simulation. CPU only.
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
P=/home/jason1122/.conda/envs/go/bin/python
start=$(date +%s)
$P -m scripts.strength.rank_qmerge --job "${SLURM_JOB_ID}"
echo "seconds=$(( $(date +%s) - start ))"
