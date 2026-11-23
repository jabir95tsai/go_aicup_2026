#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=rank-opp-refine
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=02:30:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/rank/opp_link2/refine-%j.out
#SBATCH --exclude=25a-hgpn006
# Opponent identity link v2 refinement round (scripts/strength/rank_opp_refine.py, RESULTS 2026-11-23). CPU.
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
P=/home/jason1122/.conda/envs/go/bin/python
start=$(date +%s)
$P -m scripts.strength.rank_opp_refine --job "${SLURM_JOB_ID}" ${REFINE_ARGS:-}
echo "seconds=$(( $(date +%s) - start ))"
