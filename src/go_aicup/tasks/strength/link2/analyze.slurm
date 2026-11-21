#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=rank-link2-analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/rank/link2/analyze-%j.out
#SBATCH --exclude=25a-hgpn006
# L10 v2 analysis (CPU; the partition needs one GPU): (b) pools and (c) pairs first, then (a) with the
# opening scores, then the overlay file if any question is linked. SHARDS must match the embed job.
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
P=/home/jason1122/.conda/envs/go/bin/python
start=$(date +%s)
$P -m scripts.strength.rank_link2 analyze --shards "${SHARDS:-4}"
$P -m scripts.strength.rank_link2 export --job "${SLURM_JOB_ID}"
echo "seconds=$(( $(date +%s) - start ))"
