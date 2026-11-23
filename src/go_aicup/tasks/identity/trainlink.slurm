#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=pid-trainlink
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/pid/trainlink/trainlink-%j.out
#SBATCH --exclude=25a-hgpn006
# Added-set candidates vs training players, stage 0 (scripts/identity/pid2_trainlink.py, RESULTS 2026-11-23).
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1
P=/home/jason1122/.conda/envs/go/bin/python
start=$(date +%s)
OMP_NUM_THREADS=8 $P -m scripts.identity.pid2_trainlink embed
echo "embed seconds=$(( $(date +%s) - start ))"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $P -m scripts.identity.pid2_trainlink score --workers 11
$P -m scripts.identity.pid2_trainlink analyze
echo "seconds=$(( $(date +%s) - start ))"
