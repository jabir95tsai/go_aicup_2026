#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=pid2-cheap-tta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=02:30:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/pid/cheap/tta-%x-%j.out
#SBATCH --exclude=25a-hgpn006
# A-c D4 TTA game vectors for one encoder (2026-11-20; registered in RESULTS.md before the run).
# usage: sbatch cheap_tta.slurm B|A1|A2 [limit]
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=12
P=/home/jason1122/.conda/envs/go/bin/python
df -h /home/jason1122 /work/jason1122 | tail -2
$P -m scripts.identity.pid2_cheap tta --enc "$1" --limit "${2:-0}"
