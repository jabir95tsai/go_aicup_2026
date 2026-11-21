#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=pid2-lik-pseudo2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/pid/likelihood/pseudo2-%j.out
#SBATCH --exclude=25a-hgpn006
# Pseudo-label retune (tau 0.2-0.5 x rounds 2-3) + cross-fitted weight refit (2026-11-21, plan A after 824896):
# dev draws 4200-4219, sealed 4300- against the current base (tau 0.5, 2 rounds, current weights).
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=12
P=/home/jason1122/.conda/envs/go/bin/python
df -h /work/jason1122 | tail -1
$P -m scripts.identity.pid2_likelihood selfcheck
$P -m scripts.identity.pid2_likelihood evaluate9
