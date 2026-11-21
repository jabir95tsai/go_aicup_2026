#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=pid2-lik-pseudo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/pid/likelihood/pseudo-%j.out
#SBATCH --exclude=25a-hgpn006
# Pseudo-label expansion of every fusion part (2026-11-21, PID_PSEUDO_COLOUR_PLAN.md plan 1):
# oracle bound + tau x rounds on dev draws 3800-3819, sealed test 3900- against the base fusion.
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=12
P=/home/jason1122/.conda/envs/go/bin/python
df -h /work/jason1122 | tail -1
$P -m scripts.identity.pid2_likelihood selfcheck
$P -m scripts.identity.pid2_likelihood evaluate8
