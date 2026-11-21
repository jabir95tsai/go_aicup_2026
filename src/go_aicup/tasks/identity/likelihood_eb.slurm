#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=pid2-lik-eb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/pid/likelihood/eb-%j.out
#SBATCH --exclude=25a-hgpn006
# Empirical-Bayes prior for the player-likelihood term (2026-11-21, PID_EB_COMBINER_PLAN.md plan 1):
# Sigma from pool-0 block fits, selection on dev draws 3200-3219, sealed test 3300- against RICH.
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=12
P=/home/jason1122/.conda/envs/go/bin/python
df -h /work/jason1122 | tail -1
$P -m scripts.identity.pid2_likelihood selfcheck
$P -m scripts.identity.pid2_likelihood evaluate4
