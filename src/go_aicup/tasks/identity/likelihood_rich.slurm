#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=pid2-lik-rich
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=03:30:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/pid/likelihood/rich-%j.out
#SBATCH --exclude=25a-hgpn006
# Rich per-move features for the player-likelihood term (2026-11-21, PID_LIKELIHOOD_RICH_PLAN.md):
# board-replay features (stops if the replay extrapolates past 1 h), then selection and sealed test.
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=12 NUMBA_NUM_THREADS=12
P=/home/jason1122/.conda/envs/go/bin/python
df -h /work/jason1122 | tail -1
$P -m scripts.identity.pid2_likelihood selfcheck
[ -f data/processed/identity/likelihood/rich/meta.json ] || $P -m scripts.identity.pid2_likelihood rich-features
$P -m scripts.identity.pid2_likelihood evaluate3
