#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=pid2-lik-export
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/pid/likelihood/export-%j.out
#SBATCH --exclude=25a-hgpn006
# Player-likelihood stage 0 official export (passed the 11/20 switch rule, 2026-11-20).
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
P=/home/jason1122/.conda/envs/go/bin/python
$P -m scripts.identity.pid2_likelihood selfcheck
$P -m scripts.identity.pid2_likelihood export
