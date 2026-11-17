#!/bin/bash
#SBATCH --partition=dev
#SBATCH --account=GOV115094
#SBATCH --job-name=pid2-multilevel-diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=00:30:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/pid/pid2_multilevel/diag-%j.out
#SBATCH --exclude=25a-hgpn006,25a-hgpn007
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
P="$HOME/.conda/envs/go/bin/python"
"$P" -m pytest -q tests/test_pid2_multilevel.py
"$P" -m scripts.identity.pid2_multilevel keys --workers 12
"$P" -m scripts.identity.pid2_multilevel diag
