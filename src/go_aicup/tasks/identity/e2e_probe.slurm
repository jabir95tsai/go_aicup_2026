#!/bin/bash
#SBATCH --partition=dev
#SBATCH --account=GOV115094
#SBATCH --job-name=pid-e2e-zero-bn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=00:30:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/pid/e2e_adam/probe-%j.out
#SBATCH --exclude=25a-hgpn006,25a-hgpn007
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
P="$HOME/.conda/envs/go/bin/python"
"$P" -m pytest -q tests/test_identity_e2e.py tests/test_identity_ge2e.py
"$P" -m go_aicup.tasks.identity.e2e_probe --config configs/identity/pid_e2e_adam_probe.json
