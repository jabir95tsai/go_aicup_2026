#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=pid2-cheap-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/pid/cheap/eval-%j.out
#SBATCH --exclude=25a-hgpn006
# A-a / A-b / A-c selection and sealed evaluation (2026-11-20; registered in RESULTS.md before the run).
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=12
P=/home/jason1122/.conda/envs/go/bin/python
$P -m scripts.identity.pid2_cheap selfcheck
$P -m scripts.identity.pid2_cheap evaluate
