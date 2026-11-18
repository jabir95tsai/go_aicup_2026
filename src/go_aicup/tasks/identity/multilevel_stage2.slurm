#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=pid2-multilevel-s2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/pid/pid2_multilevel/stage2-%j.out
#SBATCH --exclude=25a-hgpn006
# Multi-level opening back-off, stage 2 (2026-11-18; registered in RESULTS.md before the run).
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=12
P=/home/jason1122/.conda/envs/go/bin/python
[ -f outputs/pid/pid2_multilevel/stage2/keys2.npz ] || $P -m scripts.identity.pid2_multilevel_stage2 keys --workers 12
$P -m scripts.identity.pid2_multilevel_stage2 run
