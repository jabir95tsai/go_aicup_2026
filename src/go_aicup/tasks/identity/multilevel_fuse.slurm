#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=pid2-multilevel-s1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/pid/pid2_multilevel/stage1-%j.out
#SBATCH --exclude=25a-hgpn006
# Multi-level opening back-off, stage 1 (2026-11-17; registered in RESULTS.md before the run).
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=12
P=/home/jason1122/.conda/envs/go/bin/python
[ -f outputs/pid/pid2_multilevel/stage1/pop_keys.npz ] || $P -m scripts.identity.pid2_multilevel_fuse popkeys --workers 12
$P -m scripts.identity.pid2_multilevel_fuse run
