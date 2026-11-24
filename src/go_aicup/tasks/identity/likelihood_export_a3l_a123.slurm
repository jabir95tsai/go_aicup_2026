#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=pid2-lik-psexp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/pid/likelihood/export-%j.out
#SBATCH --exclude=25a-hgpn006
# 824896 setting with A1 / A2 columns = 2/3 own + 1/3 A3L-9600 (A123w form; 2026-11-24, user-requested, RESULTS.md).
# First the A1_23w export must reproduce 851729-comb-pseudo-a3l.csv byte for byte.
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=12
P=/home/jason1122/.conda/envs/go/bin/python
$P -m scripts.identity.pid2_likelihood export --eb --combine --pseudo --a3 --a3l
cmp "submissions/pid/${SLURM_JOB_ID}-comb-pseudo-a3l.csv" submissions/pid/851729-comb-pseudo-a3l.csv
rm "submissions/pid/${SLURM_JOB_ID}-comb-pseudo-a3l.csv"   # identical copy made by this job
echo REPRODUCED_851729
$P -m scripts.identity.pid2_likelihood export --eb --combine --pseudo --a3 --a3l --a123
