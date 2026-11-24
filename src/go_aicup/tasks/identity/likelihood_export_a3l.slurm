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
# 824896 setting with the A2 part = mean(A2, A3L-9600) (2026-11-24, user-requested exception, RESULTS.md).
# First the unmodified export must reproduce 824896-comb-pseudo.csv byte for byte.
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=12
P=/home/jason1122/.conda/envs/go/bin/python
$P -m scripts.identity.pid2_likelihood selfcheck
$P -m scripts.identity.pid2_likelihood export --eb --combine --pseudo
cmp "submissions/pid/${SLURM_JOB_ID}-comb-pseudo.csv" submissions/pid/824896-comb-pseudo.csv
rm "submissions/pid/${SLURM_JOB_ID}-comb-pseudo.csv"   # identical copy made by this job
echo REPRODUCED_824896
$P -m scripts.identity.pid2_likelihood export --eb --combine --pseudo --a3 --a3l
