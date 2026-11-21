#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=pid2-lik-ps2exp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/pid/likelihood/export-%j.out
#SBATCH --exclude=25a-hgpn006
# Export of the pseudo-label retune (passed or probe); first reproduces 824896-comb-pseudo.csv.
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=12
P=/home/jason1122/.conda/envs/go/bin/python
$P -m scripts.identity.pid2_likelihood selfcheck
$P -m scripts.identity.pid2_likelihood export --eb --combine ${COLOUR:-} --pseudo --pseudo2
