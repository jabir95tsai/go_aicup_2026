#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=rank-opp-next
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/rank/opp_llr/next-%j.out
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
python_path=/home/jason1122/.conda/envs/go/bin/python
if [[ ${1-} == causes ]]; then
  shift
  exec "$python_path" -m scripts.strength.rank_opp_causes --job "$SLURM_JOB_ID" "$@"
fi
if [[ ${1-} == experiments ]]; then
  shift
  exec "$python_path" -m scripts.strength.rank_opp_experiments --job "$SLURM_JOB_ID" "$@"
fi
if [[ ${1-} == llr-mix ]]; then
  shift
  exec "$python_path" -m scripts.strength.rank_opp_llr_mix --job "$SLURM_JOB_ID" "$@"
fi
exec "$python_path" -m scripts.strength.rank_opp_next "$@" --job "$SLURM_JOB_ID"
