#!/bin/bash
#SBATCH --job-name=rank-cvec2-policy-export
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=01:00:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/slurm-%x-%j.out
# CPU-only work (three B2 fits and two test parts); the GPU is only the smallest unit this queue accepts.
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
: "${STAGE1_OUT:?set STAGE1_OUT to a completed stage-1 directory}"
"${HOME}/.conda/envs/go/bin/python" -m go_aicup.tasks.strength.cvec2_policy.export \
  --stage1-out "$STAGE1_OUT" --job "$SLURM_JOB_ID" --legacy-zip "${LEGACY_ZIP:-data/processed/strength_train_hc.zip}"
