#!/bin/bash
#SBATCH --job-name=rank-cvec2-policy-s1
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=200G
#SBATCH --time=05:00:00
#SBATCH --exclude=25a-hgpn007,25a-hgpn069,25a-hgpn083
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/slurm-%x-%j.out
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LIBS="$PWD/data/external/cuda-libs-12.8"
export LD_LIBRARY_PATH="${LIBS}/lib:${LIBS}/stdcxx${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
data/external/katago-v1.18.2/katago version
OUT="${OUT:-outputs/rank/humansl/cvec2_policy/stage1/$SLURM_JOB_ID}"
LEGACY_ZIP="${LEGACY_ZIP:-data/processed/strength_train_hc.zip}"
"${HOME}/.conda/envs/go/bin/python" -m go_aicup.tasks.strength.cvec2_policy.run stage1 \
  --out "$OUT" --legacy-zip "$LEGACY_ZIP" --workers 8
