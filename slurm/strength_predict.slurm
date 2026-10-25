#!/bin/bash
#SBATCH --job-name=v2-predict
#SBATCH --partition=dev
#SBATCH --account=GOV115094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=00:40:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/slurm-%x-%j.out

# Competition-2 test inference (1 GPU). Override with environment variables:
#   STORE_DIR (default data/processed), WORK_DIR (default outputs/rank_submit),
#   NAMES (default "rank_public rank_private"), RUN_DIR (default outputs/v2_strength).
# Example rehearsal:
#   STORE_DIR=outputs/rehearsal/store WORK_DIR=outputs/rehearsal/strength sbatch slurm/strength_predict.slurm
set -euo pipefail
PROJECT_DIR=/work/jason1122/go_aicup_2026
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/go/bin/python}"
STORE_DIR="${STORE_DIR:-data/processed}"
WORK_DIR="${WORK_DIR:-outputs/rank_submit}"
NAMES="${NAMES:-rank_public rank_private}"
RUN_DIR="${RUN_DIR:-outputs/v2_strength}"
cd "$PROJECT_DIR"
export PYTHONPATH="${PROJECT_DIR}/src" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

printf 'Job=%s node=%s\n' "${SLURM_JOB_ID:-unknown}" "${SLURMD_NODENAME:-unknown}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
for name in $NAMES; do
    start=$(date +%s)
    "$PYTHON_BIN" -m go_aicup.tasks.strength.v2.predict --store-dir "$STORE_DIR" --name "$name" \
        --work-dir "$WORK_DIR" --run-dir "$RUN_DIR" --submission "$WORK_DIR/$name/submission.csv"
    echo "$name seconds=$(( $(date +%s) - start ))"
done
