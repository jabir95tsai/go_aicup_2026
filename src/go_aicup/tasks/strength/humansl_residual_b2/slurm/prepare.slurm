#!/bin/bash
#SBATCH --job-name=hrb2-prepare
#SBATCH --partition=dev
#SBATCH --account=GOV115094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/slurm-%x-%j.out
source src/go_aicup/tasks/strength/humansl_residual_b2/slurm/environment.sh
# EXTENSION=1 builds the additive capped pools (fit_all); PHASE=release adds the M+U release pools
# and is only allowed after the U lock. Without EXTENSION the base prepare runs (idempotent).
args=()
if [[ "${EXTENSION:-0}" == 1 || "$PHASE" == release ]]; then args=(--extension); fi
if [[ "$PHASE" == release ]]; then args+=(--release); fi
"$PYTHON_BIN" -m "$MODULE.prepare" --config "$CONFIG" "${args[@]}"
