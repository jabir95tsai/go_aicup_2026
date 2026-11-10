#!/bin/bash
#SBATCH --job-name=hrb2-teacher
#SBATCH --partition=8gpus
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
# Submit the eight shards explicitly with: sbatch --array=0-7 .../teacher.slurm
SHARD="${SLURM_ARRAY_TASK_ID:-${SHARD:-0}}"
exec "$PYTHON_BIN" -m "$MODULE.teacher" --config "$CONFIG" --phase "$PHASE" --shard "$SHARD" "$@"
