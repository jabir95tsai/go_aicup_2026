#!/bin/bash
#SBATCH --job-name=hrb2-teacher
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/slurm-%x-%j.out
source src/go_aicup/tasks/strength/humansl_residual_b2/slurm/environment.sh
# Submit the eight shards explicitly with: sbatch --array=0-7 .../teacher.slurm
SHARD="${SLURM_ARRAY_TASK_ID:-${SHARD:-0}}"
# Labels are committed as fsynced chunk files; a separate process then re-reads the published arrays,
# so a "complete" status can never come from this process's own cached view.
# --signal=B:USR1 reaches only this shell, so forward it for a clean commit before the time limit.
"$PYTHON_BIN" -m "$MODULE.teacher" --config "$CONFIG" --phase "$PHASE" --shard "$SHARD" "$@" &
LABEL_PID=$!
trap 'kill -USR1 "$LABEL_PID" 2>/dev/null || true' USR1
STATUS=0
while true; do
    wait "$LABEL_PID" && STATUS=0 || STATUS=$?
    # wait returns >128 when interrupted by the trapped signal; keep waiting for the real exit.
    kill -0 "$LABEL_PID" 2>/dev/null || break
done
trap - USR1
if [[ "$STATUS" -ne 0 ]]; then
    echo "teacher label exited with status $STATUS" >&2
    exit "$STATUS"
fi
exec "$PYTHON_BIN" -m "$MODULE.teacher" --config "$CONFIG" --phase "$PHASE" --shard "$SHARD" --verify "$@"
