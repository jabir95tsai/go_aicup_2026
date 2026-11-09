#!/bin/bash
#SBATCH --job-name=hrb2-smoke100
#SBATCH --partition=dev
#SBATCH --account=GOV115094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/slurm-%x-%j.out
CONFIG="${CONFIG:-configs/strength/humansl_residual_smoke100.json}"
source src/go_aicup/tasks/strength/humansl_residual_b2/slurm/environment.sh
RUN_ROOT=$("$PYTHON_BIN" -c 'import sys; from go_aicup.tasks.strength.humansl_residual_b2.config import read_config; print(read_config(sys.argv[1])["run_dir"])' "$CONFIG")
PHASE=dev
STAGE=1
stop_file="$RUN_ROOT/$PHASE/fold_$FOLD/stage1/.stop-requested"
request_stop() { mkdir -p "$(dirname "$stop_file")"; touch "$stop_file"; }
trap request_stop USR1 TERM INT
for STAGE in 1 2; do
    stop_file="$RUN_ROOT/$PHASE/fold_$FOLD/stage$STAGE/.stop-requested"
    source_args=()
    if [[ "$STAGE" == 2 ]]; then
        source_args=(--source "$RUN_ROOT/$PHASE/fold_$FOLD/stage1/final.pt")
    fi
    srun --ntasks=1 "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
        -m "$MODULE.train" --config "$CONFIG" --stage "$STAGE" --phase "$PHASE" --fold "$FOLD" "${source_args[@]}" &
    launcher_pid=$!
    status=0
    wait "$launcher_pid" || status=$?
    if (( status )); then wait "$launcher_pid"; fi
    "$PYTHON_BIN" -c 'import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s["status"]=="completed" else 1)' \
        "$RUN_ROOT/$PHASE/fold_$FOLD/stage$STAGE/summary.json"
done
