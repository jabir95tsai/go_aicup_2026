#!/bin/bash
#SBATCH --job-name=se3-hc-b2-train
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/slurm-%x-%j.out

set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/go/bin/python}"
CONFIG="${CONFIG:-configs/strength/se3_hc_b2.json}"
export PYTHONPATH="$PWD/src" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1

IFS=, read -ra devices <<< "${CUDA_VISIBLE_DEVICES:?Slurm must allocate eight GPUs}"
[[ ${#devices[@]} -eq 8 ]] || { echo 'Expected eight GPUs' >&2; exit 1; }
RUN_DIR=$("$PYTHON_BIN" -c 'import sys; from go_aicup.tasks.strength.SE_H_C_B2.common import read_config; print(read_config(sys.argv[1])["paths"]["output"])' "$CONFIG")
LOG_DIR="$RUN_DIR/slurm_logs/${SLURM_JOB_ID:?Slurm job ID is required}"
mkdir -p "$LOG_DIR"

run_step() {
    printf '\n[%s] %s\n' "$(date -Is)" "$*"
    "$PYTHON_BIN" -m go_aicup.tasks.strength.SE_H_C_B2 --config "$CONFIG" "$@"
}

extract_parallel() {
    local component="$1" scope="$2" task pid status=0
    local -a pids=()
    printf '\n[%s] extract %s %s on eight GPUs\n' "$(date -Is)" "$component" "$scope"
    for task in {0..7}; do
        CUDA_VISIBLE_DEVICES="${devices[$task]}" "$PYTHON_BIN" -m go_aicup.tasks.strength.SE_H_C_B2 --config "$CONFIG" \
            extract --component "$component" --scope "$scope" --shard "$task" --shards 8 \
            > "$LOG_DIR/${component}_${scope}_${task}.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || status=1
    done
    if (( status != 0 )); then
        echo "Feature extraction failed; inspect $LOG_DIR/${component}_${scope}_*.log" >&2
        return 1
    fi
}

echo "job=$SLURM_JOB_ID config=$CONFIG output=$RUN_DIR"
run_step prepare
extract_parallel se3 dev
run_step train --stage dev --arm a
run_step train --stage dev --arm se3
run_step evaluate --stage dev --role select

GATE=$("$PYTHON_BIN" -c 'import json,sys; value=json.load(open(sys.argv[1]))["continue_development"]; assert type(value) is bool; print("continue" if value else "stop")' "$RUN_DIR/reports/dev_select/result.json")
if [[ "$GATE" != continue ]]; then
    echo "Development gate stopped the experiment; see $RUN_DIR/reports/dev_select/result.json"
    exit 0
fi

extract_parallel se3 fit
extract_parallel h newgames
extract_parallel c newgames
run_step train --stage final --arm a
run_step train --stage final --arm se3
run_step evaluate --stage final --role newgames
echo "Training and evaluation complete: $RUN_DIR/reports/final_newgames/result.json"
