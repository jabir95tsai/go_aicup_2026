#!/bin/bash
#SBATCH --job-name=se3-hc-predict
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/slurm-%x-%j.out
# BASE_WORK must contain complete existing AH+C rank_public/private work dirs.
set -euo pipefail
: "${BASE_WORK:?Set BASE_WORK to the existing AH+C rank work root}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/go/bin/python}"
CONFIG="${CONFIG:-configs/strength/se3_hc_b2.json}"
SAMPLE="${SAMPLE:-data/raw/tests/sample_submission_rank.csv}"
SUBMISSION="${SUBMISSION:-submissions/rank/submission_rank_se3_hc_b2_${SLURM_JOB_ID}.csv}"
export PYTHONPATH="$PWD/src" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
IFS=, read -ra devices <<< "${CUDA_VISIBLE_DEVICES:?Slurm must allocate eight GPUs}"
[[ ${#devices[@]} -eq 8 ]] || { echo 'Expected eight GPUs' >&2; exit 1; }
mkdir -p outputs/se3_hc_b2_predict_logs
pids=()
for task in {0..7}; do
    (
        export CUDA_VISIBLE_DEVICES="${devices[$task]}"
        part=public
        (( task < 4 )) || part=private
        work="$BASE_WORK/rank_$part"
        for component in se3 h c; do
            "$PYTHON_BIN" -m go_aicup.tasks.strength.SE_H_C_B2 --config "$CONFIG" extract \
                --component "$component" --work "$work" --shard "$((task % 4))" --shards 4
        done
    ) > "outputs/se3_hc_b2_predict_logs/${SLURM_JOB_ID}-$task.out" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || exit "$status"
for part in public private; do
    "$PYTHON_BIN" -m go_aicup.tasks.strength.SE_H_C_B2 --config "$CONFIG" aggregate \
        --work "$BASE_WORK/rank_$part" --fallback "$BASE_WORK/rank_$part/rankvec_predictions.parquet"
done
"$PYTHON_BIN" -m go_aicup.tasks.strength.SE_H_C_B2 --config "$CONFIG" export \
    --work-root "$BASE_WORK" --sample "$SAMPLE" --output "$SUBMISSION"
