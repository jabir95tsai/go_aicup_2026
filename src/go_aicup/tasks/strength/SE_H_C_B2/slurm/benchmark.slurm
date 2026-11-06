#!/bin/bash
#SBATCH --job-name=se3-hc-benchmark
#SBATCH --partition=dev
#SBATCH --account=GOV115094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/slurm-%x-%j.out
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/go/bin/python}"
CONFIG="${CONFIG:-configs/strength/se3_hc_b2.json}"
export PYTHONPATH="$PWD/src" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
IFS=, read -ra devices <<< "${CUDA_VISIBLE_DEVICES:?Slurm must allocate eight GPUs}"
[[ ${#devices[@]} -eq 8 ]] || { echo 'Expected eight GPUs' >&2; exit 1; }
mkdir -p outputs/se3_hc_b2_benchmark_logs
pids=()
for task in {0..7}; do
    (
        export CUDA_VISIBLE_DEVICES="${devices[$task]}"
        "$PYTHON_BIN" -m go_aicup.tasks.strength.SE_H_C_B2 --config "$CONFIG" extract --component se3 --scope dev --limit 1000 --shard "$task" --shards 8
        for component in h c; do
            "$PYTHON_BIN" -m go_aicup.tasks.strength.SE_H_C_B2 --config "$CONFIG" extract --component "$component" --scope newgames --limit 100 --shard "$task" --shards 8
        done
    ) > "outputs/se3_hc_b2_benchmark_logs/${SLURM_JOB_ID}-$task.out" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
