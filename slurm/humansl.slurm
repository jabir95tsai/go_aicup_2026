#!/bin/bash
#SBATCH --job-name=humansl
#SBATCH --partition=dev
#SBATCH --account=GOV115094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=00:20:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/slurm-%x-%j.out

# HumanSL per-move likelihoods (A-039). NUM_SHARDS engine processes share the GPU.
#   RUN=outputs/humansl_pilot NUM_SHARDS=4 LIMIT=10 sbatch slurm/humansl.slurm
set -euo pipefail
PROJECT_DIR=/work/jason1122/go_aicup_2026
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/go/bin/python}"
RUN="${RUN:-outputs/humansl_pilot}"
NUM_SHARDS="${NUM_SHARDS:-1}"
EXTRA=()
[[ -n "${LIMIT:-}" ]] && EXTRA+=(--limit "$LIMIT")
[[ -n "${MAX_SECONDS:-}" ]] && EXTRA+=(--max-seconds "$MAX_SECONDS")
[[ -n "${OVERRIDE:-}" ]] && EXTRA+=(--override-config "$OVERRIDE")
[[ -n "${IN_FLIGHT:-}" ]] && EXTRA+=(--in-flight "$IN_FLIGHT")
[[ -n "${KOMI:-}" ]] && EXTRA+=(--komi "$KOMI")
[[ -n "${CACHE:-}" ]] && EXTRA+=(--cache "$CACHE")
[[ -n "${ALLOW_FAILURES:-}" ]] && EXTRA+=(--allow-failures)
GAMES="${GAMES:-$RUN/games.parquet}"
cd "$PROJECT_DIR"
export PYTHONPATH="${PROJECT_DIR}/src" PYTHONUNBUFFERED=1
# KataGo v1.18.2 cuda12.8/cudnn9.8 build: CUDA libs from the NVIDIA PyPI wheels, libstdc++ from the go env.
LIBS="${PROJECT_DIR}/data/external/cuda-libs-12.8"
export LD_LIBRARY_PATH="${LIBS}/lib:${LIBS}/stdcxx${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
printf 'Job=%s node=%s\n' "${SLURM_JOB_ID:-unknown}" "${SLURMD_NODENAME:-unknown}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
mkdir -p "$RUN"
start=$(date +%s)
pids=()
# SHARD=i runs only shard i (one 1-GPU job per shard); otherwise all shards share this GPU.
for shard in ${SHARD:-$(seq 0 $((NUM_SHARDS - 1)))}; do
    "$PYTHON_BIN" -m "${MODULE:-go_aicup.features.humansl}" --games "$GAMES" --out "$RUN/shard_$shard" \
        --shard "$shard" --num-shards "$NUM_SHARDS" "${EXTRA[@]}" > "$RUN/shard_$shard.log" 2>&1 &
    pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
echo "seconds=$(( $(date +%s) - start )) status=$status"
exit "$status"
