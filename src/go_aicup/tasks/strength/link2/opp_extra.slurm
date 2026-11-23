#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=rank-opp-extra
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=600G
#SBATCH --time=01:30:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/rank/opp_link2/extra-%j.out
#SBATCH --exclude=25a-hgpn006
# Opponent identity link v2, stage 0 (scripts/strength/rank_opp_link2.py, RESULTS 2026-11-23):
# opponent-side B / A1 / A2 vectors in 4 shards (one per GPU), then single-game opening + style scores against the
# 8,007 training players, for the remaining 42,867 Valid-S games (stage-1 refinement, RESULTS 2026-11-23).
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1
P=/home/jason1122/.conda/envs/go/bin/python
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES}"
N=${#GPUS[@]}
start=$(date +%s)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
pids=()
for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    $P -m scripts.strength.rank_opp_link2 embed --extra --shard "$i" --shards "$N" \
    > "outputs/rank/opp_link2/extra-${SLURM_JOB_ID}-shard${i}.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
tail -n 2 outputs/rank/opp_link2/extra-${SLURM_JOB_ID}-shard*.log
echo "embed status=$status seconds=$(( $(date +%s) - start ))"
[[ $status -eq 0 ]] || exit 1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  $P -m scripts.strength.rank_opp_link2 score --extra --shards "$N" --workers "${WORKERS:-40}"
echo "score seconds=$(( $(date +%s) - start ))"
echo "seconds=$(( $(date +%s) - start ))"
