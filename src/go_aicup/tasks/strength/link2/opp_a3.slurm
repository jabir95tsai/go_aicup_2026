#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=rank-opp-a3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1000G
#SBATCH --time=02:30:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/rank/opp_a3/run-%j.out
#SBATCH --exclude=25a-hgpn006
# Competition 2: opponent identification with the competition-1 encoder A3 (scripts/strength/rank_opp_a3.py, RESULTS 2026-11-24).
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1
P=/home/jason1122/.conda/envs/go/bin/python
ENC=${ENC:-A3}
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES}"
N=${#GPUS[@]}
start=$(date +%s)
pids=()
for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} OMP_NUM_THREADS=4 $P -m scripts.strength.rank_opp_a3 embed --encoder "$ENC" \
    --shard "$i" --shards "$N" > "outputs/rank/opp_a3/embed-${SLURM_JOB_ID}-shard${i}.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
tail -n 1 outputs/rank/opp_a3/embed-${SLURM_JOB_ID}-shard*.log
echo "embed status=$status seconds=$(( $(date +%s) - start ))"
[[ $status -eq 0 ]] || exit 1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $P -m scripts.strength.rank_opp_a3 score --encoder "$ENC" --shards "$N" --workers 80
$P -m scripts.strength.rank_opp_a3 llr --encoder "$ENC"
OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 $P -m scripts.strength.rank_opp_a3 fuse --encoder "$ENC" --job "${SLURM_JOB_ID}"
echo "seconds=$(( $(date +%s) - start ))"
