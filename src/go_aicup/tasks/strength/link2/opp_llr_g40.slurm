#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=rank-opp-llr40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1000G
#SBATCH --time=03:00:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/rank/opp_llr/llr40-%j.out
#SBATCH --exclude=25a-hgpn006
# LLR rerank with theta on every style-gallery game (scripts/strength/rank_opp_llr.py, RESULTS 2026-11-23).
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
P=/home/jason1122/.conda/envs/go/bin/python
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES}"
N=${#GPUS[@]}
start=$(date +%s)
pids=()
for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} $P -m scripts.strength.rank_opp_llr policy --jobs jobs_g40.parquet --prefix g40_ \
    --shard "$i" --shards "$N" --workers 10 > "outputs/rank/opp_llr/policy40-${SLURM_JOB_ID}-shard${i}.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
tail -n 1 outputs/rank/opp_llr/policy40-${SLURM_JOB_ID}-shard*.log
echo "policy status=$status seconds=$(( $(date +%s) - start ))"
[[ $status -eq 0 ]] || exit 1
CUDA_VISIBLE_DEVICES=${GPUS[0]} OMP_NUM_THREADS=12 $P -m scripts.strength.rank_opp_llr llr --shards "$N" --g40
echo "llr seconds=$(( $(date +%s) - start ))"
OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 $P -m scripts.strength.rank_opp_llr fuse --job "${SLURM_JOB_ID}" \
  --llr llr_g40.npz --alphas 0.35,0.45,0.55,0.65 --tag opp5-l10
echo "seconds=$(( $(date +%s) - start ))"
