#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=rank-link2-embed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=600G
#SBATCH --time=02:00:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/rank/link2/embed-%j.out
#SBATCH --exclude=25a-hgpn006
# L10 v2 (scripts/strength/rank_link2.py, RESULTS 2026-11-21): engineering check on one GPU, then the
# B / A1 / A2 game vectors of every needed game in 4 shards, one per GPU. `plan` must have run.
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
P=/home/jason1122/.conda/envs/go/bin/python
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES}"
N=${#GPUS[@]}
start=$(date +%s)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[[ -f outputs/rank/link2/selfcheck.json ]] || CUDA_VISIBLE_DEVICES=${GPUS[0]} $P -m scripts.strength.rank_link2 selfcheck
echo "selfcheck done $(( $(date +%s) - start ))s"
pids=()
for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} $P -m scripts.strength.rank_link2 embed --shard "$i" --shards "$N" \
    > "outputs/rank/link2/embed-${SLURM_JOB_ID}-shard${i}.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
tail -n 3 outputs/rank/link2/embed-${SLURM_JOB_ID}-shard*.log
echo "shards=$N status=$status seconds=$(( $(date +%s) - start ))"
exit $status
