#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=rank-delta-policy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=80
#SBATCH --mem=450G
#SBATCH --time=00:30:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/rank/opp_llr/delta-policy-%j.out

set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python_bin=/home/jason1122/.conda/envs/go/bin/python
out=outputs/rank/opp_llr
mkdir -p "$out/delta_policy"
available=$(df -PB1 "$out" | awk 'END {print $4}')
if (( available < 2000000000 )); then
  echo "Insufficient workspace space for the extra policy shards (${available} bytes available)." >&2
  exit 1
fi
"$python_bin" -m scripts.strength.rank_opp_llr plan-delta

IFS=, read -ra gpus <<< "$CUDA_VISIBLE_DEVICES"
shards=${#gpus[@]}
pids=()
for (( shard = 0; shard < shards; shard++ )); do
  CUDA_VISIBLE_DEVICES=${gpus[$shard]} "$python_bin" -m scripts.strength.rank_opp_llr policy \
    --jobs jobs_delta_extra.parquet --prefix delta_policy/shard_ --shard "$shard" --shards "$shards" --workers 10 \
    > "$out/delta-policy-${SLURM_JOB_ID}-shard${shard}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
tail -n 1 "$out"/delta-policy-"${SLURM_JOB_ID}"-shard*.log
exit "$status"
