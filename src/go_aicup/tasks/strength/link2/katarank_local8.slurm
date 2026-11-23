#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=rank-kata-local8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=64G
#SBATCH --time=00:23:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/rank/katarank/local-gpu-%j.out
set -euo pipefail
ulimit -c 0
out="$PWD/outputs/rank/katarank/$SLURM_JOB_ID"
export PYTHONPATH="$PWD/src:$PWD:$PWD/outputs/rank/katarank/845543/rank/src"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
p=/home/jason1122/.conda/envs/go/bin/python
"$p" -m scripts.strength.rank_katarank_local plan --out "$out"
scratch=$(mktemp -d "/tmp/katarank-${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
pids=()
for shard in {0..7}; do
    mkdir "$scratch/$shard"
    srun --exclusive --exact --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=12 --mem=8G \
        --output="$out/shard-$shard.out" "$p" -m scripts.strength.rank_katarank_local extract \
        --out "$out" --shard "$shard" --scratch "$scratch/$shard" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
