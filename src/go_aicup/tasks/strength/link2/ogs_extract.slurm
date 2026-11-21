#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=rank-ogs-h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=800G
#SBATCH --time=02:00:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/rank/ogs/extract-%j.out
#SBATCH --exclude=25a-hgpn006,25a-hgpn007,25a-hgpn069,25a-hgpn083
# HumanSL H on the OGS sample (scripts/strength/rank_ogs.py): one KataGo analysis engine per GPU.
set -euo pipefail
PROJECT_DIR=/work/jason1122/go_aicup_2026
P=/home/jason1122/.conda/envs/go/bin/python
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LIBS="${PROJECT_DIR}/data/external/cuda-libs-12.8"
export LD_LIBRARY_PATH="${LIBS}/lib:${LIBS}/stdcxx${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES}"
N=${#GPUS[@]}
echo "ulimit -u $(ulimit -u)"; data/external/katago-v1.18.2/katago version | head -1
start=$(date +%s)
pids=()
for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} $P -m scripts.strength.rank_ogs extract --shard "$i" --num-shards "$N" \
    > "outputs/rank/ogs/extract-${SLURM_JOB_ID}-shard${i}.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
tail -n 2 outputs/rank/ogs/extract-${SLURM_JOB_ID}-shard*.log
echo "shards=$N status=$status seconds=$(( $(date +%s) - start ))"
exit $status
