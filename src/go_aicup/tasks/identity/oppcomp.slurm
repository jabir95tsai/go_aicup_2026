#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=pid-oppcomp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=600G
#SBATCH --time=01:30:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/pid/oppcomp/oppcomp-%j.out
#SBATCH --exclude=25a-hgpn006
# Competition-1 added set: shared-opponent term on 824896 (scripts/identity/pid2_oppcomp.py, RESULTS 2026-11-23).
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1
P=/home/jason1122/.conda/envs/go/bin/python
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES}"
N=${#GPUS[@]}
start=$(date +%s)
pids=()
for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} OMP_NUM_THREADS=4 $P -m scripts.identity.pid2_oppcomp embed --shard "$i" --shards "$N" \
    > "outputs/pid/oppcomp/embed-${SLURM_JOB_ID}-shard${i}.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
tail -n 1 outputs/pid/oppcomp/embed-${SLURM_JOB_ID}-shard*.log
[[ $status -eq 0 ]] || exit 1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $P -m scripts.identity.pid2_oppcomp score --shards "$N" --workers 40 && $P -m scripts.identity.pid2_oppcomp evaluate
echo "seconds=$(( $(date +%s) - start ))"
