#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=pid2-lik-policy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=01:15:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/pid/likelihood/policy-%A_%a.out
#SBATCH --exclude=25a-hgpn006
# Player-likelihood stage 0: HumanSL top-32 policies, one shard per array task
# (2026-11-20; plan PID_PLAYER_LIKELIHOOD_PLAN.md). usage: sbatch --array=0-3 likelihood_policy.slurm
# (task i runs shards i and i + 4; the 8gpus QOS allows 4 running / 9 submitted jobs per user).
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
P=/home/jason1122/.conda/envs/go/bin/python
df -h /work/jason1122 | tail -1
for S in "${SLURM_ARRAY_TASK_ID:-0}" $(( ${SLURM_ARRAY_TASK_ID:-0} + 4 )); do
  $P -m scripts.identity.pid2_likelihood policy --shard "$S" --num-shards 8
done
