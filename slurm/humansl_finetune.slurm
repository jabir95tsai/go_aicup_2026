#!/bin/bash
#SBATCH --job-name=c4
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/rank/humansl/slurm-%x-%j.out

# C4 HumanSL fine-tuning (A-057). NPROC must equal the GPUs requested.
#   OUT=outputs/rank/humansl/c4_s1 ARGS="--shards 1 --max-positions 2000000" sbatch slurm/humansl_finetune.slurm
#   NPROC=8 OUT=outputs/rank/humansl/c4_s2 ARGS="--epochs 1 --resume" sbatch --gres=gpu:8 --cpus-per-task=96 --mem=800G slurm/humansl_finetune.slurm
set -euo pipefail
PROJECT_DIR=/work/jason1122/go_aicup_2026
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/go/bin/python}"
NPROC="${NPROC:-1}"
DATA="${DATA:-outputs/rank/humansl/humansl_finetune}"
OUT="${OUT:?OUT required}"
cd "$PROJECT_DIR"
export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}/data/external/katago-src-v1.18.2/KataGo-1.18.2/python" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Settings that worked for the project's Nano4 single-node runs (A-012, A-022).
export NCCL_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1
printf 'Job=%s node=%s nproc=%s\n' "${SLURM_JOB_ID:-unknown}" "${SLURMD_NODENAME:-unknown}" "$NPROC"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
start=$(date +%s)
# shellcheck disable=SC2086
"$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="$NPROC" \
    -m go_aicup.tasks.strength.humansl_finetune --data "$DATA" --out "$OUT" ${ARGS:-}
echo "seconds=$(( $(date +%s) - start ))"
