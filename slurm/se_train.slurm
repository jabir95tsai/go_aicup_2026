#!/bin/bash
#SBATCH --job-name=se-train-8gpu
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/slurm-%x-%j.out
set -euo pipefail
PROJECT_DIR=/work/jason1122/go_aicup_2026
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/go/bin/python}"
CONFIG="${CONFIG:-configs/strength/se_seed2026.json}"
cd "$PROJECT_DIR"
export PYTHONPATH="${PROJECT_DIR}/src" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NCCL_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1
RUN_DIR=$("$PYTHON_BIN" -c 'import sys; from go_aicup.tasks.strength.reproduce_paper.config import read_config; print(read_config(sys.argv[1])["paths"]["output_dir"])' "$CONFIG")
stopping=0
request_stop() {
    stopping=1
    # The trainer polls this file; signaling torchrun itself can terminate workers immediately.
    touch "$RUN_DIR/.stop-requested"
}
trap request_stop USR1 TERM INT
srun --ntasks=1 "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
    -m go_aicup.tasks.strength.reproduce_paper.train --config "$CONFIG" "$@" &
launcher_pid=$!
status=0
wait "$launcher_pid" || status=$?
if (( stopping )); then
    status=0
    wait "$launcher_pid" || status=$?
fi
exit "$status"
