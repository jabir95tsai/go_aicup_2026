#!/bin/bash
# Shared environment for this experiment's Slurm entry points.
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/go/bin/python}"
CONFIG="${CONFIG:-configs/strength/humansl_residual_train.json}"
export PYTHONPATH="$PWD/src" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NCCL_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1
LIBS="$PWD/data/external/cuda-libs-12.8"
if [[ -d "$LIBS" ]]; then
    export LD_LIBRARY_PATH="${LIBS}/lib:${LIBS}/stdcxx${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
MODULE=go_aicup.tasks.strength.humansl_residual_b2
PHASE="${PHASE:-dev}"
FOLD="${FOLD:-all}"
RUN_ROOT=$("$PYTHON_BIN" -c 'import sys; from go_aicup.tasks.strength.humansl_residual_b2.config import read_config; print(read_config(sys.argv[1])["run_dir"])' "$CONFIG")
