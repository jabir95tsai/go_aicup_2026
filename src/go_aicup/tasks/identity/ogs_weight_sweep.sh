#!/bin/bash
# Added-set style weight sweep after A1 (2026-11-19, registered in RESULTS.md before the run).
set -euo pipefail
cd /work/jason1122/go_aicup_2026
export PYTHONPATH=/work/jason1122/go_aicup_2026/src:/work/jason1122/go_aicup_2026 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
/home/jason1122/.conda/envs/go/bin/python -m scripts.identity.pid2_ogs_weight_sweep
