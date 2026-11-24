#!/bin/bash
# A3 evaluation (2026-11-23, registered in RESULTS.md before the run): bench move cache, then A123w / A1_23w vs A12w.
set -euo pipefail
cd /work/jason1122/go_aicup_2026
export PYTHONPATH=/work/jason1122/go_aicup_2026/src:/work/jason1122/go_aicup_2026 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
P=/home/jason1122/.conda/envs/go/bin/python
C=/work/jason1122/ogs_a3_cache
RD=outputs/pid/ge2e/pid_e2e_ogsk_A3_YO7200
df -h /work/jason1122 | tail -1
cp "$RD/config.json" "$RD/step7200/config.json"
[ -f "$C/A3/bench/games.npy" ] || $P -m scripts.identity.pid2_seqagg extract --run "pid_e2e_ogsk_A3_YO7200/step7200" \
   --bench-run pid_ge2e_v4_d4_hd_9600 --parts bench --cache "$C/A3"
$P -m scripts.identity.pid2_ogs_a3_eval
