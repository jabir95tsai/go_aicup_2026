#!/bin/bash
# A2 evaluation (2026-11-19, registered in RESULTS.md before the run).
set -euo pipefail
cd /work/jason1122/go_aicup_2026
export PYTHONPATH=/work/jason1122/go_aicup_2026/src:/work/jason1122/go_aicup_2026 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
P=/home/jason1122/.conda/envs/go/bin/python
C=/home/jason1122/ogs_runs/a2_cache
RD=outputs/pid/ge2e/pid_e2e_ogs_A2_YO9600
df -h /home/jason1122 /work/jason1122 | tail -2
cp "$RD/config.json" "$RD/step9600/config.json"
[ -f "$C/A2/bench/games.npy" ] || $P -m scripts.identity.pid2_seqagg extract --run "pid_e2e_ogs_A2_YO9600/step9600" \
   --bench-run pid_ge2e_v4_d4_hd_9600 --parts bench --cache "$C/A2"
$P -m scripts.identity.pid2_ogs_a2_eval
