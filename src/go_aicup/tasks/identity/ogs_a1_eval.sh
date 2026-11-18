#!/bin/bash
# A1 evaluation (2026-11-19, registered in RESULTS.md before the run): E2E-scratch 9,600 steps trained on
# Yugen only (existing E2E9600, = "Y") vs Yugen+OGS (A1, = "YO"), same fusion and decision as the short run,
# fresh sealed draws 1700- (pool 0 1700-1719, pool 1 1700-1729).
set -euo pipefail
cd /work/jason1122/go_aicup_2026
export PYTHONPATH=/work/jason1122/go_aicup_2026/src:/work/jason1122/go_aicup_2026 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
P=/home/jason1122/.conda/envs/go/bin/python
C=/work/jason1122/ogs_a1_cache
RD=outputs/pid/ge2e/pid_e2e_ogs_A1_YO9600
cp "$RD/config.json" "$RD/step9600/config.json"
[ -f "$C/A1/bench/games.npy" ] || $P -m scripts.identity.pid2_seqagg extract --run "pid_e2e_ogs_A1_YO9600/step9600" \
   --bench-run pid_ge2e_v4_d4_hd_9600 --parts bench --cache "$C/A1"
$P -m scripts.identity.pid2_ogs_short_eval --first 1700 --out outputs/pid/ogs/a1_eval \
   --arms "Y=/home/jason1122/e2e_scratch32_cache/step9600:outputs/pid/ge2e/pid_e2e_scratch32_9600/step9600/best.pt" \
          "YO=$C/A1:$RD/step9600/best.pt"
