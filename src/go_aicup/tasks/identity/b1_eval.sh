#!/bin/bash
# B1 evaluation (2026-11-19, registered in RESULTS.md before the run): seed-2027 E2E-scratch 9,600 steps
# added to the current fusion; fresh sealed draws 1700-.
set -euo pipefail
cd /work/jason1122/go_aicup_2026
export PYTHONPATH=/work/jason1122/go_aicup_2026/src:/work/jason1122/go_aicup_2026 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
P=/home/jason1122/.conda/envs/go/bin/python
RD=outputs/pid/ge2e/pid_e2e_scratch32_9600_s2027
cp "$RD/config.json" "$RD/step9600/config.json"
[ -f /home/jason1122/ogs_short_cache/S2027/bench/games.npy ] || $P -m scripts.identity.pid2_seqagg extract \
   --run "pid_e2e_scratch32_9600_s2027/step9600" --bench-run pid_ge2e_v4_d4_hd_9600 --parts bench \
   --cache /home/jason1122/ogs_short_cache/S2027
$P -m scripts.identity.pid2_seed_mix_b1
