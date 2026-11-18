#!/bin/bash
# Evaluation of the two cross-site short runs (2026-11-18, registered in RESULTS.md before the run).
set -euo pipefail
cd /work/jason1122/go_aicup_2026
export PYTHONPATH=/work/jason1122/go_aicup_2026/src:/work/jason1122/go_aicup_2026 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
P=/home/jason1122/.conda/envs/go/bin/python
C=/work/jason1122/ogs_short_cache
arms=()
for X in Y YO; do
  RD=outputs/pid/ge2e/pid_e2e_ogs_short_$X
  cp "$RD/config.json" "$RD/step960/config.json"
  [ -f "$C/$X/bench/games.npy" ] || $P -m scripts.identity.pid2_seqagg extract --run "pid_e2e_ogs_short_$X/step960" \
     --bench-run pid_ge2e_v4_d4_hd_9600 --parts bench --cache "$C/$X"
  arms+=("$X=$C/$X:$RD/step960/best.pt")
done
$P -m scripts.identity.pid2_ogs_short_eval --arms "${arms[@]}" --out outputs/pid/ogs/short_eval
