#!/bin/bash
# A3L evaluation (2026-11-24, registered in RESULTS.md before the run). STEPS=7200 or 9600; same draws and rule as A3.
set -euo pipefail
cd /work/jason1122/go_aicup_2026
export PYTHONPATH=$PWD/src:$PWD PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
P=/home/jason1122/.conda/envs/go/bin/python
R=pid_e2e_ogsk_A3L_B30_YO${STEPS:?}
RD=outputs/pid/ge2e/$R
C=/work/jason1122/ogs_a3l_cache/$STEPS
mkdir -p "$C"
df -h /work/jason1122 | tail -1
cp "$RD/config.json" "$RD/step$STEPS/config.json"
[ -f "$C/bench/games.npy" ] || $P -m scripts.identity.pid2_seqagg extract --run "$R/step$STEPS" \
   --bench-run pid_ge2e_v4_d4_hd_9600 --parts bench --cache "$C"
$P -m scripts.identity.pid2_ogs_a3_eval --cache "$C" --head "$RD/step$STEPS/best.pt" --out "outputs/pid/ogs/a3l_eval_$STEPS"
rm -rf "/work/jason1122/ogs_a3l_cache/$STEPS"   # this script's own recomputable cache; reached only on success
