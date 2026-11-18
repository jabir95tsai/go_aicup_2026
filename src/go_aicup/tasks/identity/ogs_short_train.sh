#!/bin/bash
# Cross-site short run (2026-11-18): start only if A0 passed (>= 30,000 eligible OGS players).
# Submit with: sbatch --export=ALL,CONFIG=configs/identity/pid_e2e_ogs_short_{Y,YO}.json --time=01:45:00 \
#   --job-name=ogs-short-X --output=outputs/pid/ogs/train-X-%j.out src/go_aicup/tasks/identity/ogs_short_train.sh
set -euo pipefail
cd /work/jason1122/go_aicup_2026
/home/jason1122/.conda/envs/go/bin/python - <<'PY'
import json, sys
r = json.load(open("/home/jason1122/boards/yugen_ogs_v1/external.json"))
print("A0:", json.dumps({k: r[k] for k in ("eligible_players", "selected_players", "external_rows", "bytes")}))
sys.exit(0 if r["eligible_players"] >= 30000 else 3)
PY
exec bash slurm/pid_ge2e.slurm
