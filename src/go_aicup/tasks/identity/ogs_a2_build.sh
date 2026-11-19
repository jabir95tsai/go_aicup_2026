#!/bin/bash
# A2 cache (2026-11-19, registered in RESULTS.md): Yugen + every eligible OGS player (>= 20 ranked games,
# human opponent) x 10 games, from the filtered archive already on /home. CPU work; the GPU is the allocation unit.
set -euo pipefail
cd /work/jason1122/go_aicup_2026
export PYTHONPATH=/work/jason1122/go_aicup_2026/src:/work/jason1122/go_aicup_2026 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=12
P=/home/jason1122/.conda/envs/go/bin/python
$P -m go_aicup.tasks.identity.external build --filtered /home/jason1122/ogs2025/filtered \
   --out /home/jason1122/boards/yugen_ogs_v2 --min-games 20 --games-per-player 10 --human-opponent --seed 2026 --workers 12
$P - <<'PY'
import json
from go_aicup.data.board_cache import BoardCache
r = json.load(open("/home/jason1122/boards/yugen_ogs_v2/external.json")); print("A2 cache:", json.dumps(r))
assert r["selected_players"] > 80000 and r["external_rows"] > 800000, r
print("merged cache opens:", len(BoardCache("/home/jason1122/boards/yugen_ogs_v2")), "games")
PY
df -h /home/jason1122 | tail -1
