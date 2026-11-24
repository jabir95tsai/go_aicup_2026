#!/bin/bash
# A3 cache (2026-11-23, registered in RESULTS.md): Yugen + OGS + KGS high-dan (u-go.net) players, every KGS player with
# >= 20 human-vs-human games forced in, the rest of 40,000 drawn from OGS (seed 2027), 20 games each.
set -euo pipefail
cd /work/jason1122/go_aicup_2026
export PYTHONPATH=/work/jason1122/go_aicup_2026/src:/work/jason1122/go_aicup_2026 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=12
P=/home/jason1122/.conda/envs/go/bin/python
$P -m go_aicup.tasks.identity.external build --filtered /work/jason1122/ext_ogs_kgs/filtered \
   --out /work/jason1122/boards/yugen_ogs_kgs_A3 --min-games 20 --max-players 40000 --games-per-player 20 \
   --human-opponent --seed 2027 --force-min-id 1000000000000 --workers 12
$P - <<'PY'
import json
import pandas as pd
from go_aicup.data.board_cache import BoardCache
d = "/work/jason1122/boards/yugen_ogs_kgs_A3"
r = json.load(open(f"{d}/external.json")); print("A3 cache:", json.dumps(r))
p = pd.read_parquet(f"{d}/external_players.parquet"); print(p.head(2).to_dict("records"))
print("merged cache opens:", len(BoardCache(d)), "games")
PY
df -h /work/jason1122 | tail -1
