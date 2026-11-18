#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=ogs-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=05:00:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/pid/ogs/ogs-build-%j.out
#SBATCH --exclude=25a-hgpn006
# OGS external data (2026-11-18, RESULTS.md A0): filter the downloaded archive, delete it, count, build the
# merged Yugen+OGS board cache (CPU work; the GPU is only the allocation unit).
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=12
P=/home/jason1122/.conda/envs/go/bin/python
A=/home/jason1122/ogs2025/sgfs-by-username.tar.gz
F=/home/jason1122/ogs2025/filtered
[ "$(stat -c %s $A)" = 16191860680 ] || { echo "archive incomplete"; exit 1; }
if [ ! -f $F/owners.npy ]; then
  rm -rf $F
  $P -m scripts.identity.ogs_stream_filter $F < $A
fi
$P - <<'PY'
import json; s = json.load(open("/home/jason1122/ogs2025/filtered/stats.json"))
assert s["final"] and not s["counters"].get("truncated") and s["counters"]["members"] > 100_000_000, s["counters"]
print("filter complete:", json.dumps({k: s[k] for k in ("owners", "owners_ge20_ranked")}), s["counters"])
PY
rm -f $A
for H in "" "--human-opponent"; do
  $P -m go_aicup.tasks.identity.external build --filtered $F --out /dev/null --count-only $H
  $P -m go_aicup.tasks.identity.external build --filtered $F --out /dev/null --count-only --min-year 2021 $H
done
$P -m go_aicup.tasks.identity.external build --filtered $F --out /home/jason1122/boards/yugen_ogs_v1 \
  --min-games 20 --games-per-player 20 --max-players 40000 --human-opponent --seed 2026 --workers 12
$P - <<'PY'
from go_aicup.data.board_cache import BoardCache
c = BoardCache("/home/jason1122/boards/yugen_ogs_v1"); print("merged cache opens:", len(c), "games")
PY
df -h /home/jason1122 | tail -1
