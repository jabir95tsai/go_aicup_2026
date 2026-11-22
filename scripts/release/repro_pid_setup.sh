#!/bin/bash
# Competition 1 reproducibility audit, step 1 (2026-11-22): build a clean tree that shares only the
# frozen artifacts with the main repo.
#
#   bash scripts/release/repro_pid_setup.sh <main repo> <new tree> <commit>
#
# The new tree is a git worktree at <commit> (so uncommitted code cannot be used). Linked read-only
# from the main repo: training data and training-derived caches, trained weights, validation-bench
# results and the fitted parameters (theta_pop, EB precision, fusion weights, pseudo-label choice,
# multilevel choice). Recomputed from scratch: everything built from the official test files
# (raw CSVs unzipped fresh from data/raw/raw.zip -> stores, board caches, predictions, encoder
# vectors, multilevel keys, HumanSL positions, rich features). Intermediate submission files that
# later steps assert against are copied in as references.
set -euo pipefail
MAIN=$(realpath "$1"); NEW=$2; COMMIT=$3
[ ! -e "$NEW" ] || { echo "$NEW exists"; exit 1; }
git -C "$MAIN" worktree add --detach "$NEW" "$COMMIT"
NEW=$(realpath "$NEW")

link_all_except() {   # link_all_except <src dir> <dst dir> <names to skip...>
  local src=$1 dst=$2; shift 2
  mkdir -p "$dst"
  for p in "$src"/* "$src"/.[!.]*; do
    [ -e "$p" ] || [ -L "$p" ] || continue
    local n; n=$(basename "$p")
    local skip=0; for s in "$@"; do [ "$n" = "$s" ] && skip=1; done
    [ $skip = 1 ] || ln -s "$p" "$dst/$n"
  done
}

# data: raw tests fresh from the archive, everything else linked
mkdir -p "$NEW/data/raw"
ln -s "$MAIN/data/raw/training" "$NEW/data/raw/training"
ln -s "$MAIN/data/external" "$NEW/data/external"
unzip -q "$MAIN/data/raw/raw.zip" 'raw/tests/*' -d "$NEW/data/raw/_unzip"
mv "$NEW/data/raw/_unzip/raw/tests" "$NEW/data/raw/tests"; rm -r "$NEW/data/raw/_unzip"
link_all_except "$MAIN/data/processed" "$NEW/data/processed" identity
link_all_except "$MAIN/data/processed/identity" "$NEW/data/processed/identity" artifacts likelihood
link_all_except "$MAIN/data/processed/identity/artifacts" "$NEW/data/processed/identity/artifacts" day0_1112
mkdir -p "$NEW/data/processed/identity/artifacts/day0_1112"                       # fresh store
link_all_except "$MAIN/data/processed/identity/likelihood" "$NEW/data/processed/identity/likelihood" rich
mkdir -p "$NEW/data/processed/identity/likelihood/rich"                           # fresh bench + official

# outputs: competition-1 official intermediates fresh, the rest linked
link_all_except "$MAIN/outputs" "$NEW/outputs" pid
link_all_except "$MAIN/outputs/pid" "$NEW/outputs/pid" day0_1112 likelihood pid2_multilevel
mkdir -p "$NEW/outputs/pid/day0_1112"
ln -s ../../../data/processed/identity/artifacts/day0_1112/store "$NEW/outputs/pid/day0_1112/store"
link_all_except "$MAIN/outputs/pid/likelihood" "$NEW/outputs/pid/likelihood" official
mkdir -p "$NEW/outputs/pid/likelihood/official"
link_all_except "$MAIN/outputs/pid/pid2_multilevel" "$NEW/outputs/pid/pid2_multilevel" stage1 ml_mix_compare
link_all_except "$MAIN/outputs/pid/pid2_multilevel/stage1" "$NEW/outputs/pid/pid2_multilevel/stage1" export_report.json

# reference submissions (copies) that the exports assert against
mkdir -p "$NEW/submissions/pid"
for f in 772748-hung.csv hung-seqB.csv ogsa12w.csv 820880-likeb.csv 821010-comb-probe.csv; do
  cp "$MAIN/submissions/pid/$f" "$NEW/submissions/pid/$f"
done
echo "tree ready: $NEW ($(git -C "$NEW" rev-parse --short HEAD))"
