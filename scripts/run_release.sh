#!/bin/bash
# One command from official CSVs to both competitions' provisional submissions (T3, RESULTS A-054).
# Submits Slurm jobs with dependencies and returns; the last job writes <out>/SUMMARY.md.
#
#   bash scripts/run_release.sh <raw-dir> <out-dir>
#   e.g. bash scripts/run_release.sh data/raw outputs/release_1104
#        bash scripts/run_release.sh outputs/rehearsal/raw outputs/rehearsal_t3
#
# Jobs (all `dev`, account GOV115094):
#   prep       1 GPU (the dev partition requires one; CPU work) preprocess release -> GameStores
#   pid        1 GPU competition 1: GE2E insurance vectors, then opening + insurance per pool (A-053).
#              Default assumes questions and candidates are NOT one-to-one (A-059, user decision
#              2026-10-27): pid_<part>.csv has no L6 assignment. LB-probe variants: pid_<part>_hungarian /
#              _sinkhorn (fused + L6), pid_<part>_opening_only (no insurance, no assignment) and
#              pid_<part>_style_only (insurance alone, no assignment; D-005 controlled probe) and
#              pid_<part>_robust (robust opening eps 0.1 + 32 x insurance, no assignment; A-062 probe)
#   rank-v2    1 GPU competition 2 fallback path: V2 prepare / infer / C2 aggregate (A-030)
#   rank-hsl   8 x 1 GPU HumanSL extraction, 4 shards per question file (A-042)
#   rank-c     8 x 1 GPU C2 rank-vector extraction (A-055), 4 shards per question file
#   finish     1 GPU (CPU work) Day-0 audit, SUMMARY.md and competition-2 B2 aggregates:
#                rank_<part>.csv     AH+C, the primary (A-055); questions without C use AH
#                rank_<part>_ah.csv  AH, the control (A-044); questions without HumanSL use V2
#                rank_<part>_allplayers.csv  AH+C fitted on Valid-S + Valid-U players (A-058), LB probe
#                rank_<part>_balanced.csv  AH+C probabilities, rank-balanced Hungarian decoding (A-064), LB probe
#                rank_<part>_v2_fallback.csv  A-024 + C2
# If an sbatch call times out, check `squeue -u $USER` before rerunning: the job may exist.
set -euo pipefail
RAW="${1:?raw dir}"; OUT="${2:?out dir}"
PROJECT_DIR=/work/jason1122/go_aicup_2026
PY="${PYTHON_BIN:-${HOME}/.conda/envs/go/bin/python}"
cd "$PROJECT_DIR"
if [[ -e "$OUT" ]]; then echo "refusing to reuse existing $OUT" >&2; exit 1; fi
# /work has a 100 GB directory quota; when full, jobs die without any log line (RESULTS A-057).
FREE_GB=$(df -BG --output=avail /work/jason1122 | tail -1 | tr -dc '0-9')
if (( FREE_GB < ${MIN_FREE_GB:-10} )); then
    echo "only ${FREE_GB} GB free under /work/jason1122 (need ${MIN_FREE_GB:-10}); a run writes about 5 GB. Free space first:" >&2
    du -sh "$PROJECT_DIR"/outputs/* 2>/dev/null | sort -h | tail -8 >&2
    exit 1
fi
mkdir -p "$OUT/logs" "$OUT/submissions"
STORE="$OUT/store"
ENV="export PYTHONPATH=$PROJECT_DIR/src:$PROJECT_DIR/scripts PYTHONUNBUFFERED=1"
COMMON=(--parsable -p dev -A GOV115094 --chdir="$PROJECT_DIR")

submit() {   # name, extra sbatch args..., -- command
    local name=$1; shift
    local args=()
    while [[ $1 != "--" ]]; do args+=("$1"); shift; done; shift
    timeout 60 sbatch "${COMMON[@]}" --job-name="rel-$name" --output="$OUT/logs/$name-%j.out" "${args[@]}" \
        --wrap "set -euo pipefail; $ENV; $*"
}

PREP=$(submit prep --gres=gpu:1 -c 12 --mem=100G -t 01:00:00 -- \
    "$PY -m go_aicup.data.preprocess release --raw-dir $RAW --out-dir $STORE")

# Insurance failure falls back to the opening model alone (still a valid submission).
PID=$(submit pid --dependency=afterok:$PREP --gres=gpu:1 -c 12 --mem=150G -t 01:30:00 -- \
    "if $PY -m go_aicup.tasks.identity.insurance --store-dir $STORE --name pid_public pid_private pool_public pool_private --work-dir $OUT/pid/insurance; \
     then INS='--insurance-dir $OUT/pid/insurance'; else echo 'insurance failed: opening only'; INS=''; fi; \
     for part in public private; do $PY -m go_aicup.tasks.identity.predict --store-dir $STORE --pool pool_\$part --questions pid_\$part \
       \$INS --assign none --out-dir $OUT/pid/\$part --submission $OUT/submissions/pid_\$part.csv; done; \
     for part in public private; do for a in hungarian sinkhorn; do $PY -m go_aicup.tasks.identity.predict --store-dir $STORE \
       --pool pool_\$part --questions pid_\$part \$INS --assign \$a --out-dir $OUT/pid/\${part}_\$a \
       --submission $OUT/submissions/pid_\${part}_\$a.csv || echo \"variant \$a failed for \$part\"; done; \
       $PY -m go_aicup.tasks.identity.predict --store-dir $STORE --pool pool_\$part --questions pid_\$part --assign none \
       --out-dir $OUT/pid/\${part}_opening_only --submission $OUT/submissions/pid_\${part}_opening_only.csv \
       || echo \"opening-only failed for \$part\"; \
       if [[ -n \"\$INS\" ]]; then $PY -m go_aicup.tasks.identity.predict --store-dir $STORE --pool pool_\$part \
       --questions pid_\$part \$INS --no-opening --assign none --out-dir $OUT/pid/\${part}_style_only \
       --submission $OUT/submissions/pid_\${part}_style_only.csv || echo \"style-only failed for \$part\"; \
       $PY -m go_aicup.tasks.identity.predict --store-dir $STORE --pool pool_\$part --questions pid_\$part \$INS \
       --robust-eps 0.1 --insurance-lambda 32 --assign none --out-dir $OUT/pid/\${part}_robust \
       --submission $OUT/submissions/pid_\${part}_robust.csv || echo \"robust failed for \$part\"; fi; done")

V2=$(STORE_DIR="$STORE" WORK_DIR="$OUT/rank" timeout 60 sbatch "${COMMON[@]}" --dependency=afterok:$PREP \
    --job-name=rel-rank-v2 --output="$OUT/logs/rank-v2-%j.out" --export=ALL slurm/strength_predict.slurm)

HSL=()
for name in rank_public rank_private; do
    W="$OUT/rank/$name"
    for i in 0 1 2 3; do
        HSL+=($(RUN="$W/humansl" GAMES="$W/cache/games.parquet" CACHE="$W/cache" NUM_SHARDS=4 SHARD=$i ALLOW_FAILURES=1 \
            timeout 60 sbatch "${COMMON[@]}" --dependency=afterok:$V2 --kill-on-invalid-dep=yes --time=01:30:00 \
            --job-name="rel-hsl-${name#rank_}$i" --output="$OUT/logs/hsl-$name-$i-%j.out" --export=ALL slurm/humansl.slurm))
    done
done

RV=()
for name in rank_public rank_private; do
    W="$OUT/rank/$name"
    for i in 0 1 2 3; do
        RV+=($(MODE=extract FEAT="$W/rankvec" GAMES="$W/cache/games.parquet" CACHE="$W/cache" NUM_SHARDS=4 SHARD=$i \
            timeout 60 sbatch "${COMMON[@]}" --dependency=afterok:$V2 --kill-on-invalid-dep=yes --time=01:30:00 \
            --job-name="rel-c-${name#rank_}$i" --output="$OUT/logs/c-$name-$i-%j.out" --export=ALL slurm/humansl_rankvec.slurm))
    done
done

DEPS=$(IFS=:; echo "${HSL[*]}:${RV[*]}")
FINISH=$(submit finish --dependency=afterany:$DEPS:$PID --gres=gpu:1 -c 8 --mem=64G -t 01:00:00 -- \
    "for name in rank_public rank_private; do $PY -m go_aicup.tasks.strength.humansl_predict aggregate --work $OUT/rank/\$name \
       --submission $OUT/submissions/\${name}_ah.csv || echo \"HumanSL aggregate failed for \$name\"; \
       $PY -m go_aicup.tasks.strength.humansl_predict aggregate --work $OUT/rank/\$name \
         --model outputs/humansl_rankvec_model/model.pkl --predictions rankvec_predictions.parquet \
         --fallback humansl_predictions.parquet --submission $OUT/submissions/\$name.csv \
       || { echo \"AH+C aggregate failed for \$name: primary = AH\"; cp $OUT/submissions/\${name}_ah.csv $OUT/submissions/\$name.csv; } \
       || echo \"no AH submission for \$name\"; \
       $PY -m go_aicup.tasks.strength.balanced --predictions $OUT/rank/\$name/rankvec_predictions.parquet \
         --submission $OUT/submissions/\${name}_balanced.csv || echo \"balanced decoding failed for \$name\"; \
       $PY -m go_aicup.tasks.strength.humansl_predict aggregate --work $OUT/rank/\$name \
         --model outputs/humansl_allplayers_model/model.pkl --predictions allplayers_predictions.parquet \
         --fallback humansl_predictions.parquet --submission $OUT/submissions/\${name}_allplayers.csv \
       || echo \"all-player aggregate failed for \$name\"; \
       cp $OUT/rank/\$name/submission.csv $OUT/submissions/\${name}_v2_fallback.csv || echo \"no V2 submission for \$name\"; done; \
     $PY scripts/day0_audit.py --store-dir $STORE --raw-dir $RAW --identity-dir $OUT/pid --out $OUT/day0_audit.json || echo 'day0 audit failed'; \
     $PY scripts/release_summary.py --out $OUT")

printf 'prep=%s pid=%s rank-v2=%s humansl=%s rank-c=%s finish=%s\n' "$PREP" "$PID" "$V2" "${HSL[*]}" "${RV[*]}" "$FINISH" \
    | tee "$OUT/jobs.txt"
