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
#   pid        1 GPU competition 1: GE2E insurance vectors, then opening + insurance + L6 per pool (A-053)
#   rank-v2    1 GPU competition 2 fallback path: V2 prepare / infer / C2 aggregate (A-030)
#   rank-hsl   8 x 1 GPU HumanSL extraction, 4 shards per question file (A-042)
#   finish     1 GPU (CPU work) HumanSL B2 aggregate (falls back to V2 per question), Day-0 audit, SUMMARY.md
# If an sbatch call times out, check `squeue -u $USER` before rerunning: the job may exist.
set -euo pipefail
RAW="${1:?raw dir}"; OUT="${2:?out dir}"
PROJECT_DIR=/work/jason1122/go_aicup_2026
PY="${PYTHON_BIN:-${HOME}/.conda/envs/go/bin/python}"
cd "$PROJECT_DIR"
if [[ -e "$OUT" ]]; then echo "refusing to reuse existing $OUT" >&2; exit 1; fi
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
       \$INS --out-dir $OUT/pid/\$part --submission $OUT/submissions/pid_\$part.csv; done")

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

DEPS=$(IFS=:; echo "${HSL[*]}")
FINISH=$(submit finish --dependency=afterany:$DEPS:$PID --gres=gpu:1 -c 8 --mem=64G -t 01:00:00 -- \
    "for name in rank_public rank_private; do $PY -m go_aicup.tasks.strength.humansl_predict aggregate --work $OUT/rank/\$name \
       --submission $OUT/submissions/\$name.csv || echo \"HumanSL aggregate failed for \$name\"; \
       cp $OUT/rank/\$name/submission.csv $OUT/submissions/\${name}_v2_fallback.csv || echo \"no V2 submission for \$name\"; done; \
     $PY scripts/day0_audit.py --store-dir $STORE --raw-dir $RAW --identity-dir $OUT/pid --out $OUT/day0_audit.json || echo 'day0 audit failed'; \
     $PY scripts/release_summary.py --out $OUT")

printf 'prep=%s pid=%s rank-v2=%s humansl=%s finish=%s\n' "$PREP" "$PID" "$V2" "${HSL[*]}" "$FINISH" | tee "$OUT/jobs.txt"
