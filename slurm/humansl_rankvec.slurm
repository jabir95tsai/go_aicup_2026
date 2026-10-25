#!/bin/bash
#SBATCH --job-name=rankvec
#SBATCH --partition=dev
#SBATCH --account=GOV115094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=01:00:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/slurm-%x-%j.out

# C2 HumanSL rank vectors (A-055), one GPU per job.
#   MODE=check                                        sbatch slurm/humansl_rankvec.slurm
#   MODE=pack    SHARD=0 NUM_SHARDS=4                 sbatch slurm/humansl_rankvec.slurm
#   MODE=train                                        sbatch --time=03:00:00 slurm/humansl_rankvec.slurm
#   MODE=extract SHARD=0 NUM_SHARDS=10 [LIMIT=N]      sbatch slurm/humansl_rankvec.slurm
set -euo pipefail
PROJECT_DIR=/work/jason1122/go_aicup_2026
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/go/bin/python}"
RUN="${RUN:-outputs/humansl_rankvec}"
FEAT="${FEAT:-outputs/humansl_rankvec_feat}"
cd "$PROJECT_DIR"
export PYTHONPATH="${PROJECT_DIR}/src:${PROJECT_DIR}/scripts:${PROJECT_DIR}/data/external/katago-src-v1.18.2/KataGo-1.18.2/python" PYTHONUNBUFFERED=1
printf 'Job=%s node=%s mode=%s\n' "${SLURM_JOB_ID:-unknown}" "${SLURMD_NODENAME:-unknown}" "$MODE"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
EXTRA=()
[[ -n "${LIMIT:-}" ]] && EXTRA+=(--limit "$LIMIT")
[[ -n "${WORKERS:-}" ]] && EXTRA+=(--workers "$WORKERS")
start=$(date +%s)
case "$MODE" in
    check)   "$PYTHON_BIN" scripts/humansl_rankvec.py check ;;
    pack)    "$PYTHON_BIN" -m go_aicup.features.humansl_rankvec pack --games "$RUN/games.parquet" \
                 --out "$RUN/pack_$SHARD" --shard "$SHARD" --num-shards "$NUM_SHARDS" "${EXTRA[@]}" ;;
    train)   "$PYTHON_BIN" -m go_aicup.features.humansl_rankvec train --run "$RUN" "${EXTRA[@]}" ;;
    extract) mkdir -p "$FEAT"
             "$PYTHON_BIN" -m go_aicup.features.humansl_rankvec extract --games "${GAMES:-outputs/emb_stage1/games.parquet}" \
                 --vectors "$RUN/vectors.pt" --out "$FEAT/shard_$SHARD" --shard "$SHARD" --num-shards "$NUM_SHARDS" \
                 "${EXTRA[@]}" ;;
    *) echo "unknown MODE=$MODE"; exit 2 ;;
esac
echo "seconds=$(( $(date +%s) - start ))"
