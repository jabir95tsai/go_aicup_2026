#!/bin/bash
#SBATCH --job-name=hrb2-stage3
#SBATCH --partition=dev
#SBATCH --account=GOV115094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/slurm-%x-%j.out
source src/go_aicup/tasks/strength/humansl_residual_b2/slurm/environment.sh
STAGE="${STAGE:-1}"
if [[ "$PHASE" == release ]]; then
    # After the U lock: refit the locked arm on dev OOF (fit pool) + fold-averaged M+U questions.
    : "${ARM:?Set ARM=new|stack}" "${AGGREGATION:?Set AGGREGATION=prob|logp}" "${CANDIDATE:?Set the locked candidate}"
    "$PYTHON_BIN" -m "$MODULE.b2" --config "$CONFIG" --stage "$STAGE" --phase release \
        --candidate "$CANDIDATE" --arm "$ARM" --aggregation "$AGGREGATION"
else
    # Both aggregations for the new arm; the config aggregation (logp) drives best/final selection.
    for candidate in best final; do
        for aggregation in prob logp; do
            "$PYTHON_BIN" -m "$MODULE.b2" --config "$CONFIG" --stage "$STAGE" --candidate "$candidate" \
                --arm new --aggregation "$aggregation"
        done
    done
    "$PYTHON_BIN" -m "$MODULE.evaluate" --config "$CONFIG" --stage "$STAGE" --select
fi
