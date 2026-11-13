#!/bin/bash
#SBATCH --job-name=hrb2-stage1
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/slurm-%x-%j.out
source src/go_aicup/tasks/strength/humansl_residual_b2/slurm/environment.sh
STAGE=1
source_args=()
mkdir -p "$RUN_ROOT/$PHASE/fold_$FOLD/stage$STAGE"
stop_file="$RUN_ROOT/$PHASE/fold_$FOLD/stage$STAGE/.stop-requested"
stopping=0
request_stop() { stopping=1; touch "$stop_file"; }
trap request_stop USR1 TERM INT
srun --ntasks=1 "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
    -m "$MODULE.train" --config "$CONFIG" --stage "$STAGE" --phase "$PHASE" --fold "$FOLD" "${source_args[@]}" "$@" &
launcher_pid=$!
status=0
wait "$launcher_pid" || status=$?
if (( stopping )); then
    status=0
    wait "$launcher_pid" || status=$?
fi
exit "$status"
