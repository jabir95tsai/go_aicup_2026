#!/bin/bash
#SBATCH --job-name=hrb2-probe-cache
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
# One HumanSL forward per fixed position (30 per game) for fit_all, fit and M games; chunked and resumable.
srun --ntasks=1 "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
    -m "$MODULE.probe" --config "$CONFIG" --command cache
