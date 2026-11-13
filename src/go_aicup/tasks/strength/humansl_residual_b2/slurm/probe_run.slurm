#!/bin/bash
#SBATCH --job-name=hrb2-probe-run
#SBATCH --partition=dev
#SBATCH --account=GOV115094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/slurm-%x-%j.out
source src/go_aicup/tasks/strength/humansl_residual_b2/slurm/environment.sh
# Four-fold OOF linear/MLP probes, per-game prob/log-prob means, per-k B2 and M report (+gate if H+C exists).
"$PYTHON_BIN" -m "$MODULE.probe" --config "$CONFIG" --command run
