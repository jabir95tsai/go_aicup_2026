#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=rank-opp-expand
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --mem=260G
#SBATCH --time=00:35:00
#SBATCH --chdir=/work/jason1122/go_aicup_2026
#SBATCH --output=outputs/rank/opp_llr/expand-score-%j.out

set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=1
python_bin=/home/jason1122/.conda/envs/go/bin/python
if [[ ! -s outputs/rank/opp_llr/llr_g40_expanded.npz ]]; then
  # The second GPU allocation supplies host RAM for assembling the combined policy tensors.
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} "$python_bin" -m scripts.strength.rank_opp_llr llr --g40 --expanded
fi
status=0
for experiment in wide delta; do
  CUDA_VISIBLE_DEVICES= "$python_bin" -m scripts.strength.rank_opp_llr fuse \
    --job "$SLURM_JOB_ID" --experiment "$experiment" || status=1
done
exit "$status"
