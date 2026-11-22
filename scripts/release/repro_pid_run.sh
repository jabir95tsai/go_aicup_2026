#!/bin/bash
#SBATCH --partition=8gpus
#SBATCH --account=GOV115094
#SBATCH --job-name=pid-repro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --exclude=25a-hgpn006
# Competition 1 reproducibility audit, step 2 (2026-11-22): run the whole chain behind
# 824896-comb-pseudo.csv inside the clean tree made by repro_pid_setup.sh. Commands are the ones the
# original jobs ran (sacct submit lines of 756473, 756474, 772748, 788615, 806692, 815017, 793633,
# 817714, 817975, 820880, 821010, 824896); only the output names differ.
#
#   cd <clean tree> && sbatch --chdir=$PWD --output=$PWD/repro-%j.out scripts/release/repro_pid_run.sh
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=12 NUMBA_NUM_THREADS=12
P=/home/jason1122/.conda/envs/go/bin/python
D=outputs/pid/day0_1112
step() { echo "== $(date '+%F %T') $*"; }
step "tree $(git rev-parse --short HEAD) at $PWD"; df -h . | tail -1

step "1. stores from the raw test CSVs (756473)"
$P -m go_aicup.data.preprocess release --raw-dir data/raw/tests --out-dir $D/store

step "2. original test set: GE2E insurance + opening model + Hungarian (756474)"
$P -m go_aicup.tasks.identity.insurance --store-dir $D/store --name pid_public pid_private pool_public pool_private --work-dir $D/insurance
for part in public private; do
  $P -m go_aicup.tasks.identity.predict --store-dir $D/store --pool pool_$part --questions pid_$part \
     --insurance-dir $D/insurance --assign hungarian --hd-weight 2 --out-dir $D/${part}_hungarian
done

step "3. added set: opening + category + v4 style (772748)"
for part in public private; do
  $P -m go_aicup.tasks.identity.predict2 --store-dir $D/store --pool pool2_$part --questions pid2_$part \
     --out-dir $D/${part}2_d4_9600 --bench-summary outputs/pid/pid2_bench/pid_ge2e_v4_d4_hd_9600/summary.json
done

step "4. seqB style head (788615)"
for part in public private; do
  $P -m scripts.identity.pid2_seqagg export --head outputs/pid/pid2_seqagg/head_B --src $D/${part}2_d4_9600 \
     --out $D/${part}2_d4_9600_seqB --store-dir $D/store --pool pool2_$part --questions pid2_$part
done

step "5. A1 / A2 end-to-end encoders (806692, 815017)"
$P -m scripts.identity.pid2_style_export --run pid_e2e_ogs_A1_YO9600/step9600 \
   --head outputs/pid/ge2e/pid_e2e_ogs_A1_YO9600/step9600/best.pt --tag e2eogs9600
$P -m scripts.identity.pid2_style_export --run pid_e2e_ogs_A2_YO9600/step9600 \
   --head outputs/pid/ge2e/pid_e2e_ogs_A2_YO9600/step9600/best.pt --tag e2eogsA2

step "6. multilevel opening keys L1 / L2 (793633)"
$P -m scripts.identity.pid2_multilevel_export

step "7. HumanSL positions + stage-0 likelihood file (817714)"
$P -m scripts.identity.pid2_likelihood export

step "8. rich per-move features, bench and official (817975)"
$P -m scripts.identity.pid2_likelihood rich-features

step "9. EB prior (820880), learned fusion (821010), pseudo-labels (824896)"
$P -m scripts.identity.pid2_likelihood export --eb
$P -m scripts.identity.pid2_likelihood export --eb --combine --probe
$P -m scripts.identity.pid2_likelihood export --eb --combine --pseudo

step "done"
ls -la submissions/pid/
