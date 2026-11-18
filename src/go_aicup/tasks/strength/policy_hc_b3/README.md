# Frozen policy H/C + per-k B3

User-approved 2026-11-18 experiment. Two GPUs: GPU0 extracts H features and GPU1 extracts C
features from the original TRAIN/monitor policy ZIPs. Each gathers a policy chunk once and
runs the two frozen monitor-selected CNNs, accumulating own/opponent per-game means in FP32.
No teacher is instantiated, no raw SGF is replayed, and no policy ZIP is extracted to disk.
GPU0 trains B3 after both extraction workers exit; GPU1 is idle during B3 fitting.

| Arm | Feature order | Width |
|---|---|---:|
| A0 | original q_h, original q_c | 38 |
| R-HC | CNN-R10_h (q+delta), original q_h, CNN-R10_c (q+delta), original q_c | 76 |
| Z-HC | CNN-Z32_h z only (own32, opp32), original q_h, CNN-Z32_c z only, original q_c | 166 |

H uses nine profiles per side; C uses ten rank hypotheses per side. The z features are the
32-dimensional output per side, not the hidden64 layer or the combined [q,z] output.
Each game has equal weight in a question, regardless of game length. No D4 augmentation
is applied to frozen feature extraction, matching upstream evaluation.

Source runs: H 799148, C 796606. Fixed dev features: H 804562, C 796930. These are checked
against the upstream run identity and selected checkpoint steps; feature/checkpoint hashes,
ZIP marker hashes, exact H/C selection equality and question membership are recorded/checked.
No dev policy or teacher inference is needed. The preflight hashes dev artifacts but does not
use their feature values or labels for training or selection.

Each arm has four independent Linear(width,10) heads for k=5/10/15/20. Scalers use only
that k's TRAIN questions. All arms start from the same TRAIN-only sklearn H+C LR (C=1,
max_iter=5000); extra learned-feature columns have zero initial head weights. All feature
columns are standardized with fixed TRAIN statistics. The frozen CNNs are never optimized.

Per head: 2,000 TRAIN questions, batch8, seed2027, 250 updates/epoch, 12 epochs/3,000 updates,
AdamW LR .0003 with default betas/eps, weight_decay=0. CE plus weight squared sum/(2*2000),
no bias L2. Step0 and every125 steps evaluate all corresponding TRAIN/monitor questions;
strict monitor official-score improvement selects the best, ties keep the earlier step.
No early stopping. Latest full optimizer state and boundary best/history are saved atomically
at each epoch. Interrupted epochs are replayed deterministically; there is no stochastic layer.

After all 12 fits, each selected head is evaluated on fixed dev. Softmax + score_rank decoding;
report four k scores, Equal-k, and player-paired bootstrap2,000 for R-HC-A0, Z-HC-A0, Z-HC-R-HC.
Candidate gate: delta Equal-k>=.005, every k>=-.005, bootstrap positive fraction>=.90.
One upstream seed and historically reused dev limit conclusions; upstream CNNs used this
same monitor, and upstream C-vector training is not certified player-disjoint here.

CPU read-only preflight (no output files, no GPU allocation):

```bash
PYTHONPATH=src:. ~/.conda/envs/go/bin/python -m go_aicup.tasks.strength.policy_hc_b3.train --prepare-only
```

Submit only after separately authorized:

```bash
sbatch src/go_aicup/tasks/strength/policy_hc_b3/train.slurm
```

Slurm requests two H200s, 24 CPU, 400G RAM, 1h (at most 2 allocated GPU-h). Historical per-side
full evaluation is ~5.5 seconds/model, plus ZIP loading (~50 seconds for H), aggregation,
CPU sklearn initialization, B3 fitting and reports. Combined runtime has not been measured;
allow tens of minutes, then record actual wall time and allocated GPU-h. The second GPU
remains allocated during B3 fitting. CPU tests do not validate CUDA peak memory or throughput.

Output: outputs/rank/humansl/policy_hc_b3/seed2027_b8_e12/<job_id>/.
Feature workers save small per-game q/r/z NPZ files and checksums under features/{H,C}/;
completed matching features are reused on restart without re-reading full policy ZIPs.
B3 heads, scalers, selected step, latest optimizer/history and predictions are under
{A0,R-HC,Z-HC}/k{k}/ and each arm's prediction tables. summary.json and complete.json
are published only after all fits, dev scoring, comparisons and source revalidation succeed.
Use --out <original-run> to resume; input/config/code identity changes are rejected.
No submission CSV or leaderboard upload is produced.

## Two-GPU smoke50

`configs/strength/policy_hc_b3_smoke.json` uses the same full TRAIN/monitor policy extraction,
scalers, initialization, batches and LR, but caps each of the twelve heads at 50 updates.
It evaluates step0/25/50, never reads dev feature values, and produces no dev predictions,
bootstrap or candidate gate. The output is marked smoke=true. Full training defaults are unchanged.

```bash
sbatch --job-name=rank-policy-hc-b3-smoke --time=00:30:00 --export=ALL,CONFIG=configs/strength/policy_hc_b3_smoke.json src/go_aicup/tasks/strength/policy_hc_b3/train.slurm
```

This requests at most 1 allocated GPU-h. Worker markers record ZIP/cache loading time,
CNN feature time and peak GPU allocation; summary records feature wall time and B3 wall time.
