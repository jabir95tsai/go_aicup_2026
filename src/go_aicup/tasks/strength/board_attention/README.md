# Board-conditioned C pooling

HumanSL and C v2 vectors stay frozen. One shared 7,955-parameter network reads
pre-move mover/other stone planes and 13 move features, predicts a scalar,
and softmaxes separately over each game's own and opponent moves. Each side's
weighted ten-rank log likelihood goes directly into CE against the labelled
player's rank; no trainable rank head, flags, H, or temperature is used.

Architecture: three unpadded 5x5 convolutions (2→8→8→8, ReLU), flatten 392→10,
concatenate 13 standardized scalars, 23→16/ReLU→1. Zero final layer starts at
plain-mean C. Scalars are centred rank log p (10), population SD, mean log p,
and ply/100; their moments use all training positions only.

The exact existing `rank_move_weights.split_players` partition is reused:
84,591 train / 36,025 hold / 26,676 U games, HD0. Train batches sample N distinct
players per rank, one game per player; all moves and both sides are retained.
C v2 has seen some train/hold players through other games; U players are unseen.
Packed boards already exist in `data/processed/boards/strength_v1`; post-move
snapshots are shifted inside each game, first board zero, then mover-normalized.
Per-move C v2 likelihoods are joined and audited by game_index, ply and side.
Public/Private caches are not opened by training.

From the repo root on Nano4:

```bash
MODE=benchmark sbatch --time=01:00:00 src/go_aicup/tasks/strength/board_attention/train.slurm
# Read benchmark.json, report actual projected training cost, then:
sbatch src/go_aicup/tasks/strength/board_attention/train.slurm
```

Config: `configs/strength/board_attention.json`. Output:
`outputs/rank/humansl/board_attention/seed2027_hd0_n128/`.
Benchmark compares N=32/64/128, reporting the smallest within 5% of maximum
positions/s, then measures a full hold pass. The user subsequently fixed training
at N=128 regardless of the automatic suggestion; config per_rank is authoritative. Training is AdamW lr=.001,
weight_decay=.0001, max 2,000 updates, hold CE every 100; minimum 500 updates,
patience 5 strict non-improvements. Best includes the uniform step-0 model.
`last.pt` is atomic and resumes optimizer, sampling RNG and monitor history;
`model.pt` is the best hold CE checkpoint. Each monitor records own/opp CE,
effective move count and maximum weight. `features.npz` contains best-model C
for all three partitions. This does not itself claim improved B2 question scores.

Synthetic checks: `PYTHONPATH=src:. python -m pytest -q tests/test_board_attention.py`.
No external packages or new board cache is required; one H200, 12 CPUs, 100 GB
host RAM. Likelihoods and packed boards are resident on GPU. No leaderboard
upload or submission replacement is performed.

After completion, `evaluate.slurm` fits the same prior-matched B2 to mean C
and learned C, with two zero HD1 flags, five seeds (2027–2031). Both arms use
train+hold players for B2, never U. Question generation reuses
`go_aicup.data.cv.make_rank_questions` on U HD0 games: 4,000 same-colour,
rank-balanced questions for each seed 2027/2028, named HD0_Q2027/HD0_Q2028.
These are new HD0 diagnostic sets, not the old E1/E2. Paired player bootstrap
is reported for the first B2 seed; no production promotion or export follows.
The first-stage network remains trained exclusively on train players, selected
on hold CE, and fixed before B2 evaluation reads U labels.

## Joint question-level GD (four independent jobs)

`joint.py` reuses the original HD0 split, frozen C v2 likelihoods, board loader
and attention architecture. Four linear 20→10 B2 heads (one per k) now receive
the question's equal-game mean C. All four arms start at the same zero-score
attention and TRAIN-only sklearn C=1 B2 solution; those four question-level
StandardScalers remain fixed. No fixed temperature or direct side CE is used.

Arms: `mean_gd` (uniform pooling, only B2 updates), `joint` (attention+B2),
`joint_kl001` (joint + .01 KL), `joint_kl01` (joint + .1 KL).
KL is `sum(w*log(w*n))` within each game/side, averaged over sides, then games
within each question, then questions. A batch contains two questions in each
(k, rank) cell: 80 questions / 1,000 game occurrences, all moves. Duplicate
games within a batch are evaluated once, with gradients summed through reuse.

`make_rank_questions` is the sole question generator: fixed 80,000 TRAIN
questions (20,000 per k), 4,000 hold questions, same player/colour within a
question, distinct games, balanced ranks and k. Sampling draws two distinct
questions per cell per update; players may recur between questions. All arms
use identical question seeds, initial weights and sample RNG. TRAIN excludes
hold and U players. Unlike the historical B2 comparison, hold is not added to
B2 fitting. These controlled rank-balanced draws differ from the historical
PriorMatched training prior, so the old 0.725 score is not a direct baseline.

AdamW .001; attention decay .0001. B2 uses explicit `||W||²/(2*N_k)` L2,
matching the mean-CE form of sklearn C=1, with no bias penalty. At most 2,000
updates, official hold Equal-k monitored every 200, strict improvement,
patience 5 after at least 500 updates. Step 0 is eligible. The currently trained
B2 heads are used directly for monitoring: no B2 refit at each checkpoint.
Best is selected separately per arm, never by U. One training seed, 2027.

```bash
for arm in mean_gd joint joint_kl001 joint_kl01; do
  sbatch --export=ALL,ARM="$arm" src/go_aicup/tasks/strength/board_attention/joint.slurm
done
```

Config: `configs/strength/board_attention_joint.json`. Each job requests one
H200 / 12 CPUs / 100GB / 2h. No GPU short test, as explicitly requested.
Historical throughput suggests roughly 10–20 minutes per arm, unmeasured for
joint GD; total allocation cap 8 GPU-h. Jobs are independent and not polled.
Each loads/audits data and fits its identical warm start independently to avoid
shared preparation races. No new full board cache is created.

Output: `outputs/rank/humansl/board_attention_joint/seed2027_hd0/<arm>/`.
`last.pt` resumes optimizer, question RNG, step and patience; `model.pt` is best
hold score, `learning_curve.json` records CE/score and weight concentration.
`initial.pt` preserves the sklearn baseline, scalers and attention start;
`features.npz` stores all best-model C features, aligned with `games.parquet`.
After freezing the best checkpoint, each job automatically scores both U HD0
question draws (2027/2028), writes probabilities/predictions and paired player
bootstrap versus its TRAIN-only sklearn mean baseline in `result.json`, then
publishes `complete.json`. U never changes checkpoints, budgets or hyperparameters.
Across-arm analysis is left to the user; U has been reused historically.

Tests: `tests/test_board_attention_joint.py` covers KL, question/game weighting,
same-colour split isolation, sklearn→Torch equivalence, joint/backbone gradients,
fixed-mean control, exact CPU resume and final U publication/retry. No new
submission files, production model replacement, or leaderboard upload occurs.
