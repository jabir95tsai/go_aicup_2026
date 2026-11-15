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
