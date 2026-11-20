# cvec2 policy experiment

Stage 1 reproduces cvec2 on Valid-U and all 800 release predictions. It then fits A
(original H15/C10), B (H9 and normalized C10), and C (normalized H9 and C10).
The normalizer is the per-side mean of per-move logsumexp across profiles, subtracted
from the canonical cvec2 log-likelihoods. It uses the existing per-move C and HD1 H
shards and recomputes missing H9 offsets with KataGo. The original archive can remain
zipped at `data/processed/strength_train_hc.zip`; only required members are staged
temporarily. An alternate archive is accepted via `LEGACY_ZIP`.

```bash
sbatch src/go_aicup/tasks/strength/cvec2_policy/stage1.slurm
```

The output directory is `outputs/rank/humansl/cvec2_policy/stage1/<job_id>/`. Read
`result.json` for the four k buckets, paired C−B result and suggested anchor.
`complete.json` fingerprints this run's scored questions and 40/52-dimensional features.

Stage 2 requires a manual submission, a completed stage 1 directory and an explicit
choice of B or C. It gets H9/C10 teachers once using joined GPU shards and trains
R10-C, R10-D, CNN-R10-C and CNN-R10-D on identical B2 seed-2027 draws. The per-k
initial B2 is frozen during the 2,500 encoder updates. Each encoder is then frozen
and the original B2 refitted on seeds 2027–2031. All four share the same anchor.

```bash
STAGE1_OUT=outputs/rank/humansl/cvec2_policy/stage1/<job_id> ANCHOR=C \
  sbatch src/go_aicup/tasks/strength/cvec2_policy/stage2.slurm
```

Stage 2 stores model checkpoints, per-game features, paired results, and score rows
under `outputs/rank/humansl/cvec2_policy/stage2/<job_id>/`. Full-board caches are
volatile. An explicit restart with the same `OUT` and unchanged settings rebuilds
the volatile cache and resumes the encoder checkpoints saved every 500 updates.
The two stages report local validation; they do not publish submissions.

## Locked A/H15 eight-arm experiment

[PLAN.md](PLAN.md) defines the split, 52-column A anchor, five learned arms, two
normalization controls, and E2 thresholds. Use the `go` environment with
`PYTHONPATH="$PWD/src:$PWD"` from the repository root.

```bash
MODULE=go_aicup.tasks.strength.cvec2_policy.run
OUT=outputs/rank/humansl/cvec2_policy/all/<run_id>
python -m "$MODULE" all --phase preflight
python -m "$MODULE" all --phase prepare --out "$OUT"
OUT="$OUT" sbatch src/go_aicup/tasks/strength/cvec2_policy/all.slurm
# Run on CPU after the GPU job succeeds.
python -m "$MODULE" all --phase score --out "$OUT"
```

Run full preparation on CPU compute resources. It publishes shared zstd shards
under `data/processed/strength_features/humansl/cvec2_policy/input_planes/` and
locks the CPU artifacts in `$OUT/cpu/`. The eight-card GPU job uses H15 on cards
0–3 and C10 on cards 4–7. The four joint arms train on cards 4–7. The weighted arm
starts as soon as its CNN mother arm finishes. GPU output includes raw teacher
parity, per-side offsets, final checkpoints, game features and `gpu_complete.json`.
Resubmit with the same `OUT` to resume completed updates and reuse validated features.

CPU scoring fits both normalization controls from the saved offsets and fixed
draws. It writes `question_scores.parquet`, `question_predictions.parquet`,
`result.json` and `complete.json`. Every arm supplies its 52-column fitting
features and a 40-column compatibility projection.

```bash
SMOKE_STEPS=3 sbatch --partition=dev --time=00:15:00 \
  src/go_aicup/tasks/strength/cvec2_policy/all.slurm
```

The short run keeps whole optimizer batches and uses the full-pool initial A head.
It exercises the nominal full-cache allocation on every GPU, fits the controls
on smaller draws from the fixed question tables and scores one E2 question per k.
Use the short run for engineering metrics. It has no candidate selection.
The per-arm peer byte count measures payload gathered across GPUs, and
`gpu_samples.csv` records device memory usage.
