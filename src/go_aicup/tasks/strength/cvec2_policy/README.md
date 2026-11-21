# cvec2 policy experiment

Stage 1 reproduces cvec2 on Valid-U and all 800 release predictions. It then fits A
(original H15/C10), B (H9 and normalized C10), and C (normalized H9 and C10).
The normalizer is the per-side mean of per-move logsumexp across profiles, subtracted
from the canonical cvec2 log-likelihoods. The played-move inputs are grouped under
`data/processed/strength_features/humansl/cvec2_policy/`.

| Path | Contents |
| --- | --- |
| `h_hd0/` | H9 for 130,243 games at komi 6.5, including applicable HD≥2 games |
| `h_hd1/` | Original komi-0.5 H shards for 35,979 games. The 15 profiles also supply the canonical baseline |
| `c_v2/` | Original played-move C v2 shards for all 184,698 fit/evaluation games |
| `h_missing.parquet` | 18,476 Valid-U games missing played-move H. Includes 17,432 HD0 and 1,044 HD≥2 games |
| `manifest.json` | Coverage, baseline reproduction and per-file SHA-256 hashes |

Stage 1 reads these shards and any reusable offsets from the selected output directory
before dispatching the remaining games to KataGo. The existing
`stage1/812279/` output has validated offsets for all 18,476 games on the missing
list. A fresh output directory requires those games to be analyzed again.
The previous C v2 and komi-0.5 H
paths resolve to the same shard directories through symbolic links. Public/private
played-move caches for export retain their existing paths.

`data/processed/strength_train_hc/` supplies the original `standard/features.npy`,
`standard/game_index.npy`, the train20 game index and the E1 question table. The old
ZIP and unused historical shards were removed. Older experiments that read those
historical shards or their tables require those inputs to be recovered separately.
`LEGACY_ZIP` remains a fallback for installations using the original layout.

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
Job 815890 completed Stage 1 in the `stage1/812279/` directory. Its suggested anchor
is B. Stage 2 can use this directory directly and generates full-board policies
from the replay shards and teacher checkpoints at runtime. The Stage 1 export
has a separate entry point.

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
locks the CPU artifacts in `$OUT/cpu/`. To reuse a completed plane cache, add
`--reuse-planes <version-directory>` to `prepare` with a fresh `OUT`. Preparation
verifies the cache, its sources and the game order before referencing it in the
new configuration. The eight-card GPU job uses H15 on cards
0–3 and C10 on cards 4–7. C10 cards finish both splits, then help complete
H15 fitting games and write back to the resident H15 shards. After fitting-cache
parity passes, the same C10 cards start training while H15 processes E2. The
weighted arm starts as soon as its CNN mother arm finishes. Extraction waits
for E2 parity. GPU output includes per-side offsets, final checkpoints, game
features, `teacher_overlap.json` and `gpu_complete.json`.
Resubmit with the same `OUT` to resume completed updates and reuse validated features.

CPU scoring fits both normalization controls from the saved offsets and fixed
draws. It writes `question_scores.parquet`, `question_predictions.parquet`,
`result.json` and `complete.json`. Every arm supplies its 52-column fitting
features and a 40-column compatibility projection.

```bash
SMOKE_STEPS=3 sbatch --partition=8gpus --time=00:15:00 \
  src/go_aicup/tasks/strength/cvec2_policy/all.slurm
```

The short run keeps whole optimizer batches and uses the full-pool initial A head.
It exercises the nominal full-cache allocation on every GPU, fits the controls
on smaller draws from the fixed question tables and scores one E2 question per k.
Use the short run for engineering metrics. It has no candidate selection.
The per-arm peer byte count measures payload gathered across GPUs, and
`gpu_samples.csv` records device memory usage.
