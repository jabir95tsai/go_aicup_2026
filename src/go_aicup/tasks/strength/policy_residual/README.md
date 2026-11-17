# Frozen-policy C-only residual experiment

User-approved five arms: A0 (raw C), S (seven per-rank distribution statistics),
F (played-token embedding plus attention over legal candidates), F-mask (same
network, other candidates' probability fields hidden), C-conv (30-channel
P/logP/played-marker policy image). HumanSL and C v2 vectors remain frozen.
Each residual produces `e_t = q_t + delta_t`, ten values per move. The final
delta layer is zero-initialised. Own/opponent moves are averaged separately,
then games equally within each question. Each arm independently trains one
shared-across-k `Linear(20,10)` head, CE plus sklearn-compatible head L2.
No H, HD flags, production prior matching, or leaderboard export is included.

`data/cv.py:make_policy_residual_games` selects seed-2027 A/B-source players:
50 training and 10 monitoring players per rank and half. Each selected player
has twenty HD<=1 games in their most frequent colour (tie: black), reselected
from canonical metadata. This intentionally differs from C4x's mixed-colour
twenty-game sample. Existing 762 dev player/game identities stay fixed, but
all moves are now encoded, rather than only the original 50 moves per side.
Train/monitor/dev have 20,000/4,000/12,974 games and
4,460,508/898,462/2,896,761 positions. All positions of every sampled game enter
its mean. Duplicate games across questions in a batch are evaluated once.

All five GPUs jointly extract policies, each using ten CPU encoder workers.
The checkpoint and vectors are frozen/eval; HD0 komi=6.5, HD1 komi=0.5. The
existing C legal mask (empty, not ko-banned, plus pass) is preserved; it does
not additionally exclude suicide/superko. Policy float16, exact played q
float32, targets and side flags live only in RAM-backed shared-memory tmpfs.
The approximately 56 GiB shared cache is not a persistent disk artifact.
Compilation caches also use a temporary RAM directory. Cache completion,
finite legal probabilities, probability mass and cached-vs-original mean C
(max 0.002) are checked before any arm trains. No old cache is deleted.

After extraction, GPU 0/1/2/3/4 train A0/S/F/F-mask/C-conv concurrently.
Each gets the same TRAIN-only sklearn warm-start, fixed TRAIN-only head
scaler, questions, question RNG, optimizer rules and budget. S fits its input
moments on TRAIN positions only. F and F-mask have identical initial weights;
their played token uses the same exact q, preventing FP16 renormalisation from
leaking hidden probabilities. CNN D4 augmentation moves P/logP/marker together;
pass stays separate. Set-based arms are invariant to candidate order.

Bounded-memory backprop first computes game features without gradients, then
backpropagates the question loss to those features, then recomputes chunks
with their exact upstream gradients. No batch norm or dropout is present in
these encoders. CPU tests compare parameter gradients against direct autograd.

Budget: one seed, up to 200 updates, eight questions/update (two per k),
AdamW .001, residual decay .0001. This is about 20,000 game occurrences, not a
guarantee of visiting each training game. Monitoring every 50 updates; best
strict Equal-k improvement, step zero eligible, patience three after 100.
Only monitoring selects checkpoints. Dev is evaluated after each arm freezes
its selected checkpoint; there is no dev-based budget/model selection.
Training game/player coverage is reported. The fixed outer dev has been used
historically, and pre-existing C-vector learning-player overlap is reported.

Submit once from repository root:

```bash
sbatch src/go_aicup/tasks/strength/policy_residual/train.slurm
```

Single node, 5 H200 GPUs, 60 CPUs, 500 GiB RAM, four-hour allocation cap
(20 allocated GPU-hours maximum). Historical forward-only estimate for all
three roles: 2.2-4.4 GPU-hours, excluding encoding/training; real throughput
is unmeasured. No separate GPU smoke job. Per user request, submission is not
followed by polling. The job performs numeric gates and reports failures itself.

Output: `outputs/rank/humansl/policy_residual/seed2027_t1000_m200/<job_id>/`.
Only config, source identities, selected games/questions, raw per-game C,
small model/optimizer/RNG checkpoints, curves and final predictions/features
are persistent. `memory_plan.json`, extraction reports and final
`result.json.resources` record RAM and actual output bytes; expected disk
use below 100 MiB. Full policies and encoded position inputs are never saved.
Successful output has root `complete.json`; partial output must not be treated
as a completed experiment. Checkpoint replay within an arm is supported if
the exact source identity is retained; loss of the RAM cache requires a fresh
extraction. The default job-id output directory starts a separate run.

Main comparisons: S-A0, F-S, F-F-mask; also F/A0, F-mask/A0 and CNN comparisons.
Paired player bootstrap (2,000 draws) recomputes Equal-k per draw. Gate:
Equal-k >= .005, bootstrap positive fraction >= .90, every k >= -.005.
Gain unknown, expected small. This C-only pilot does not establish gains on H+C.

CPU checks:

```bash
PYTHONPATH=src:. python -m pytest -q tests/test_policy_residual.py
```

Frozen-checkpoint training diagnostic (no optimizer updates):

```bash
sbatch src/go_aicup/tasks/strength/policy_residual/evaluate_train.slurm
```

This reuses the original `games.parquet`, 8,000 `train_questions.parquet`, raw C,
source fingerprints, and A0/S/F best/latest checkpoints. Only TRAIN policies
are re-extracted into shared RAM; the original raw C baseline is retained after
checking re-extraction error <= .002. Outputs are isolated under
`<source-run>/train_diagnostic/<job_id>/`: full fixed-TRAIN CE, head L2 and
CE+L2, official scores per k, per-question probabilities, and per-game features
for steps 100/200. Monitoring metrics come from the original saved curves.
These are in-sample diagnostics, not new validation evidence. The original
checkpoints and experiment reports are read-only. Estimated runtime from the
original extraction is 17–22 minutes on five H200s (1.4–1.8 allocated GPU-h);
the diagnostic Slurm cap is one hour (five allocated GPU-h).

After successful extraction/evaluation, rebuild CPU-only sampling diagnostics:

```bash
PYTHONPATH=src:. python -m go_aicup.tasks.strength.policy_residual.summarize_train \
  --diagnostic outputs/rank/humansl/policy_residual/seed2027_t1000_m200/785715/train_diagnostic/786101
```

`sampling_analysis.json` compares the initial shared A0 head and both frozen
checkpoints on all fixed TRAIN questions, the fixed 755 questions sampled
through step 100, and the fixed 1,429 sampled through step 200. The original
sklearn warm-start already used all 8,000 training labels; sampled subsets
refer only to the subsequent AdamW updates. Their exact replay is checked
against saved RNG and game-coverage states. This is a CPU report operation;
it does not fit a new head or change any checkpoint.

Six-arm complete-question-epoch follow-up (2026-11-16 user approved):

```bash
sbatch src/go_aicup/tasks/strength/policy_residual/epoch.slurm
```

`epoch.py` reuses job 785715's original games, TRAIN/monitor/dev questions,
raw C, and initial sklearn head. It does not read `outputs/rank/humansl/permove/`.
The complete policy is re-extracted once into shared RAM for all six cases:
A0/S/F x LR .001/.0003. Five GPUs extract together; four then train the four
S/F cases concurrently, while the fifth runs the two inexpensive A0 cases
sequentially. This is the same residual architecture, not plan.md's seven arms.

Each epoch independently shuffles the 2,000 questions in each k bucket and
consumes them without replacement in 250 batches of 32 (eight per k), with
an exact-coverage check. All cases use identical epoch orders and D4 choices.
There is one optimizer update per full 32-question batch, with bounded-memory
chunk recomputation and deduplicated games; no gradient accumulation or clipping.
Four full epochs give 1,000 updates/32,000 question occurrences per case.
Only complete epochs evaluate full fixed TRAIN and monitor; step zero is a
reference and eligible checkpoint, and there is no early stopping in this
four-epoch comparison. Strict monitor Equal-k improvement saves the best model.
Dev is evaluated only after all four epochs in that case, never for selection.

Every update writes head/phi/scorer/delta gradient norms, parameter norms,
actual update norms/ratios, missing-gradient counts and nonfinite counts to
`gradients_epoch_N.jsonl`. Nonfinite parameters/gradients/updates fail the arm.
No gradient tensors or complete policies are written to persistent disk.
Each completed epoch atomically saves optimizer/model/head/history/coverage
in last.pt; resuming restores that epoch and recomputes a partial next epoch
with the same deterministic question/D4 order. Loss of the RAM cache requires
re-extraction. Initial/best checkpoints and all original experiment files remain.

Outputs: `outputs/rank/humansl/policy_residual_epoch/seed2027_b32_e4/<job_id>/`.
Each case retains learning_curve.json (TRAIN/monitor CE, CE+head L2 and k scores),
gradient summaries, best/latest weights, selected per-game features and
per-question probabilities for TRAIN/monitor/dev, and result/complete markers.
Root result.json includes paired-player bootstrap comparisons within each LR
and between the two LRs. The prior numerical gate is unchanged: delta Equal-k
>= .005, bootstrap positive fraction >= .90, every k >= -.005; this one-seed
historically reused dev remains limited evidence and cannot tune the budget.

Estimated cost from prior extraction/update timings: 50–65 minutes on five
H200s (4.2–5.4 allocated GPU-h); Slurm cap 90 minutes/7.5 GPU-h, 60 CPUs,
500 GiB RAM. Shared policy cache about 56 GiB, persistent output budget 200 MB.
32-question throughput and gradient-stat overhead are not yet GPU-measured.
Per user request, submission is not followed by queue/execution monitoring.


Four-arm branch experiment (2026-11-17, root `plan.md`):

```bash
sbatch src/go_aicup/tasks/strength/policy_residual/branches.slurm
```

Independent `branches.py` and `configs/strength/policy_branches.json` reuse the
original immutable 785715 data and helpers. A0/R10/Z10/Z32 have 20/20/40/84
question features; q uses the original scaler and z stays unscaled. All heads
start with the same sklearn predictions, R10 output starts at zero, and Z
outputs start random with zero head readout. Shared phi/scorer/hidden weights
are identical. Every arm runs 3,000 AdamW updates at .0003, 32 balanced questions
per update, 12 complete epochs. Full TRAIN/monitor evaluation occurs at step
zero and every 125 updates. Strict monitor Equal-k alone selects the model;
dev is evaluated only after the fixed budget. No automatic follow-up seeds.

Five H200s extract into shared RAM, then four GPUs each train one arm. Allocation:
60 CPUs, 500 GiB RAM, two hours, maximum ten allocated GPU-hours; expected
90–120 minutes is extrapolated, not a measured guarantee. Deterministic CUDA
aggregation is enabled for epoch replay. `last.pt` contains optimizer/RNG,
coverage, history and the epoch-boundary best, so interrupted half-epoch best
selection is rolled back before replay. Initial, selected and latest survive.

Output: `outputs/rank/humansl/policy_branches/seed2027_b32_e12/<job_id>/`.
Common games/questions/raw C use relative links. Persistent records include
source/config/plan snapshots, environment, coverage per k, extraction precision,
all evaluation metrics and deltas, q/added-logit RMS, compressed per-question
predictions for all 25 TRAIN/monitor checkpoints, every-update gradient and
update norms by group, selected features/predictions for all roles, player
bootstrap comparisons (2,000 draws, unrounded gates), parameter counts, GPU
memory, wall time and disk bytes. The 200 MiB disk budget excludes shared data
links and the external Slurm stdout file. No full policy/gradient tensors are
persisted. Missing root `complete.json` means the experiment is incomplete.

Validation: `PYTHONPATH=src:. python -m pytest -q tests/test_policy_branches.py
 tests/test_policy_residual.py tests/test_policy_residual_epoch.py` (one line).
Submission is followed by no queue/execution polling; analysis waits for the user.


### TRAIN/monitor persistent cache (2026-11-17)

`python -m scripts.strength.select_cached_policy` freezes the A+B/original-monitor split in
`data/processed/strength_features/humansl/policy_cnn/cached_AB_mondev_seed2027/` (refuses overwrite).
`python -m go_aicup.tasks.strength.policy_residual.persistent_cache --validate-only` checks sources,
packed headers, CPU ZIP/replay parity, C-vector player overlap, and free space without GPU work.
Submit `src/go_aicup/tasks/strength/policy_residual/persistent_cache.slurm` for 5 GPUs, 60 CPUs,
500 GiB RAM, and a 2-hour bound. It only extracts TRAIN/monitor; dev and model training are deferred.

The optimized pipeline runs two independent producers per GPU: ZIP inputs have a four-shard buffer;
replay workers continuously rebuild missing games into a bounded RAM buffer sized to that GPU's
missing shards (about 2 GiB of packed replay inputs across the full job). Each process pool keeps two
shards submitted. Inputs stay bit-packed through IPC and H2D, then reuse `humansl_rankvec.unpack`
on the GPU. No supplementary input planes are persisted. Output `policy/unit_*/` shards contain `logp.npy` (FP16), `q.npy` (original FP32),
`target.npy`, `side.npy`, `offsets.npy`, `game_index.npy`, and a checksum completion marker.
The same command resumes verified shards; changed sources or corrupted shards fail explicitly.
One background writer per GPU waits for the output-copy CUDA event, checks numerical precision,
publishes and re-reads each shard to verify checksums. At most two outputs are in flight, so slow I/O
applies bounded backpressure and failures propagate. No per-inference-batch `.cpu()` or scalar reads
synchronize the GPU. Final publication checks the writer receipts rather than re-reading all new
policy data. Resumed shards are still fully verified. Worker reports include input wait time, writer
backpressure, and CUDA-event inference/copy times; these are not hardware utilization percentages.

Consumers must verify `policy/complete.json` and the referenced shard markers before training.
At the user's request, the cancelled job 792644's outputs (25 shards, about 1.3 GiB) are removed before
restarting the optimized schema-2 extractor under the original `policy/` and `extract/<job_id>/` names.
Unused `policy_stream_v2/` and `extract_stream_v2/` preflight artifacts are removed too. Shared selection,
questions, ZIP inputs and teacher weights remain the same. Old and new extraction versions never mix.
Cold-start source/CRC checks, compilation, and final draining still have fixed overhead. GPU throughput
and asynchronous-copy behavior remain unmeasured. After submission, do not poll the new job.
The original `Cache`/training runners still use RAM; this extraction does not launch them.
