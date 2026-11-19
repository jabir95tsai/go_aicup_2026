# Large policy experiment

Approved eight-GPU 200k TRAIN experiment. The full specification is in root `plan.md`.
GPU0–3 own H shards; GPU4–7 own C shards. Each teacher sees all selected positions exactly
once. H and C run in separate spawned processes, each advancing immediately from its
teacher to its initial LR, A0, four parallel experimental encoders, and selected feature
extraction on its own four GPUs. The processes isolate teacher TF32/FX state from the
other family's training. A shared cancellation event stops both families on failure.
B3 waits for both families; selected per-game features cross back to the parent in RAM.
All thirty heads are locked before dev scoring.
No input planes, policy, q, per-game or B3 feature cache is written to disk.

```bash
PYTHONPATH=src:. ~/.conda/envs/go/bin/python -m go_aicup.tasks.strength.policy_large.run --prepare-only
sbatch src/go_aicup/tasks/strength/policy_large/train.slurm
```

CPU validation and source/capacity checks remain mandatory. Maximum allocation: eight H200, 96 CPUs,
200G RAM, 36h (288 allocated GPU-h). Output: config's out / Slurm job ID.

`data/` contains only split and question tables; `source/` archives source code. Family/arm
folders contain model, optimizer state and learning curves; B3 has five arms x four k heads.
Dev predictions are output reports, not training features. `summary.json` and root
`complete.json` are published only when all training, dev and bootstrap finish.

An incomplete run cannot automatically resume: volatile policy disappears with the job.
The entry point refuses to rerun teachers under an existing incomplete identity. Completed
runs reopen without GPU work. No automatic requeue, submission CSV, upload or promotion.

For the completed run 806639, `sbatch src/go_aicup/tasks/strength/policy_large/export.slurm`
infers A0, R10 and CNN-Z32 through their frozen H/C encoders and four selected B3 heads.
It writes `submissions/rank/806639-hc-{a0,r10,cnn-z32}.csv`, with score_rank decoding.
There is no refitting, prior adjustment or L10 overlay. Public and Private each have
400 complete questions; all 10,000 games use HD0 komi 6.5 or HD1 komi 0.5.
Reusable per-game H/C features, per-question B3 inputs and probabilities are saved in
`data/processed/strength_features/humansl/policy_large/806639_test/` with a final manifest.
Whole-board policy tensors remain in VRAM only. Export diagnostics and replay arrays
are under the training run's `test_export/`. Existing submission files are not overwritten.
