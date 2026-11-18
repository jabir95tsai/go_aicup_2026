# Large policy experiment

Approved eight-GPU 200k TRAIN experiment. The full specification is in root `plan.md`.
GPU0–3 own H shards; GPU4–7 own C shards. Each teacher sees all selected positions exactly
once. Eight deeper experimental encoders train concurrently after the two A0 fits.
Selected per-game features live in RAM; all thirty heads are locked before dev scoring.
No input planes, policy, q, per-game or B3 feature cache is written to disk.

```bash
PYTHONPATH=src:. ~/.conda/envs/go/bin/python -m go_aicup.tasks.strength.policy_large.run --prepare-only
sbatch src/go_aicup/tasks/strength/policy_large/train.slurm
```

The user explicitly requested full submission without a GPU smoke test. CPU validation
and source/capacity checks remain mandatory. Maximum allocation: eight H200, 96 CPUs,
200G RAM, 36h (288 allocated GPU-h). Output: config's out / Slurm job ID.

`data/` contains only split and question tables; `source/` archives source code. Family/arm
folders contain model, optimizer state and learning curves; B3 has five arms x four k heads.
Dev predictions are output reports, not training features. `summary.json` and root
`complete.json` are published only when all training, dev and bootstrap finish.

An incomplete run cannot automatically resume: volatile policy disappears with the job.
The entry point refuses to rerun teachers under an existing incomplete identity. Completed
runs reopen without GPU work. No automatic requeue, submission CSV, upload or promotion.
