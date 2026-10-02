# 0004 — Bounded run artifacts: best+last checkpoints, subsampled periodic telemetry

- Status: accepted
- Date: 2026-09-30

## Context

Run directories grew without bound: ~226 GB across genesis and studies by
2026-09-30. Two writers dominated:

- **Checkpoints (~119 GB).** rsl-rl saves `model_<iter>.pt` every
  `save_interval` (100) iterations and again whenever `learn()` returns.
  `run_rsl` calls `learn()` once per eval chunk, so each eval adds one more
  file. A camera checkpoint is 70 MB, and 47 MB of that is Adam state. Nothing
  reads them: consumers use `model.pt`, and `spec.resume` loads weights only.
- **Periodic-eval telemetry (~78 GB).** Periodic evals run on the training
  sim and recorded every step of every env. A 1024-env feature study produced
  141 MB per eval. The files were already float32 with zstd compression.
  Measured, lossless re-encoding (env-major sort, byte-stream-split, zstd-9)
  saved about 0 %: the data is high-entropy. Only fewer rows help.

Telemetry is needed for evaluation and must stay lossless.

## Decision

`EvalConfig` / `Evaluation` gain three output-policy knobs. They are
**hash-exempt** in `ExperimentSpec.id()`, because they never change training:

- `telemetry_envs: int | None = 64`: periodic-eval telemetry records this many
  evenly spaced envs (`analysis.telemetry.telemetry_env_ids`). None records all
  of them. Metrics still cover every env, and the final and holdout telemetry
  stay full.
- `keep_checkpoints: "best_last" | "all" = "best_last"`: after each eval
  chunk, a full `model_last.pt` replaces rsl-rl's `model_<iter>.pt` files.
  This happens before `on_eval`, so HPO-pruned trials are covered too. When
  the run finishes, `model.pt` holds the same full state and `model_last.pt`
  is removed. A finished run therefore keeps `model.pt` + `model_best.pt`; a
  stopped one keeps `model_last.pt` + `model_best.pt`.
- `best_metric = "mean_progress_m"` (one of `BEST_METRICS`, all
  higher-is-better): each eval that strictly improves it writes a
  **weights-only** `model_best.pt`, including the final eval. NaN never
  wins, and a tie keeps the earlier checkpoint. Provenance is recorded in
  `EvalRecord.train` (`best_checkpoint`, `best_metric`, `best_value`,
  `best_frames`).

## Alternatives

- *Quantized or float16 telemetry* (2.4–3.1× smaller). Rejected because
  telemetry must stay lossless.
- *Periodic evals with no telemetry, keeping per-episode aggregates only.*
  Rejected because per-step traces are needed across training.
- *Raising `save_interval`.* This does not bound the per-chunk saves, and
  leaves no "best" checkpoint.
- *Lower-is-better metrics (`offtrack_rate`, `lap_time_s`) for `best_metric`.*
  These would need a direction flag. Deferred until someone needs it.

## Consequences

- A camera run now keeps about 95 MB of checkpoints instead of tens of GB.
  Periodic telemetry is `telemetry_envs / num_envs` of its old size (16× on
  1024-env studies).
- Code that globbed `model_<iter>.pt` must use `model.pt`, `model_best.pt` or
  `model_last.pt`, or opt back in with `keep_checkpoints="all"`.
- Periodic-telemetry analyses see a subset of envs. Episode aggregates drawn
  from it are noisier than the eval metrics in `eval_record.json`, which
  still cover every env.
- Existing spec ids and run dirs are unchanged.
