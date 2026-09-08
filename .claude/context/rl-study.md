# Context: RL study pipeline (Gymnasium + HPO)

Read this when the task touches the RL study orchestrator — defining, running, or
analyzing studies over Gymnasium environments with hyperparameter optimization. It
layers domain specifics on top of the universal standards in the workspace `CLAUDE.md`
(the repo's parent folder). Skip it for pure simulator/analysis/utility work.

## This repo's contract (read first)
The training-facing contract here is **rsl-rl-lib 5.x VecEnv**, not raw Gymnasium:
TensorDict observation groups (`"state"` / `"camera"`), no `reset()` from the runner,
`extras["time_outs"]`. Where this file says "Gymnasium", read "the batched VecEnv
contract" — the reasoning (stable interface, wrappers as decorators, seeds threaded
through) carries over unchanged. HPO is Optuna, tracking is MLflow (both optional
extras in `pyproject.toml`).

## Domain conventions
- Follow the Gymnasium API (`reset`/`step`/`render`/`close`, `observation_space`/
  `action_space`) and treat environment wrappers as decorators, consistent with
  Gymnasium's own wrappers.
- **Reproducibility is paramount.** Thread seeds through the environment, the
  algorithm, and the study — the same config + seed must reproduce results.

## Ports (define as Protocols; the core depends only on these)
- **EnvironmentPort** — construct/reset/step environments (Gymnasium adapter).
- **OptimizerPort** — the HPO backend: suggest params, report results, manage the
  study (Optuna adapter — swap the adapter if you use Ray Tune, W&B Sweeps, etc.).
- **TrackerPort** — metrics, logging, checkpointing (MLflow / W&B / filesystem
  adapter).
- **StoragePort** — study storage/DB, checkpoints, results.

## Patterns (apply where they help)
- **Registry + Factory** to construct environments, agents, and algorithms by
  name/config (mirrors Gymnasium's registry).
- **Strategy** for interchangeable algorithms, policies, exploration schedules, and
  reward shaping.
- **Observer / callbacks** for logging, metrics, checkpointing, and early stopping.
- **Builder** to assemble a study from config; **Template Method** for the train/eval
  loop skeleton.
- **Pure functions** for reward and metric computation; **frozen dataclasses** for
  `StudyConfig` and result objects.

## Smoke pre-flight (quick and bounded — a sanity check, NOT the crash catcher)
- Maintain a harness (`uv run python -m <pkg>.smoke` and/or `uv run pytest -m smoke`)
  that runs the REAL pipeline end to end — real Gymnasium env, real HPO study — with
  MINIMAL settings: a fast env (e.g. CartPole), a few trials, few timesteps each.
- Keep it SHORT and bounded. It checks the wiring holds together: the study starts,
  runs trials, and finishes; best params/results are retrievable; metrics are finite;
  expected artifacts (logs, checkpoints, study storage) are written; no unhandled
  exceptions. Run it before RL-pipeline-affecting work is called done.
- **This is NOT the instrument for long-run failures.** For the simulator crash near
  ~30 min, use the SOAK run + health checkpoint in `.claude/playbooks/debug.md`, which
  deliberately runs past 30 minutes. Smoke = fast pre-flight; soak = long-run health.
