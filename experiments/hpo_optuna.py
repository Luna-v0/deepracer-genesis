"""Hyperparameter optimization with Optuna — single-file, single GPU.

The integration constraints and how this template deals with them:

- Genesis builds ONE scene per process, so every trial runs in a fresh
  subprocess (`uv run` this file; each trial is `python -c "... run(...)"`).
  The GPU is used by exactly one trial at a time — no contention.
- The HPO *signal* is the trainer's periodic deterministic evaluation
  (`eval_every_steps`): every eval prints `[trainer] eval @ N: completion X`
  and lands in `EvalRecord.eval_history`. This file tails the subprocess
  output, reports each eval to Optuna, and PRUNES bad trials mid-training
  (HyperbandPruner) — a pruned 20M-step trial costs only its first evals.
- Specs are content-hashed: a re-sampled duplicate config is a free cache
  hit, and the best trial's run directory (checkpoint, TensorBoard, record)
  already exists when the study finishes.

Run:  uv run experiments/hpo_optuna.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import optuna

STEPS = int(os.environ.get("HPO_STEPS", 5_000_000))
EVAL_EVERY = int(os.environ.get("HPO_EVAL_EVERY", 500_000))
N_TRIALS = int(os.environ.get("HPO_TRIALS", 20))
METRIC = "completion_rate"

_EVAL_RE = re.compile(r"\[trainer\] eval @ (\d+): completion ([\d.]+)")


def launch_trial(trial: optuna.Trial) -> float:
    """Sample a config, train it in a subprocess, prune on periodic evals."""
    overrides = {
        # ExperimentSpec-level knobs route by name through run()
        "eval_every_steps": EVAL_EVERY,
        "total_env_steps": STEPS,
        "seed": 0,
    }
    ppo = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "entropy_coef": trial.suggest_float("entropy_coef", 1e-3, 3e-2, log=True),
        "epochs": trial.suggest_int("epochs", 3, 8),
        "clip": trial.suggest_float("clip", 0.1, 0.3),
    }

    code = f"""
import experiments, json, dataclasses
from deepracer_genesis.experiment import run
from deepracer_genesis.experiment.ablation import override
spec = run("feature_baseline", build_only=True, **{overrides!r})
for k, v in {ppo!r}.items():
    spec = override(spec, "algorithm.ppo." + k, v)
record = run(spec, force=True, root="runs/hpo")
print("FINAL_METRIC", json.dumps(record.metrics))
"""
    proc = subprocess.Popen([sys.executable, "-u", "-c", code],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    final = None
    for line in proc.stdout:
        m = _EVAL_RE.search(line)
        if m:
            frames, completion = int(m.group(1)), float(m.group(2))
            trial.report(completion, frames)
            if trial.should_prune():
                proc.kill()
                proc.wait()
                raise optuna.TrialPruned()
        elif line.startswith("FINAL_METRIC"):
            final = json.loads(line.split(" ", 1)[1])
    proc.wait()
    if proc.returncode != 0 or final is None:
        raise RuntimeError(f"trial subprocess failed (exit {proc.returncode})")
    return float(final[METRIC])


if __name__ == "__main__":
    os.makedirs("runs/hpo", exist_ok=True)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=EVAL_EVERY, max_resource=STEPS, reduction_factor=3),
        study_name="feature_ppo",
        storage="sqlite:///runs/hpo/study.db",   # resumable + inspectable
        load_if_exists=True,
    )
    study.optimize(launch_trial, n_trials=N_TRIALS)
    print("best:", study.best_value, study.best_params)
