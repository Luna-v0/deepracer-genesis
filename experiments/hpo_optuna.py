"""Hyperparameter optimization with Optuna — single file, single GPU.

Trials run IN-PROCESS (Genesis scenes rebuild fine within one process), so
this is plain Python end to end: sample a config, `run()` it, report the
periodic deterministic evals (`eval_every_steps`) to the pruner through the
trainer's `on_eval` hook, and let Hyperband kill bad trials mid-training.
The study is resumable (sqlite), and the content-hash identity cache makes
re-sampled duplicate configs free.

Run:  uv run experiments/hpo_optuna.py
"""

from __future__ import annotations

import os

import optuna

from deepracer_genesis.experiment import FeatureEnvironment, VectorPolicy, run
from deepracer_genesis.experiment.ablation import override

STEPS = int(os.environ.get("HPO_STEPS", 5_000_000))
EVAL_EVERY = int(os.environ.get("HPO_EVAL_EVERY", 500_000))
N_TRIALS = int(os.environ.get("HPO_TRIALS", 20))
METRIC = "completion_rate"


def objective(trial: optuna.Trial) -> float:
    spec = (FeatureEnvironment(lookahead_k=10, num_envs=1024)
            >> VectorPolicy()).build(seed=0, total_env_steps=STEPS,
                                     eval_every_steps=EVAL_EVERY,
                                     ablation_group="hpo")
    for key, value in {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "entropy_coef": trial.suggest_float("entropy_coef", 1e-3, 3e-2, log=True),
        "epochs": trial.suggest_int("epochs", 3, 8),
        "clip": trial.suggest_float("clip", 0.1, 0.3),
    }.items():
        spec = override(spec, f"algorithm.ppo.{key}", value)

    def report(frames: int, metrics: dict) -> None:
        trial.report(metrics[METRIC], frames)
        if trial.should_prune():
            raise optuna.TrialPruned()

    record = run(spec, root="runs/hpo", force=True, on_eval=report)
    return float(record.metrics[METRIC])


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
    study.optimize(objective, n_trials=N_TRIALS)
    print("best:", study.best_value, study.best_params)
