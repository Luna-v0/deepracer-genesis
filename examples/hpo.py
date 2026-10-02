"""HPO example: one Optuna study over PPO knobs AND network architecture.

Because network shape is ordinary spec data, the same study that tunes
learning rate can search topology: MLP depth/width, the activation
function, and MLP vs recurrent (LSTM/GRU) actors — the recurrent knobs are
conditional parameters, only suggested when the recurrent arm is drawn.

Trials run in-process: sample PPO knobs from a declarative ``{name: Space}``
map (the SAME Space types domain randomization uses — Part H), sample the
architecture with Optuna's conditional API, ``run()`` the spec, and report
the periodic deterministic evals to the pruner so Hyperband can kill bad
trials mid-training. Not a single registered experiment (it is a study), so
it runs as a script:

    uv run examples/hpo.py
"""

from __future__ import annotations

import os

import optuna

from deepracer_genesis.experiment import FeatureEnvironment, PPO, VectorPolicy, run
from deepracer_genesis.randomization.spaces import FloatRange, IntRange

STEPS = int(os.environ.get("HPO_STEPS", 5_000_000))
EVAL_EVERY = int(os.environ.get("HPO_EVAL_EVERY", 500_000))
N_TRIALS = int(os.environ.get("HPO_TRIALS", 20))
METRIC = "completion_rate"

# declare the PPO search space once; each site chooses its verb (suggest vs sample)
SEARCH_SPACE = {
    "lr": FloatRange(1e-4, 1e-2, log=True),
    "entropy_coef": FloatRange(1e-3, 3e-2, log=True),
    "epochs": IntRange(3, 8),
    "clip": FloatRange(0.1, 0.3),
}


def suggest_architecture(trial: optuna.Trial) -> dict:
    """Sample the policy network's shape as an ``mlp`` config dict.

    Args:
        trial: The Optuna trial (recurrent knobs are conditional — only
            suggested when the lstm/gru arm is drawn).

    Returns:
        The ``mlp`` dict for ``VectorPolicy``: layer widths (a halving
        funnel from the sampled width/depth), activation, and — for the
        recurrent arms — an ``rnn`` config selecting rsl-rl's ``RNNModel``.
    """
    depth = trial.suggest_int("depth", 2, 4)
    width = trial.suggest_categorical("width", [128, 256, 512])
    mlp = {
        "hidden": tuple(max(width // 2**i, 32) for i in range(depth)),
        "activation": trial.suggest_categorical("activation",
                                                ["elu", "relu", "tanh"]),
    }
    arch = trial.suggest_categorical("arch", ["mlp", "lstm", "gru"])
    if arch != "mlp":
        mlp["rnn"] = {"type": arch,
                      "hidden": trial.suggest_categorical("rnn_hidden",
                                                          [128, 256]),
                      "layers": 1}
    return mlp


def objective(trial: optuna.Trial) -> float:
    """Train one sampled (PPO knobs x architecture) config and score it.

    Args:
        trial: The Optuna trial.

    Returns:
        The final deterministic eval's ``completion_rate``.
    """
    p = {name: space.suggest(trial, name) for name, space in SEARCH_SPACE.items()}
    spec = (
        FeatureEnvironment(num_envs=1024)
        >> VectorPolicy(keys=("state",), mlp=suggest_architecture(trial))
        >> PPO(lr=p["lr"], entropy_coef=p["entropy_coef"],
               epochs=p["epochs"], clip=p["clip"])
    ).build(seed=0, total_env_steps=STEPS, eval_every_steps=EVAL_EVERY,
            group="hpo", variant=f"t{trial.number:03d}")

    def report(frames: int, metrics: dict) -> None:
        trial.report(metrics[METRIC], frames)
        if trial.should_prune():
            raise optuna.TrialPruned()

    record = run(spec, root="runs/hpo", on_eval=report)
    return float(record.metrics[METRIC])


if __name__ == "__main__":
    os.makedirs("runs/hpo", exist_ok=True)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=EVAL_EVERY, max_resource=STEPS, reduction_factor=3),
        study_name="feature_ppo_arch",
        storage="sqlite:///runs/hpo/study.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIALS)
    print("best:", study.best_value, study.best_params)
