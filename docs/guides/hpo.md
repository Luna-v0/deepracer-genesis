# Hyperparameter optimization

HPO searches training hyperparameters with [Optuna](https://optuna.org/), reusing
the same `Space` types that [domain randomization](../concepts/domain-randomization.md)
uses. The reference study is `experiments/hpo_optuna.py`.

> Mental model in one sentence: declare a `{name: Space}` search space, have each
> trial `suggest()` a value into the `>>` chain, train from scratch in-process, and
> report the eval metric back to Optuna's pruner.

---

## Declaring the search space

A dict of `Space` objects (`randomization/spaces.py`):

```python
SEARCH_SPACE = {
    "lr":           FloatRange(1e-4, 1e-2, log=True),
    "entropy_coef": FloatRange(1e-3, 3e-2, log=True),
    "epochs":       IntRange(3, 8),
    "clip":         FloatRange(0.1, 0.3),
}
```

`FloatRange`/`IntRange` implement both `suggest(trial, name)` (HPO) and
`sample(n, device)` (DR). `Choice` is HPO-only (categorical); `SymRange` is DR-only
(`suggest` raises — a symmetric magnitude has no single scalar to freeze).

## The objective

Each trial suggests values, folds them **directly into stage constructors** (no
dotted-path override needed), builds, and trains:

```python
def objective(trial):
    p = {name: space.suggest(trial, name) for name, space in SEARCH_SPACE.items()}
    spec = (
        FeatureEnvironment(lookahead_k=10, num_envs=1024)
        >> VectorPolicy()
        >> PPO(lr=p["lr"], entropy_coef=p["entropy_coef"],
               epochs=p["epochs"], clip=p["clip"])
    ).build(seed=0, total_env_steps=STEPS, eval_every_steps=EVAL_EVERY,
            group="hpo")

    def report(frames, metrics):
        trial.report(metrics[METRIC], frames)
        if trial.should_prune():
            raise optuna.TrialPruned()

    record = run(spec, on_eval=report)
    return record.metrics[METRIC]     # e.g. completion_rate
```

## The study

```python
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
```

Trials run **in-process** (Genesis rebuilds a scene fine within one process) and
each trains from scratch in its own content-hashed run dir. Hyperband prunes
underperforming trials mid-training via the `on_eval` callback.

## Running

```bash
uv sync --extra hpo
HPO_STEPS=5_000_000 HPO_EVAL_EVERY=500_000 HPO_TRIALS=20 \
    uv run experiments/hpo_optuna.py
```

The SQLite storage makes the study resumable and inspectable with Optuna's
dashboard. `notebooks/hpo_cnn.ipynb` shows the same pattern searching CNN
architecture for a camera policy.

## Searching architectures and topologies

Network shape is spec data, so it goes into a study like any hyperparameter.
The policy stages accept an `mlp` dict with three knobs:

- `hidden` — tuple of layer widths (depth **and** width),
- `activation` — rsl-rl activation name (`"elu"`, `"relu"`, `"tanh"`, ...),
- `rnn` — `{"type": "lstm"|"gru", "hidden": int, "layers": int}` switches
  the actor and critic to rsl-rl's recurrent `RNNModel` (an LSTM/GRU trunk
  in front of the MLP head). **Vector policies only** — upstream `RNNModel`
  has no CNN trunk, and `spec.validate()` refuses the camera + rnn combo.

An Optuna objective that searches topology, activation, and MLP-vs-recurrent
in one space (conditional parameters — recurrent knobs are only suggested
when the recurrent arm is drawn):

```python
def objective(trial: optuna.Trial) -> float:
    depth = trial.suggest_int("depth", 2, 4)
    width = trial.suggest_categorical("width", [128, 256, 512])
    mlp = {
        # funnel: each layer half the previous, e.g. 512-256-128
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

    spec = (FeatureEnvironment(num_envs=1024)
            >> VectorPolicy(mlp=mlp)).build(
        seed=0, total_env_steps=STEPS, eval_every_steps=EVAL_EVERY,
        group="arch_search", variant=f"t{trial.number:03d}")
    return run(spec, root="runs/arch_search").metrics["completion_rate"]
```

For camera policies the searchable shape lives in the `cnn` dict
(`channels`/`kernels`/`strides`/`activation`) plus the same `mlp` head
knobs — see `AsymmetricCameraPolicy`. Everything a trial sets is part of
the spec's content hash, so each architecture gets its own run dir and the
study table doubles as an architecture leaderboard.
