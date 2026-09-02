# Experiments & the `>>` DSL

Experiments are **config-as-code**: you compose an environment, optional domain
randomization, a policy, and (optionally) an algorithm into an immutable
`ExperimentSpec` using the `>>` operator, then `build()` or `run()` it.

> Mental model in one sentence: each `>>` stage is a pure function
> `ExperimentSpec -> ExperimentSpec`; the chain must start with an environment and
> contain exactly one policy, and `build()` folds the stages into a validated,
> frozen spec.

---

## The pipeline

`>>` chains `Stage` objects into a `Pipeline` (`experiment/stages.py`). A single
stage becomes a one-element pipeline (`Stage.__rshift__`), and pipelines concatenate
(`Pipeline.__rshift__`). `Pipeline.build(**overrides)`:

1. validates structure (`_check_structure`),
2. folds each stage's `apply()` into a fresh `ExperimentSpec`,
3. infers the algorithm if none was set (`_infer_algorithm`),
4. calls `spec.validate()` and returns the frozen spec.

Structure rules: the **first stage must be an environment**; **exactly one policy**
stage; at most one each of encoder / action-DR / algorithm / camera-DR / physics-DR;
zero or more reward-shaping and DR stages.

## Stage catalog

| Role | Stage | Key params |
|------|-------|-----------|
| Env | `FeatureEnvironment` | `feature_set`, `feature_params`, `lookahead_k`, `tracks`, `num_envs`, `random_start`, `random_direction`, `max_speed` |
| Env | `CameraEnvironment` | `render` (`"madrona"`/`"nyx"`), `resolution`, `fov`, `feature_set`, `tracks`, `num_envs`, `max_speed` |
| Env (safe-RL) | `SafeRLFeatureEnvironment` / `SafeRLCameraEnvironment` | above + `cost`, `budget` (emits a cost stream → infers PPO-Lagrangian) |
| Reward | `RewardShaping` | `fn` (custom `RewardFn` or None), `scales` (override dict) |
| DR | `DomainRandomizationTrackAppearance` | `strength` |
| DR | `DomainRandomizationCamera` | `brightness`, `contrast`, `saturation`, `hue`, `blur`, `cutout`, `noise`, `camera_jitter` |
| DR | `DomainRandomizationPhysics` | `friction`, `mass`, `com`, `gains`, `armature` |
| DR | `DomainRandomizationActions` | `steer_noise`, `speed_noise`, `delay_steps` |
| Encoder | `FrozenCNNToFeatureVector` | `checkpoint`, `output_dim`, `layer`, `out_key` |
| Policy | `VectorPolicy` | `keys`, `mlp`, `actions` |
| Policy | `AsymmetricVectorPolicy` / `AsymmetricCameraPolicy` | `actor_keys`, `critic_keys`, `mlp`/`cnn`, `actions` |
| Algorithm | `PPO`, `PPOLagrangian`, `Algo(cls=...)` | `lr`, `clip`, `epochs`, `minibatches`, `gamma`, `gae_lambda`, `entropy_coef`, `max_grad_norm`, `horizon`, `schedule`, `desired_kl`; `Algo(cls=...)` in [Custom algorithms](../guides/custom-algorithms.md) |

Policies expose **`actor_keys` / `critic_keys`** — the asymmetric-critic hook: the
critic can read richer observation keys than the actor (e.g. `critic_keys=("camera",
"state")` with `actor_keys=("camera",)`).

`max_speed` caps the top of the speed action range in m/s (`-1 → min_speed`,
`+1 → max_speed`); leave it `None` to keep the physics default (4.0 m/s). PPO's
`schedule` is `"adaptive"` (retune `lr` from the measured KL, steering toward
`desired_kl`) or `"fixed"` (keep `lr`).

## Authoring an experiment

Author each experiment as an **`Experiment` subclass**: training config as class
attributes, the `>>` chain in `pipeline()`. A variant is a subclass (or an override
like `FeatureBaseline(num_envs=256)`).

```python
from deepracer_genesis.experiment import Experiment, FeatureEnvironment, VectorPolicy, run

class FeatureBaseline(Experiment):
    seed = 0
    total_env_steps = 5_000_000
    eval_every_steps = 1_000_000
    num_envs = 1024
    def pipeline(self):
        return (FeatureEnvironment(num_envs=self.num_envs, lookahead_k=10)
                >> VectorPolicy(keys=("state",)))

class FeatureBaselineSmall(FeatureBaseline):     # a variant
    num_envs = 256

class FeatureBaselineFineTune(FeatureBaseline):  # start from trained weights
    resume = "runs/.../model_1500.pt"

run(FeatureBaseline)
```

Camera + full DR + transfer encoder + safe-RL follows the same pattern:

```python
class SafeTransfer(Experiment):
    render = "madrona"; budget = 25.0; ckpt = "runs/.../best.pt"
    def pipeline(self):
        return (
            SafeRLCameraEnvironment(render=self.render,
                                    cost="offtrack_or_overspeed", budget=self.budget)
            >> DomainRandomizationCamera(brightness=(0.7, 1.3))
            >> FrozenCNNToFeatureVector(checkpoint=self.ckpt, output_dim=256)
            >> VectorPolicy(keys=("encoded", "state"))
            >> DomainRandomizationActions(steer_noise=0.02)
        )
```

The safe-RL env emits a cost stream, so `_infer_algorithm` selects PPO-Lagrangian
automatically. For full control (e.g. a dynamic `variant` name) override `spec()`
instead of `pipeline()`.

## build / run

- `build(target, **overrides)` — validate → frozen `ExperimentSpec`. `target` is an
  `Experiment` subclass/instance, a `Pipeline`, or a spec.
- `run(target, *, root="runs", **overrides)` — build, then train via
  `Trainer(Builder(spec))`; returns the eval record. `MyExperiment().run()` and
  `uv run experiments/my_file.py` (with `run(MyExperiment)` under `__main__`) are
  equivalent entry points.

```python
from deepracer_genesis.experiment import run
run(FeatureBaseline, root="runs")
```
