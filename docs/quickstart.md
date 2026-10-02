# Quick start

From clone to a trained driving policy in about ten minutes of wall-clock
(feature-vector training is fast; camera training comes later in the
[learning path](index.md#the-learning-path)).

## 1. Install

The recommended path is installing it as a package into your own project,
straight from git (the `[vision]` extra brings the Madrona renderer for
camera training; add `[nyx]` for the path tracer):

```bash
uv add "deepracer-genesis[vision] @ git+https://github.com/Luna-v0/deepracer-genesis"
```

Track assets ship inside the package, so this is everything you need. To
hack on the library itself, clone instead:

```bash
git clone https://github.com/Luna-v0/deepracer-genesis
cd deepracer-genesis
uv sync                    # default groups include the GPU renderers
```

No GPU is strictly required: `backend="cpu"` trains **everything** with no
GPU at all — feature-vector at genuinely useful speeds, and even camera
policies through the CPU rasterizer (measured ~90 env-steps/s: fine for
debugging, ~30× too slow for real camera runs). An NVIDIA GPU is what
makes camera training *practical* (Madrona/Nyx are CUDA renderers).
Feature-vector training also works on
[Google Colab](usage/google_colab_cli.md). See
[local setup](usage/local.md) for driver quirks.

## 2. Train your first policy

Experiments are Python classes — no CLI, no YAML. Declare, then `run`:

```python
from deepracer_genesis.experiment import (Experiment, FeatureEnvironment,
                                          VectorPolicy, run)

class MyFirst(Experiment):
    """State-vector PPO on the re:Invent track."""
    total_env_steps = 5_000_000

    def pipeline(self):
        return FeatureEnvironment(num_envs=1024) >> VectorPolicy()

record = run(MyFirst)      # ~90 s on an RTX 4060 Ti
print(record.metrics["completion_rate"])
```

That is the whole idiom: an env stage, `>>`-composed extras, a policy
stage. Everything else (PPO, eval cadence, seeding, run directory naming)
has working defaults, and every knob you did not set is recorded in the
run's spec dump.

## 3. See what you trained

```python
from deepracer_genesis.experiment import rollout_video

rollout_video(MyFirst)     # bird's-eye MP4 of the trained policy driving
```

Each run lands in `runs/<group>/<variant>-<seed>-<hash>/` with TensorBoard
scalars, periodic deterministic evals, per-step telemetry parquet (see
`deepracer_genesis.analysis`), and an `eval_record.json`.

```bash
tensorboard --logdir runs      # or %tensorboard in Jupyter
```

## 4. Go deeper

The [learning path](index.md#the-learning-path) continues from here:
cameras instead of state vectors, domain randomization, the multi-track
zoo, held-out generalization, and the ONNX export that puts the policy on
a physical car.
