# DeepRacer-Genesis

AWS DeepRacer reinforcement learning rebuilt on
[Genesis](https://genesis-world.readthedocs.io/): GPU-batched physics,
rsl-rl PPO, config-as-code experiments, a domain-randomization catalog, a
multi-track zoo, and an ONNX pipeline back onto the physical car.

```python
from deepracer_genesis.experiment import Experiment, FeatureEnvironment, VectorPolicy, run

class MyFirst(Experiment):
    total_env_steps = 5_000_000
    def pipeline(self):
        return FeatureEnvironment(num_envs=1024) >> VectorPolicy()

run(MyFirst)        # ~90 s on an RTX 4060 Ti
```

## The learning path

The docs follow the same road the project drove. Each step is a working
experiment before the next one makes it harder:

1. **[Quick start](quickstart.md)** — install, train a state-vector policy
   in ~90 seconds, watch it drive. This is the "everything works" baseline:
   feature-vector PPO completes 100% of laps.
2. **[Tutorial](tutorial.md)** — the experiment idiom in full: stages, the
   `>>` DSL, specs, run directories, evaluation records.
3. **See before you train** — cameras change everything:
   [feature vector vs camera](concepts/feature-vs-camera.md) is the fork in
   the road (30× the training cost buys the only modality the physical car
   can run). Preview exactly what the cars will see with
   [`view_zoo`](concepts/renderers.md) (onboard, at the car's native
   160×120) and the [track zoo](guides/track-zoo.md) viewers (bird's-eye),
   and pick a renderer with the
   [renderer comparison](concepts/renderers.md) — measured numbers, not
   vibes: the path tracer is beautiful and ≈11× slower.
4. **Randomize** — the [DR concepts page](concepts/domain-randomization.md)
   explains the layers; the [DR catalog](reference/dr-catalog.md) is the
   full table of all 31 knobs with ranges and the compatibility matrix the
   build validates against.
5. **Shape the reward** — the
   [reward parameters](reference/reward-parameters.md) reference lists
   every term and scale, and
   [Rewards & actions](concepts/rewards-actions.md) tells the cautionary
   tale: the difference between a policy that finishes 8% of laps and one
   that finishes 78% on a never-seen track was mostly the reward design,
   not the network.
6. **Scale the search** — [HPO](guides/hpo.md) with Optuna over
   hyperparameters, and — because reward functions are ordinary values
   here — over reward *designs* too.
7. **Deploy** — [export to ONNX](guides/deployment.md) (opset 11,
   OpenVINO-2021.1-compatible) and put the bundle on the car.

## Reference shelves

When you know what you are looking for:

- [Reward parameters](reference/reward-parameters.md) — every term,
  formula, sign, and default scale.
- [DR catalog](reference/dr-catalog.md) — all randomization knobs in one
  table.
- [Renderer comparison](concepts/renderers.md) — Madrona vs Nyx vs the CPU
  rasterizer, with rendered samples and measured throughput.
- [API reference](api/experiment/index.md) — generated from the
  Google-style docstrings.

Operational notes (renderer quirks, benchmarks, CUDA fixes) live in the
repo README.
