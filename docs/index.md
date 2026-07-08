# DeepRacer-Genesis

AWS DeepRacer reinforcement learning on [Genesis](https://genesis-world.readthedocs.io/) —
GPU-batched physics, TorchRL PPO, config-as-code experiments, domain
randomization, and a sim2real dataset pipeline.

```python
from deepracer_genesis.experiment import Experiment, FeatureEnvironment, VectorPolicy, run

class MyFirst(Experiment):
    total_env_steps = 5_000_000
    def pipeline(self):
        return FeatureEnvironment(num_envs=1024) >> VectorPolicy()

run(MyFirst)        # ~90 s on an RTX 4060 Ti
```

Start with the [tutorial](tutorial.md); the README in the repo holds the
operational notes (renderer quirks, benchmarks, CUDA 13 fix).
