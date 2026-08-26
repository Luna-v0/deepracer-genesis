# Local install & run

DeepRacer-Genesis runs on **Linux x86-64 with an NVIDIA GPU**. Python 3.10–3.12
(3.12 recommended).

> Mental model in one sentence: `uv sync` installs Genesis + rsl-rl + TorchRL, then
> you train by defining an `Experiment` subclass and running it (recommended), or via
> the legacy flag-based `train.py` CLI.

---

## Install

```bash
git clone https://github.com/Luna-v0/deepracer-genesis && cd deepracer-genesis
uv venv --python 3.12 .venv && source .venv/bin/activate
uv sync                       # base deps
uv sync --extra tracking      # + mlflow (optional)
uv sync --extra hpo           # + optuna (optional)
```

Core dependencies (`pyproject.toml`): `genesis-world >= 1.2`, `rsl-rl-lib >= 5.4`,
`torch >= 2.5`, `torchrl`, `tensordict`, plus `imageio[ffmpeg]`, `pillow`, `numpy`,
`pyarrow`, `tensorboard`.

### CUDA 13 note (Madrona)

The Madrona batch renderer links `libnvrtc.so.12`. On a CUDA-13 system, run:

```bash
bash scripts/fix_madrona_cuda13.sh
```

which installs `nvidia-cuda-nvrtc-cu12`, symlinks the `.so.12` into `gs_madrona/`,
and patches the dlopen name. Feature-vector (no-camera) training does not need this.

## Train

**Experiment framework (recommended).** Author an `Experiment` subclass and run it
(see [Experiments](../concepts/experiments.md)):

```python
from deepracer_genesis.experiment import Experiment, FeatureEnvironment, VectorPolicy, run

class MyFirst(Experiment):
    total_env_steps = 5_000_000
    eval_every_steps = 1_000_000
    num_envs = 1024
    def pipeline(self):
        return FeatureEnvironment(num_envs=self.num_envs) >> VectorPolicy()

run(MyFirst)     # ~90s on an RTX 4060 Ti
```

Put it in a file under `experiments/` with `run(MyFirst)` under `__main__` and run it
directly:

```bash
uv run experiments/my_first.py
```

**Legacy CLI** (flag-based, separate entry point):

```bash
python -m deepracer_genesis.train -B 4096 --max_iterations 500 --exp_name teacher
#   -B/--num_envs, --max_iterations, --vision, --nyx, --randomize, --track, --resume
```

## Evaluate & inspect

```bash
python -m deepracer_genesis.eval --checkpoint runs/.../best.pt --num_envs 24 --res 1280x960
python -m deepracer_genesis.validation.camera_check --num_envs 4
python -m deepracer_genesis.validation.dr_check --knobs world_color,brightness   # see the DR editor guide
tensorboard --logdir runs/
```

## Output layout

```
runs/<group>/<variant>-<seed>-<id>/
  best.pt           # actor + critic weights + spec
  spec.json         # config record
  eval_record.json  # final + periodic metrics
  events.out.*      # TensorBoard
  videos/           # rollout videos
```
