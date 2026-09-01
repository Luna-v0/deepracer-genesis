# Local install & run

DeepRacer-Genesis runs on **Linux x86-64**. Everything trains CPU-only via
`backend="cpu"` (feature-vector at useful speeds; camera through the CPU
rasterizer at ~90 env-steps/s — debugging territory). An NVIDIA GPU makes
camera training practical: Madrona is ~30× faster. Python 3.10–3.12
(3.12 recommended).

> Mental model in one sentence: `uv sync` installs Genesis + rsl-rl (plus the
> GPU renderers), then you train by defining an `Experiment` subclass and
> running it.

---

## Install

### As a package (recommended)

Install straight from git into your own project — track assets ship inside
the package, so nothing else is needed:

```bash
uv add "deepracer-genesis[vision] @ git+https://github.com/Luna-v0/deepracer-genesis"
```

Extras: `[vision]` (Madrona batch renderer — needed for camera training),
`[nyx]` (path tracer), `[analysis]` (telemetry DataFrames + trajectory
plots), `[export]` (ONNX), `[hpo]` (optuna), `[tracking]` (mlflow).

### From a clone (to work on the library)

```bash
git clone https://github.com/Luna-v0/deepracer-genesis && cd deepracer-genesis
uv sync                       # base deps + GPU renderers (default groups)
uv sync --extra tracking      # + mlflow (optional)
uv sync --extra hpo           # + optuna (optional)
```

Core dependencies (`pyproject.toml`): `genesis-world >= 1.2.3`, `rsl-rl-lib >= 5.4`,
`torch >= 2.5`, `tensordict`, plus `imageio[ffmpeg]`, `pillow`, `numpy`,
`pyarrow`, `tensorboard`. The GPU renderers (`gs-madrona`, `gs-nyx`) install
via the default `renderers` group.

### CUDA 13 note (Madrona)

Historical: `gs-madrona >= 0.0.10` (what the default `renderers` group
installs) pulls its own `nvidia-cuda-nvrtc-cu12 >= 12.8` and links the
megakernel natively, so no fix is needed on CUDA-13 systems anymore.
`scripts/fix_madrona_cuda13.sh` remains only for pinned older
`gs-madrona 0.0.8` installs.

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
