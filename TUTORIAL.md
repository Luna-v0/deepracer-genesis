# Tutorial

From zero to a trained self-driving policy, step by step. Everything here
runs on one Linux machine with an NVIDIA GPU (Colab T4 works for the
feature-vector parts — see `notebooks/deepracer_genesis_colab.ipynb`).

## 1. Install

```bash
git clone https://github.com/Luna-v0/deepracer-genesis && cd deepracer-genesis
uv venv --python 3.12 .venv && source .venv/bin/activate
uv sync            # or: uv pip install -e . torchrl tensordict
# CUDA 13 system toolkit only: bash scripts/fix_madrona_cuda13.sh
```

## 2. Train your first policy (2 minutes)

Create `experiments/my_first.py`:

```python
from deepracer_genesis.experiment import Experiment, FeatureEnvironment, VectorPolicy, run

class MyFirst(Experiment):
    total_env_steps = 5_000_000        # ~90 s on a 4060 Ti
    eval_every_steps = 1_000_000
    num_envs = 1024

    def pipeline(self):
        return FeatureEnvironment(num_envs=self.num_envs) >> VectorPolicy()

if __name__ == "__main__":
    run(MyFirst)
```

```bash
uv run experiments/my_first.py
```

The policy learns from a 28-dim privileged state (velocities, track-relative
pose, lookahead waypoints). Watch training live:

```bash
tensorboard --logdir runs/
```

## 3. Watch it drive

```python
from deepracer_genesis.experiment.visualize import rollout_video
rollout_video("MyFirst")                       # bird's-eye mp4 in the run dir
rollout_video("MyFirst", track="Monaco")       # same policy, unseen track
```

## 4. Get more tracks / draw your own

```python
from deepracer_genesis.tools.track_builder import fetch_official_track
fetch_official_track("Vegas_track")            # any of 126 official routes
```

To design one interactively, open `notebooks/track_designer.ipynb`: sketch a
polygon, preview, `install_track("my_track", route)`, sanity-drive it, then
use `tracks=("my_track",)` in any env stage.

## 5. Camera policy with domain randomization

```python
from deepracer_genesis.experiment import (Experiment, CameraEnvironment,
    DomainRandomizationTrackAppearance, DomainRandomizationCamera,
    DomainRandomizationPhysics, DomainRandomizationActions,
    AsymmetricCameraPolicy, run)

class CamRacer(Experiment):
    total_env_steps = 20_000_000
    eval_every_steps = 2_000_000
    num_envs = 128

    def pipeline(self):
        return (CameraEnvironment(resolution=(160, 120), num_envs=self.num_envs,
                                  random_direction=True)
                >> DomainRandomizationTrackAppearance(strength=0.6)   # per-episode world colors
                >> DomainRandomizationCamera(brightness=(0.7, 1.3), hue=0.05)  # per-step image aug
                >> DomainRandomizationPhysics()                       # per-episode friction/mass/gains
                >> AsymmetricCameraPolicy(actor_keys=("camera",),     # actor sees pixels,
                                          critic_keys=("camera", "state"))  # critic also sees state
                >> DomainRandomizationActions(steer_noise=0.02, delay_steps=1))

if __name__ == "__main__":
    run(CamRacer)
```

The actor drives from pixels only; the critic gets privileged state too
(asymmetric actor-critic). See "Domain randomization" in the README for
exactly what varies when, and its cost.

## 6. Collect a perception dataset (no training)

A privileged expert with temporally-correlated noise drives under full DR
and records temporally-contiguous frames + aligned feature vectors — for
pretraining CNNs / sim2real:

```bash
python scripts/collect_sim2real.py --out datasets/sim2real --steps 2048 --num-envs 64
```

Parquet shards, rows sorted (env, t); a k-frame stack is k consecutive rows
of one env with the same `episode` id.

## 7. Sweeps, ablations, HPO

```python
from deepracer_genesis.experiment import run
from deepracer_genesis.experiment.ablation import seeds, sweep
for spec in seeds(run("MyFirst", build_only=True), 3):
    run(spec)                                   # 3 seeds, 3 run dirs

from deepracer_genesis.experiment.report import build_report
build_report("runs")                            # runs/report.md aggregate table
```

Hyperparameter optimization with pruning: `uv run experiments/hpo_optuna.py`
(TPE + Hyperband over lr/entropy/epochs/clip; trials are subprocesses; the
trainer's periodic evals are the pruning signal).

## 8. Experiment tracking

TensorBoard is always on (per run dir). For MLflow:

```bash
MLFLOW_TRACKING_URI=sqlite:///$PWD/mlflow.db uv run experiments/my_first.py
mlflow ui --backend-store-uri sqlite:///$PWD/mlflow.db
```

## 9. Which renderer when?

| | Madrona (`render="madrona"`) | Nyx (`render="nyx"`) | Rasterizer |
|---|---|---|---|
| Use for | **training camera policies** | correct-color eval/validation, small-fleet training | spectator videos, debugging |
| Speed @256 envs, 160x120 | ~27k steps/s | ~1.5k steps/s | ~600 steps/s |
| Colors | dash-hue quirk (R<->G on alpha textures) | ground truth | ground truth |
| Per-env scene variants | no (z-fights) | no (refuses heterogeneous) | yes |
| Notes | no mipmaps; needs Vulkan | OBJ meshes only; one scene per process | any resolution; drives the bird's-eye "spectator" camera everywhere |

Default recipe: train on Madrona; validate the policy's frames on Nyx
(`cam_nyx` experiment); record videos with the rasterizer spectator (that
happens automatically in `rollout_video`).

## 10. Saving / reusing models

A run directory is self-contained — copy it anywhere:

```
runs/<group>/<variant>-<seed>-<id>/
  best.pt            # actor+critic weights + the spec that trained them
  spec.json          # exact config (one-way record)
  eval_record.json   # final + periodic eval metrics
  events.out.*       # TensorBoard
  videos/            # rollout_video outputs
```

```python
rollout_video("MyFirst", ckpt="wherever/best.pt")     # load explicitly
```

On Colab, mount Drive and `shutil.copytree(run_dir, "/content/drive/MyDrive/...")`
— the notebook has a ready-made cell.
