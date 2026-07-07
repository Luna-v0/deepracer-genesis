# deepracer-genesis

**New here? Start with [TUTORIAL.md](TUTORIAL.md).**

AWS DeepRacer RL environment ported from ROS/Gazebo to [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) —
ROS-free, GPU-batched, vision-based, trained with rsl-rl-lib PPO.

- Car URDF + track meshes + waypoint routes reused from the original
  [aws-deepracer-community/deepracer-simapp](https://github.com/aws-deepracer-community/deepracer-simapp)
  (via the `seresheim/deepracer-env` fork); everything ROS/Gazebo-specific is gone.
- Original action/observation semantics preserved: actions map to
  `Box([-30deg, 0.1 m/s], [+30deg, 4.0 m/s])`, the onboard camera is a single
  front RGB camera at 160x120 (rendered per env by the Madrona `BatchRenderer`).
- rsl-rl-lib **5.x** VecEnv contract (TensorDict observation groups
  `"state"` / `"camera"`, no `reset()` from the runner, `extras["time_outs"]`).

## Renderers (one stack, three options)

| Renderer | Use for | Quality | Colors | Vision steps/s (RTX 4060 Ti, 160x120) |
|---|---|---|---|---|
| Madrona batch (`vision_renderer="batch"`, default) | **training camera policies** | rasterized | dash-hue quirk | ~27.5k @ 256 envs |
| Nyx (`vision_renderer="nyx"`, `pip install gs-nyx-plugin`) | correct-color eval / validation | path traced | correct | ~1.5k @ 256 envs |
| per-env rasterizer (`raster-vision` branch; also the spectator cam) | videos, debugging, per-env scene variants | rasterized | correct | ~600 ceiling |

Default recipe: train on Madrona, validate frames on Nyx, record videos with
the rasterizer spectator (`rollout_video` does this automatically). Full
decision guide + walkthrough: [TUTORIAL.md](TUTORIAL.md).

Nyx facts: reads OBJ not DAE (tracks ship converted under `assets/tracks/*/obj/`),
requires unmerged URDF links, denoise/AA kept off (temporal history smears
moving objects), no heterogeneous multi-track scenes, driver 575+.


## Layout

```
deepracer_genesis/
  envs/deepracer_env.py     # batched VecEnv-style environment
  envs/track.py             # track registry + GPU waypoint geometry
  randomization/domain_rand.py
  configs/cfgs.py           # env cfg + rsl-rl 5.x train cfg
  train.py / eval.py
  validation/camera_check.py  # paired onboard+topdown images, automated checks
  assets/                   # car URDF/meshes, track DAEs, waypoint routes
benchmarks/throughput.py    # sweep -> results.csv + results.md (final table)
```

## Setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv sync                        # everything (torch, genesis, torchrl, ...)
uv sync --extra tracking --extra hpo   # + mlflow, optuna
```

Requires Linux x86-64 + NVIDIA GPU (BatchRenderer needs CUDA; genesis-world
pulls `gs-madrona` automatically).

### CUDA 13 toolkit note (gs-madrona 0.0.7)

`gs-madrona` bundles `libnvJitLink.so.12` (CUDA 12.4) but prefers the system's
`libnvrtc.so.13` at runtime. With a CUDA 13 system toolkit the NVRTC-13 LTO
output can't be linked by nvJitLink-12 and scene build dies with
`nvJitLink error: Internal error`. Fix used here (see `scripts/fix_madrona_cuda13.sh`):

1. `uv pip install nvidia-cuda-nvrtc-cu12==12.4.127`
2. symlink `libnvrtc.so.12`, `libnvrtc.so`, `libnvrtc-builtins.so.12.4` from
   `site-packages/nvidia/cuda_nvrtc/lib/` into `site-packages/gs_madrona/`
   (madrona's `$ORIGIN` RUNPATH picks them up)
3. binary-patch the dlopen name `libnvrtc.so.13` -> `libnvrtc.so.12` in
   `site-packages/gs_madrona/libmadgs_mgr.so` (same byte length)

## Google Colab

[`notebooks/deepracer_genesis_colab.ipynb`](notebooks/deepracer_genesis_colab.ipynb)
runs the framework on a Colab GPU runtime (T4): uv-installs this repo,
defines a single-file experiment in a cell, trains it, renders the
many-agents spectator video inline, and saves the run directory to Google
Drive. Point the `REPO` variable in the install cell at your fork.

## Experiment framework (TorchRL, config-as-code)

`deepracer_genesis/experiment/` implements EXPERIMENT_PLAN.md: experiments are
Python functions/classes composing stages with `>>` into a content-hashed
`ExperimentSpec`; a `Builder` turns specs into TorchRL objects (Collector,
ClipPPOLoss, GAE — PPO-Lagrangian with a PID-controlled lambda for SafeRL*
envs); the `Trainer` writes checkpoints + an `EvalRecord` per run under
`runs/{group}/{variant}-{seed}-{id}/`; re-running an identical config is a
cache hit.

### Running experiments

An experiment is one file, one class — no command line needed: training
config as class attributes, the env / DR / policy pipeline as a `>>` chain,
`run(TheClass)` in `__main__`. Copy `experiments/template.py`:

```python
from deepracer_genesis.experiment import (CameraEnvironment, Experiment,
                                          DomainRandomizationCamera,
                                          AsymmetricCameraPolicy, run)

class MyExperiment(Experiment):
    seed = 0
    total_env_steps = 10_000_000
    eval_every_steps = 1_000_000          # deterministic eval every 1M steps
    num_envs = 128                        # your own hyperparameters, any name

    def pipeline(self):
        return (CameraEnvironment(render="madrona", num_envs=self.num_envs,
                                  resolution=(160, 120))
                >> DomainRandomizationCamera(brightness=(0.7, 1.3))
                >> AsymmetricCameraPolicy(actor_keys=("camera",),
                                          critic_keys=("camera", "state")))

if __name__ == "__main__":
    run(MyExperiment)                     # uv run experiments/my_experiment.py
```

Variants are subclasses (`class NoDR(MyExperiment): ...`) — each gets its own
content-hashed run dir; re-running an identical config is a cache hit.

The same experiments run from the CLI or from Python:

```bash
python -m deepracer_genesis.experiment --list                 # registered names
python -m deepracer_genesis.experiment feature_baseline --seed 3 --eval-every 1000000
python -m deepracer_genesis.experiment MyExperiment --set num_envs=64
python -m deepracer_genesis.experiment feature_baseline --video --track reInvent2019_track
python -m deepracer_genesis.experiment --report                # runs/report.md
```

```python
import experiments                             # registrations fire
from deepracer_genesis.experiment import run
run("feature_baseline")                        # 5M steps in ~90 s on a 4060 Ti
run("cam_baseline", seed=3)                    # Env 1: camera+asym+full DR

from deepracer_genesis.experiment.ablation import sweep, seeds
for spec in seeds(sweep(run("safe_feature", build_only=True),
                        "env.cost_budget", [10, 25, 50]), k=3):
    run(spec)
from deepracer_genesis.experiment.report import build_report
build_report("runs")                           # report.md + report.csv
```

Multi-track training: pass `tracks=(...)` to a feature env stage — Genesis
builds a heterogeneous morph per track and each parallel env simulates its
own geometry. (Camera multi-track is rejected: the batch renderer has no
per-env variant visibility in genesis 1.2.1, so all tracks would render
superimposed.)

Spawns are randomized (`random_start=True`, plus lateral/yaw noise under DR);
a lap is measured as cumulative progress from the spawn point, so the finish
line of a lap is exactly the (random) start location. Adding
`random_direction=True` to an env stage also coin-flips the driving direction
(clockwise vs counter-clockwise) each episode — heading, progress and
lookahead observations all follow the chosen direction.

### Domain randomization: what varies, when, and what can't (yet)

**Per step — effectively unlimited.** Anything expressible as a tensor op on
observations or actions runs at full speed with fresh draws every step:

- `DomainRandomizationCamera` → image aug on the rendered obs: brightness,
  contrast, saturation, hue rotation, gaussian blur, cutout patches, pixel
  noise (per env, per step).
- `DomainRandomizationActions` → gaussian steer/speed noise on the action
  path (per env, per step).

**Per episode — resampled at every reset, env-side.** These change the world
each episode without touching the compiled scene:

- `DomainRandomizationPhysics` → friction, mass shift, center-of-mass shift,
  steering kp scale, wheel kv scale, armature (per env, batched via Genesis's
  `batch_dofs_info`/`batch_links_info`).
- Camera mount jitter (pitch/position of the onboard camera).
- Spawn: random waypoint + lateral/yaw noise (`random_start`), coin-flip
  driving direction (`random_direction`).
- `DomainRandomizationTrackAppearance` → world-color remap of the rendered
  obs (hue rotation + saturation/value + channel mix + bias; invertible so
  the task stays readable). Cost @128 camera envs: 16k → 12k steps/s.

**Fixed for the whole run — the scene compiles once.** Changing these means
a new process/run (cheap: content-hashed runs make sweeps over them easy):

- Track geometry and scene textures/meshes.
- Lighting (one sun per scene: direction/intensity — **no per-env lighting**
  in Genesis 1.2).
- Camera FOV/resolution, number of envs.
- Action DELAY depth: `delay_steps=k` is a constant latency (the ring buffer
  is fixed size); the noise on top varies per step, but per-episode-random
  latency would need a variable-depth read (easy extension, not built).

**Hard renderer limits (the brick wall).** Genesis 1.2.1 never passes
per-env variant visibility (`vgeom.active_envs_mask`) to the Madrona batch
renderer — only the (slow, ~600 steps/s) rasterizer path honors it. Until
that lands upstream, under Madrona there is no per-env: scene textures,
track meshes, or multi-track camera training (all variants render
superimposed and z-fight). Nyx additionally refuses heterogeneous morphs
outright and reads only OBJ. `randomization/appearance.py` still bakes
texture-variant meshes for rasterizer-based use, and validation rejects the
unsound combinations with an explanation.

### Hyperparameter optimization

`experiments/hpo_optuna.py` is a working single-GPU Optuna study: each trial
trains in a fresh subprocess (Genesis builds one scene per process), the
trainer's periodic deterministic evals (`eval_every_steps`) stream back as the
optimization signal, and Hyperband prunes bad trials mid-training. The study
is resumable (`sqlite:///runs/hpo/study.db`), and the content-hash identity
cache makes duplicate configs free.

### Tracks

17 tracks ship registered (3 original DAE + 14 generated); any of the 126
official DeepRacer routes is one call away, and custom tracks are drawn in a
notebook:

```python
from deepracer_genesis.tools.track_builder import fetch_official_track, build_route, install_track
fetch_official_track("penbay_pro")          # any name from deepracer-race-data
route = build_route([(0,0), (6,0), (8,2), (6,4), (2,4)], half_width=0.53)
install_track("my_track", route)            # -> tracks=("my_track",) anywhere
```

Generated tracks get a procedural road mesh (asphalt, border lines, dashed
centerline) that renders identically under Madrona/Nyx/rasterizer. The
interactive flow lives in `notebooks/track_designer.ipynb`: sketch a polygon,
preview, install, sanity-drive with the built-in controller, train.

### Observability

TensorBoard always (event file per run dir). MLflow when
`MLFLOW_TRACKING_URI` is set (e.g. `sqlite:////path/mlflow.db`): spec params,
per-iteration training metrics, periodic + final eval metrics, spec/record
artifacts. Both log per collector iteration (~200 per 5M-step run), never per
env-step — the overhead is unmeasurable. `runs/report.md` aggregation and the
identity cache work regardless.

### Visual verification & data collection

```python
from deepracer_genesis.experiment.visualize import rollout_video, dr_preview_video
rollout_video("feature_baseline")                       # bird's-eye mp4, trained policy
rollout_video("feature_baseline", track="reInvent2019_track")   # same policy, new track
dr_preview_video("cam_baseline")                        # raw|augmented onboard + random-spawn view
# cam_baseline includes appearance DR: the preview shows each car in its own world

from deepracer_genesis.experiment.data_collection import collect_camera_dataset
collect_camera_dataset(track="reinvent_base", out="data/reinvent")  # .npz shards:
# image (B,H,W,3) uint8, state (B,28), pose (B,4) — teleport sweep over
# (waypoint x lateral x yaw) grid, optional ImageAug
```

### Custom algorithms (SAC, world models, ...)

`deepracer_genesis/experiment/algorithms.py` defines the `Algorithm` protocol
the Trainer drives: implement `setup(builder)`, `collect_policy`,
`eval_actor`, `train_on_batch(data)`, `observe_env_logs(logs)`,
`checkpoint()`, register with `@register_algorithm("my_kind")`, and select it
from the DSL with `... >> Algo(kind="my_kind", params={...})`. PPO and
PPO-Lagrangian are themselves implementations of the protocol — see the
module docstring for the step-by-step guide.

## Usage

```bash
# state-based teacher (fast policy search)
python -m deepracer_genesis.train -B 4096 --max_iterations 500 --exp_name teacher

# heterogeneous parallel envs: pass a list of morphs per track — each env
# simulates + renders a different track (Genesis balanced block assignment)
python -m deepracer_genesis.validation.camera_check --num_envs 6 \
    --tracks reinvent_base,reInvent2019_track,2022_reinvent_champ

# vision policy (CNN on 160x120 RGB), with domain randomization
python -m deepracer_genesis.train -B 256 --vision --randomize --max_iterations 1000 --exp_name vision

# camera validation: paired onboard/topdown snapshots + videos + automated checks
python -m deepracer_genesis.validation.camera_check --num_envs 4
python -m deepracer_genesis.validation.camera_check --checkpoint logs/vision/model_1000.pt

# eval a checkpoint: records a high-res "spectator" video (bird's-eye,
# ALL agents on the track at once, true colors) + onboard video (vision envs)
python -m deepracer_genesis.eval --checkpoint logs/teacher/model_500.pt --num_envs 24 --res 1280x960

# throughput sweep -> benchmarks/results.md (max steps/s per agent x n_agents)
python benchmarks/throughput.py --sweep
```

Training is fully headless; nothing needs a display. The vision pipeline is
validated by `camera_check.py`, which saves paired images from the onboard
camera and a top-down camera above the track and runs four automated checks
(non-degenerate frames, temporal change, per-env difference, cross-view
position consistency).

## Notes / known quirks

- The processed URDF (`assets/urdf/deepracer/deepracer_processed.urdf`) was
  generated from the original xacro with local mesh paths; the body-shell
  collision mesh was removed (its convex hull touched the ground and beached
  the car — wheels carry all contact now).
- Steering hinges need heavy velocity damping (`steer_kv=5`) — low damping
  causes front-wheel shimmy that destabilizes the whole car.
- Drive torque is clamped (`wheel_max_torque`) near the traction limit;
  unbounded torque with a P velocity controller causes wheel-slip limit cycles.
- The Madrona BatchRenderer renders some alpha-textured DAE ground materials
  fully transparent (background bleed-through). `reinvent_base` ships with the
  field submesh stripped and a Genesis-surface-colored overlay instead
  (also a handy hook for per-track visual DR).
- Per-env lighting and per-env actuator gains are not supported by Genesis
  1.2; lighting is global at build time, gains are jittered globally per reset.
- Heterogeneous tracks: `scene.add_entity([morph_a, morph_b, ...])` gives each
  env one geometry variant (contiguous blocks, `_balanced_variant_mapping`).
  `MeshSet` is unrelated (soft-body mesh collections) — the plan's original
  assumption was wrong for genesis 1.2.0.
- On the reInvent2019 track, cars under the start-gate bridge are occluded
  from the top-down camera; the cross-view validation check tolerates
  legitimate occlusion (visible cars must project within 8 px).
- The BatchRenderer requires all cameras to share one resolution. The
  "spectator" camera escapes this via `add_camera(debug=True)`: it renders
  through the rasterizer at any resolution, with true texture colors, and
  shows every env's car in a single image — used for high-res demo videos.
- Madrona renders the alpha-cutout centerline texture with R and G swapped
  (dashes look yellow-green onboard instead of orange). Asset-level fixes
  don't take; the rasterizer path (spectator, cpu-vision branch) is correct.
  Consistent for training, cosmetic otherwise.
- Branch `cpu-vision`: visual-only training on the CPU backend (per-env
  rasterizer cameras instead of Madrona). Slow; for correctness checks.
