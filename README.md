# deepracer-genesis

**New here? Start with [TUTORIAL.md](TUTORIAL.md).** The full manual (concepts,
guides, API reference) is the mkdocs site under [`docs/`](docs/) — `uv run mkdocs serve`.

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
- Reproducible by construction: `spec.seed` seeds python/numpy/torch before the
  sim builds (bit-identical same-seed runs on the CPU backend;
  `DR_DETERMINISTIC=1` additionally requests deterministic torch kernels).

## Renderers (one stack, three options)

| Renderer | Use for | Quality | Colors | Vision steps/s (RTX 4060 Ti, 160x120) |
|---|---|---|---|---|
| Madrona batch (`vision_renderer="batch"`, default) | **training camera policies** | rasterized | dash-hue quirk | ~27.5k @ 256 envs |
| Nyx (`vision_renderer="nyx"`, extras `nyx`) | correct-color eval / validation | path traced | correct | ~1.5k @ 256 envs |
| per-env rasterizer (`backend="cpu"`; also the spectator cam) | videos, debugging, GPU-free smoke runs | rasterized | correct | ~90 end-to-end camera training on CPU |

Default recipe: train on Madrona, validate frames on Nyx, record videos with
the rasterizer spectator (`rollout_video` does this automatically). Full
decision guide: [`docs/concepts/renderers.md`](docs/concepts/renderers.md).

Nyx facts: reads OBJ not DAE (tracks ship converted under `assets/tracks/*/obj/`),
requires unmerged URDF links, denoise/AA kept off (temporal history smears
moving objects), driver 575+.

## Layout

```
deepracer_genesis/
  envs/                     # batched VecEnv env, track geometry, signals bus,
                            #   feature sets, renderers, MDP (reward/termination)
  experiment/               # the >> DSL: stages -> content-hashed ExperimentSpec,
                            #   Builder, rsl-rl backend, evaluator, report, visualize
  perception/               # frozen-CNN perception: model, dataset, camera jitter
  datasets/rollout.py       # camera rollout collection -> parquet shards
  deploy/                   # ONNX export (policy + perception) + car bundle
  tools/                    # track builder, track zoo, DR editor, track split
  configs/cfgs.py           # env cfg + rsl-rl 5.x train cfg
  randomization/            # DR catalog, spaces, domain_rand
  validation/camera_check.py
  assets/                   # car URDF/meshes, track meshes, waypoint routes
benchmarks/throughput.py    # sweep -> results.csv + results.md
examples/                   # runnable single-file experiments (camera, feature,
                            #   HPO study, zoos, live viewer)
docs/                       # mkdocs manual + ADRs (docs/decisions/)
```

## Setup

```bash
uv sync                                  # core (feature-vector training runs CPU-only)
uv sync --extra vision --extra nyx       # + GPU renderers
# other extras: hpo (optuna), analysis (pandas+matplotlib), perception,
# export (onnx+onnxruntime; run exports in a genesis-free process)
```

Camera training wants Linux x86-64 + an NVIDIA GPU; feature-vector training
and the CPU rasterizer path run without one. Madrona's device heap is
pre-sized to 1 GiB on init so a shared GPU (e.g. with an LLM resident)
doesn't kill the process; override via `MADRONA_MWGPU_DEVICE_HEAP_SIZE`.
(Historical CUDA-13 / gs-madrona 0.0.7 linking fix: `scripts/fix_madrona_cuda13.sh`.)

## Google Colab

[`notebooks/deepracer_genesis_colab.ipynb`](notebooks/deepracer_genesis_colab.ipynb)
runs the framework on a Colab GPU runtime (T4): uv-installs this repo,
defines a single-file experiment in a cell, trains it, renders the
many-agents spectator video inline, and saves the run directory to Google
Drive. Point the `REPO` variable in the install cell at your fork.

## Experiment framework (config-as-code, rsl-rl backend)

`deepracer_genesis/experiment/`: experiments are Python classes composing
stages with `>>` into a frozen, content-hashed `ExperimentSpec`; `run()`
validates it, builds the Genesis sim, and trains it with rsl-rl's
`OnPolicyRunner`, writing checkpoints + an `eval_record.json` per run under
`runs/{group}/{variant}-{seed}-{id}/`. The content hash names the run dir —
identical configs share one — but **runs always retrain** (there is no result
cache). `resume="path/to/model.pt"` warm-starts from a checkpoint
(weights only, so the spec's own lr/schedule apply).

### Running experiments

An experiment is one file, one class — no command line needed: training
config as class attributes, the env / DR / policy pipeline as a `>>` chain,
`run(TheClass)` in `__main__`. Copy an example from `examples/`:

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
    run(MyExperiment)                     # uv run examples/my_experiment.py
```

Variants are subclasses (`class NoDR(MyExperiment): ...`) — each gets its own
content-hashed run dir. Experiments are referenced **by class**, not by a
name registry; the CLI takes a `module:Class` path:

```bash
python -m deepracer_genesis.experiment examples.camera:CameraMadronaDr --seed 3
python -m deepracer_genesis.experiment examples.camera:CameraMadronaDr --set num_envs=64
python -m deepracer_genesis.experiment --report                # runs/report.md
```

```python
from deepracer_genesis.experiment import build, run
from deepracer_genesis.experiment.overrides import override

base = build(MyExperiment)                     # frozen ExperimentSpec
for seed in range(3):                          # variants are plain comprehensions
    run(override(base, "seed", seed))          # 3 seeds, 3 run dirs
from deepracer_genesis.experiment.report import build_report
build_report("runs")                           # report.md + report.csv
```

Episode knobs on the env stages: `episode_length_s` (time cap, default 30 s),
`max_laps=N` (truncate after N laps — bootstrapped like a timeout, no
penalty), `random_start` / `random_direction` (a lap is cumulative progress
from the spawn, so a lap's finish line is its own random start point). Run
artifacts are bounded by `Evaluation(keep_checkpoints=..., telemetry_envs=...)`
(ADR 0004): finished runs keep `model.pt` + a weights-only best checkpoint
instead of one file per eval.

Multi-track training: pass `tracks=(...)` to an env stage. Feature envs build
a heterogeneous morph per track; camera envs use **spatial tiling** (each
track on its own world tile, works on Madrona, Nyx, and the rasterizer —
render and memory scale with the track count).

### Custom rewards, actions, agents

- **Reward functions are plain callables passed as parameters** (no registry;
  the spec hashes the fn's *name*, not its body). A fn returns named per-step
  terms weighted by `scales` — or the reward itself (a bare `(N,)` tensor);
  `_`-prefixed terms are diagnostic-only TensorBoard channels, and
  `RewardShaping(params={...})` makes constants spec-hashed and searchable
  (ADR 0002). The terminal `crash_penalty` (−10 default) is logged in the
  per-term breakdown and overridable per experiment. Declare what a fn reads
  with `@reads(...)` so the build can verify the critic sees those signals.
  Contract: [`docs/concepts/rewards-actions.md`](docs/concepts/rewards-actions.md).
- **Action space**: continuous Gaussian `[steer, speed]`. Discrete action
  tables are accepted by the DSL (`actions=discrete_grid(...)`) but have no
  trainer on the rsl-rl backend yet — `run()` raises a clear `SpecError`
  (same for PPO-Lagrangian / cost specs since the TorchRL removal).
- **Scripted agents** (`agents/`): `CenterlineFollower` / `NoisyExpert` drive
  collection and previews; subclass `PrivilegedAgent` to script your own
  behavior over the sim's track-frame state.

### Deployment: ONNX + model card

```python
from deepracer_genesis.deploy.onnx import export_policy, export_from_run_dir
export_policy(MyExperiment)            # -> run_dir/export/: policy.onnx,
                                       #    model_card.json, model_metadata.json,
                                       #    <name>.tar.gz car bundle
export_from_run_dir("runs/g/v-0-abc")  # no spec rebuild; also a CLI:
                                       #    python -m deepracer_genesis.deploy.onnx <run_dir>
```

Camera actors export with the `FRONT_FACING_CAMERA` input the car's inference
stack expects; feature actors export as `STATE` with the observation
normalization **baked into the graph**. Output is the raw Gaussian mean —
unbounded, clip to `[-1, 1]` before use. Opset 11 (the car's OpenVINO
ceiling), verified against torch, and refused inside a process that imported
genesis (clashing LLVM symbols). Guide:
[`docs/guides/deployment.md`](docs/guides/deployment.md).

### Perception: a frozen CNN instead of privileged state

`deepracer_genesis.perception`: train a CNN to predict the camera-recoverable
state channels, then drive the sim from it (`CNNPerceptionFeatures`) or add
calibrated noise instead (`NoisyPerceptionFeatures`). `PerceptionCNN` is
parameterized for architecture search and checkpoints carry their own
architecture; `RolloutDataset` serves k-frame stacks from per-track-set
memmap caches, optionally at a chosen `resolution` (decode+resize once — no
PNG decode in the training loop). Collection (`datasets/rollout.py`) writes
single frames per step; stacks are rebuilt at train time, so one dataset
serves every frame-stack depth. See
[`docs/concepts/perception.md`](docs/concepts/perception.md).

### Domain randomization: what varies, when

**Per step** (tensor ops on obs/actions, fresh draws every step):
`DomainRandomizationCamera` (brightness, contrast, saturation, hue, blur,
cutout, pixel noise), `DomainRandomizationActions` (steer/speed noise;
`delay_steps=k` adds constant action latency).

**Per episode** (env-side, no scene rebuild): `DomainRandomizationPhysics`
(friction, mass/COM shift, steering kp, wheel kv, armature — per env, batched),
camera mount jitter, spawn randomization, `DomainRandomizationTrackAppearance`
(per-env world-color remap of the rendered obs).

**Per run** (baked at scene build): track geometry/meshes, lighting (one sun
per scene — no per-env lighting), camera FOV/resolution, env count, and the
Nyx per-env sky tint/exposure (`env_map_tint` / `env_map_multiplier`).

The full knob catalog with costs is
[`docs/reference/dr-catalog.md`](docs/reference/dr-catalog.md); the
interactive DR editor (`tools/dr_editor`) previews any combination.

### Hyperparameter optimization

`examples/hpo.py` is a working single-GPU Optuna study over PPO knobs AND
network architecture (MLP vs recurrent arms as conditional parameters): the
trainer's periodic deterministic evals stream to the pruner via `on_eval`, so
Hyperband kills bad trials mid-training. Because rewards, scales,
`reward_params`, and architectures are all ordinary spec data, they are all
searchable. Guide: [`docs/guides/hpo.md`](docs/guides/hpo.md).

### Tracks

**298 tracks ship registered** (originals + the generated zoo); any official
DeepRacer route is one call away, and custom tracks are drawn in a notebook:

```python
from deepracer_genesis.tools.track_builder import fetch_official_track, build_route, install_track
fetch_official_track("penbay_pro")          # any name from deepracer-race-data
route = build_route([(0,0), (6,0), (8,2), (6,4), (2,4)], half_width=0.53)
install_track("my_track", route)            # -> tracks=("my_track",) anywhere
```

Generated tracks get a procedural road mesh (asphalt, border lines, dashed
centerline) that renders identically under Madrona/Nyx/rasterizer. The
interactive flow lives in `notebooks/track_designer.ipynb`; the zoo tools
(`tools/zoo.py`) compile and preview track sets in bulk, and
`tools/track_split.py` provides the deterministic train/test/holdout split
(`scripts/collect_sim2real.py --split train` collects accordingly).

### Observability

TensorBoard always (event file per run dir): per-iteration training metrics
plus per-term episode reward sums — the breakdown includes the terminal
`crash_penalty`, so the rows add up to the reward the learner actually saw.
Every run writes `eval_record.json` (spec dump, seed, metrics, eval history);
`build_report("runs")` aggregates records into comparison tables. Periodic
eval telemetry records a configurable env subset (`telemetry_envs`) so long
runs don't grow unbounded.

### Custom algorithms

Pass `... >> Algo(cls=MyAlgorithm, params={...})` with any class speaking the
rsl-rl runner interface (`algorithms/protocol.py`; subclassing
`rsl_rl.algorithms.PPO` and overriding `compute_returns`/`update` is the easy
path). The spec validates the interface up front instead of failing inside
the runner.

## Usage (plain CLI, no experiment framework)

```bash
# state-based teacher (fast policy search)
python -m deepracer_genesis.train -B 4096 --max_iterations 500 --exp_name teacher

# vision policy (CNN on 160x120 RGB), with domain randomization
python -m deepracer_genesis.train -B 256 --vision --randomize --max_iterations 1000 --exp_name vision

# camera validation: paired onboard/topdown snapshots + videos + automated checks
python -m deepracer_genesis.validation.camera_check --num_envs 4
python -m deepracer_genesis.validation.camera_check --num_envs 6 \
    --tracks reinvent_base,reInvent2019_track,2022_reinvent_champ

# eval a checkpoint: high-res spectator video (all agents, true colors) + onboard
python -m deepracer_genesis.eval --checkpoint logs/teacher/model_500.pt --num_envs 24 --res 1280x960

# throughput sweep -> benchmarks/results.md
python benchmarks/throughput.py --sweep
```

Training is fully headless; nothing needs a display. The vision pipeline is
validated by `camera_check.py` (non-degenerate frames, temporal change,
per-env difference, cross-view position consistency).

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
  fully transparent; `reinvent_base` ships with the field submesh stripped and
  a surface-colored overlay instead.
- Madrona renders the alpha-cutout centerline texture with R and G swapped
  (dashes look yellow-green onboard). Consistent for training, cosmetic
  otherwise; `madrona_rg_swap` in the vision cfg flips it back.
- Baked track meshes from before 2026-09-11 place centerline dashes slightly
  off the road on coarse-waypoint tracks (tangent-extrapolation bug, fixed in
  the baker); shipped assets are not yet regenerated — rebake a track to pick
  up the fix.
- Per-env lighting is not supported by Genesis (lighting is global at build
  time); per-env *tracks* are supported via spatial tiling.
- The BatchRenderer requires all cameras to share one resolution. The
  "spectator" camera escapes this via the rasterizer (`add_camera(debug=True)`):
  any resolution, true colors, every env's car in one image.
- On the reInvent2019 track, cars under the start-gate bridge are occluded
  from the top-down camera; the cross-view validation check tolerates
  legitimate occlusion (visible cars must project within 8 px).
