# deepracer-genesis (branch: cpu-vision)

**This branch trains from VISION ONLY on the CPU backend** — no CUDA, no
Madrona. `python -m deepracer_genesis.train -B 4 --cpu --max_iterations 100`
runs `gs.init(backend=gs.cpu)`, gives every env its own rasterizer camera
(rendered serially via EGL/OpenGL — the only non-CPU component), and trains a
CNN policy whose actor *and* critic see pixels only (no privileged state).
Expect ~50 steps/s at 4 envs vs ~25,000+ on the CUDA branch — use it for
correctness checks and laptops, not real training runs. Bonus: the rasterizer
renders the original texture colors exactly (orange centerline, blue sky).

---


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
uv pip install torch genesis-world rsl-rl-lib tensorboard pillow
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
runs the whole pipeline on a Colab GPU runtime (T4 works): pip-installs this
repo, applies the gs-madrona NVRTC fix, trains a policy, validates the camera
pipeline, and renders the many-agents spectator video inline. Point the
`REPO` variable in the install cell at your GitHub fork.

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
