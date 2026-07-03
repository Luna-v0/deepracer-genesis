# Technical Porting Plan: DeepRacer RL Environment → Genesis (Vision-Based, RSL-RL, Domain-Randomized, Heterogeneous Parallel)

> **STATUS (2026-07-03): implemented.** See `README.md` for usage and the
> deviations found during implementation (rsl-rl 5.4.1 interface, heterogeneous
> tracks via list-of-morphs instead of `MeshSet`, gs-madrona CUDA-13 patch,
> physics tuning). Camera validation: `logs/validation*/` (all checks pass).
> Throughput table: `benchmarks/results.md`.

## TL;DR
- **The port is feasible and well-suited to Genesis.** Rebuild the DeepRacer car (reusing its URDF and track meshes from `seresheim/deepracer-env`) as a native, ROS-free, GPU-batched `VecEnv`-style class driven by RSL-RL's PPO, closely mirroring Genesis's shipped `go2` locomotion example but with a single front RGB camera as the observation. This gives orders-of-magnitude more throughput than the Gazebo original, which is single-world and CPU-bound.
- **The single hard constraint is vision.** Physics scales to tens of thousands of parallel envs, but per-env camera rendering is the true bottleneck; you MUST use the Madrona-based `BatchRenderer` (Linux + CUDA only), keep resolution low (DeepRacer-native 160×120), and expect throughput to be dominated by rendering + CNN policy updates, not physics.
- **Heterogeneous tracks/colors per parallel env are partially supported and the riskiest requirement.** Genesis has shipped heterogeneous rigid + articulated simulation and batched textures; per-env color/texture and per-env geometry are achievable, but per-env *lighting* is on the roadmap and not confirmed shipped. Recommended: vary track geometry via a small number of scene "variant groups" (or the newer `MeshSet` heterogeneous morph, verified in source), vary colors/textures per env (supported), and treat per-env lighting as best-effort.
- **End goals (deliverables):**
  1. A working, headless-trained vision policy whose **camera input is explicitly validated** by saving paired images from (a) the car's onboard camera feed and (b) a top-down/bird's-eye validation camera above the track — training never requires a human-visible viewer; validation only has to prove the camera pipeline works.
  2. A **final throughput table** reporting, for each configuration swept, the max steps/s per agent, the number of agents running in parallel, and the aggregate steps/s (per-agent × n_agents).

## Key Findings

### Genesis status and the "major update"
- Genesis was rebranded/relaunched as **Genesis World 1.0** in a **May 2026** blog post by the Genesis AI Team ("The Role of Simulation in Scalable Robotics, Genesis World 1.0, and the Path Forward"), now officially supported by Genesis AI (previously an academic project since Dec 2024). The PyPI/import name remains `genesis-world` / `import genesis as gs`. The stack now comprises a unified multi-physics engine, a new in-house photorealistic renderer **Nyx**, and a cross-platform compiler **Quadrants** (a fork of Taichi from June 2025) that lowers Python kernels to CUDA, ROCm, Metal, Vulkan, x86, and ARM64.
- The blog defines the user-facing **Simulation Interface** layer as including "asset parsing (URDF, MJCF, OBJ, GLB, USD, …), entity accessors, controllers, sensors, **parallel and heterogeneous environments**, and a built-in GUI."
- Rendering paths are exposed as camera sensors: **Nyx** (in-house, robotics-focused), **Luisa** (DSL ray tracer), **Pyrender** (rasterizer), plus the **Madrona batch renderer** (`gs-madrona`) for high-throughput multi-env rendering.
- The core simulation API (scene, morphs, entities, `scene.build(n_envs=...)`, batched tensor control) is stable across the 0.3.x → 0.4.x → 1.0 line. Note that documentation is versioned inconsistently — pages labeled 0.3.x, 0.4.x, and 1.0.0 all coexist under `/en/latest/`. Pin your version and verify each call.

### Core Genesis API (current)
- **Init:** `gs.init(backend=gs.cuda)` (also `gs.cpu`, `gs.amdgpu`, `gs.metal`). Use `gs.device` for tensors.
- **Scene:** `gs.Scene(sim_options=gs.options.SimOptions(dt=..., gravity=...), rigid_options=gs.options.RigidOptions(...), viewer_options=..., vis_options=gs.options.VisOptions(...), renderer=..., show_viewer=False)`.
- **Entities via morphs:** `scene.add_entity(gs.morphs.Plane())`, `gs.morphs.URDF(file=...)`, `gs.morphs.MJCF(file=...)`, `gs.morphs.Mesh(file=...)`, `gs.morphs.MeshSet(...)` (a collection of meshes — the vehicle for heterogeneous per-env geometry), with `pos`, `euler`/`quat`, `scale`, and a `surface=gs.surfaces.*` for appearance.
- **Build for parallelism:** `scene.build(n_envs=B, env_spacing=(x,y), n_envs_per_row=..., center_envs_at_origin=True)`. When `n_envs>0`, the first dimension of all state tensors is the batch dimension B. `env_spacing` is visualization-only and does not affect physics poses.
- **Batched control:** the same control APIs take a leading batch dim, e.g. `car.control_dofs_velocity(torch.zeros(B, n_dofs, device=gs.device))`; a subset is addressable with `envs_idx=torch.tensor([...])`.
- **Reset:** `scene.reset(envs_idx=...)`; RL envs additionally maintain their own `reset_idx(env_ids)` to re-pose entities and zero buffers.
- Genesis JIT-compiles GPU kernels on the first `build()`; changing structural sizes triggers recompilation.

### RSL-RL integration (the shipped pattern)
- Genesis ships locomotion (`examples/locomotion/go2_env.py`, `go2_train.py`, `go2_eval.py`) and drone (`hover_env.py`) RL examples that use RSL-RL's PPO. Install with `pip install rsl-rl-lib`. **As of July 2026 the latest PyPI release is `rsl-rl-lib` 5.4.0** (v2.2.4 dates to March 2025 and is what several Genesis examples were historically pinned against). The library supports **PPO and Student–Teacher Distillation** (per Schwarke et al., "RSL-RL: A Learning Library for Robotics Research," arXiv:2509.10771) and is used by Isaac Lab, Legged Gym, MuJoCo Playground, and mjlab. **Pin explicitly and validate the interface against your installed version.**
- The environment is a **plain Python class** (not a `gymnasium.Env`) exposing the RSL-RL `VecEnv` contract. From the shipped `Go2Env`:
  - Attributes: `self.num_envs`, `self.num_obs`, `self.num_privileged_obs` (or `None`), `self.num_actions`, `self.max_episode_length`, `self.device`, `self.dt`.
  - `reset()` → returns `(obs, extras)` (older examples returned `obs, None`).
  - `step(actions)` → returns `(obs, rewards, dones, infos)`. Current RSL-RL versions structure observations as a `TensorDict` and expect an `extras["observations"]` group mapping and `extras["time_outs"]` for timeout bootstrapping; older Genesis examples return a flat obs tensor. **Match the exact tuple arity to your rsl-rl-lib version** — a known Genesis bug was `ValueError: too many values to unpack` from `get_observations()` when versions mismatched.
  - `get_observations()` → `(obs, extras)`.
  - `episode_length_buf` must be settable (RSL-RL randomizes initial episode lengths via `init_at_random_ep_len` to de-correlate simultaneous resets in large batches).
- Training wiring:
  ```python
  from rsl_rl.runners import OnPolicyRunner
  runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
  runner.learn(num_learning_iterations=max_iterations)
  ```
  `train_cfg` is a nested dict with `algorithm` (PPO: clip_param 0.2, desired_kl 0.01, entropy_coef, gamma 0.99, lam 0.95, learning_rate 3e-4, num_learning_epochs, num_mini_batches, schedule "adaptive"), `policy` (activation, actor/critic hidden dims, init_noise_std), and `runner` blocks.
- RSL-RL is GPU-only. The **teacher–student / two-stage** pattern is directly relevant: Genesis's own manipulation example trains a privileged-state teacher with PPO on large batches (1024–4096 envs), then distills into a vision (CNN) student rendered with Madrona — an efficient recipe for vision-based DeepRacer that sidesteps the render bottleneck during policy search.

### Source environment (`seresheim/deepracer-env`)
- It is a fork of `aws-deepracer-community/deepracer-simapp`, a Gymnasium environment powered by **ROS Noetic + Gazebo 11**, runnable only inside a ROS container — exactly what we are removing.
- Reusable assets live in `simulation/` (`meshes/`, `worlds/`, `urdf/`, plus `track_geom/` waypoint utilities). We reuse the **URDF** for the car and **track meshes/waypoints**; we discard the ROS colcon workspace, `agent_ctrl/RolloutCtrl` (ROS action publisher), and the Docker/ROS build chain.
- Documented interfaces to preserve semantics:
  - **Action space:** `Box([-30, 0.1], [30, 4.0])` = `[steering_angle_deg, speed_m_s]`.
  - **Camera observation:** `CAMERA`/`OBSERVATION` of shape `(120, 160, 3)` uint8 — a single front RGB camera, matching real DeepRacer.
  - **Reward params:** `all_wheels_on_track`, `progress` (0–100%), `speed`, `steering_angle`, `distance_from_center`, `is_left_of_center`, `waypoints`, `closest_waypoints`.

### Cameras, rendering backends, and the performance bottleneck
- Cameras: `cam = scene.add_camera(res=(160,120), pos=..., lookat=..., fov=..., GUI=False)`. Camera models `pinhole` and `thinlens`. A camera can be **mounted to a moving link** via `cam.attach(rigid_link, offset_T)` + `cam.move_to_attach()`, or made to follow an entity with `cam.follow_entity(entity, offset=..., fixed_axis=..., smoothing=..., fix_orientation=...)`. For a car, attach the camera to the chassis link with a forward-facing offset transform.
- Backends:
  - **Rasterizer** (`gs.renderers.Rasterizer()`): default; the ray tracer is single-env; the rasterizer is historically single-camera-per-call.
  - **BatchRenderer** (`gs.renderers.BatchRenderer(use_rasterizer=True/False)`): Madrona-based, high-throughput, renders all envs in one pass. Returns tensors shaped `(n_envs, H, W, 3)`; `scene.render_all_cameras(...)` returns `(n_cameras, n_envs, H, W, 3)`. **Requires Linux x86-64, NVIDIA CUDA (12.4+), Python ≥3.10**; not available on Windows/Mac. Install `gs-madrona` from PyPI (prebuilt wheels) — some users hit a Vulkan loader bug requiring a source build.
- **Performance reality:** Genesis's headline numbers are real but not representative of a vision-car workload. The README/benchmarks report "over 43 million FPS when simulating a Franka robotic arm with a single RTX 4090 (430,000 times faster than real-time)"; independent analysis (Stone Tao) confirms "43M+ FPS with just the robot arm and self-collisions enabled, and 27M FPS when random actions are added" — but the **same analysis found only ~0.29M FPS under a realistic Franka benchmark (~150× below the headline), on par with or slower than existing GPU simulators.** Vision compounds this: reporting notes Genesis's rasterizer camera path is a real bottleneck for vision workloads, batch-renderer cameras must share resolution, and are RGB-only. For calibration, the comparable MuJoCo-Playground + Madrona pipeline (arXiv:2502.08844, single A100) runs "Cartpole and Franka environments … at roughly 403,000 and 37,000 steps per second respectively" with pixels, and finds physics+render+inference are only ~9–43% of wall-clock in a pixel-PPO loop — **the CNN policy update dominates.** Plan benchmarks accordingly.

### Domain randomization support
- **Physics DR (supported, per-env):** `RigidLink.set_mass()` / `get_mass()`, `RigidGeom.set_friction()`, `RigidEntity.set_friction_ratio()`, and DOF setters (`set_dofs_damping`, `set_dofs_stiffness`, `set_dofs_armature`, `set_dofs_frictionloss`, `set_dofs_kp/kv`, `set_dofs_force_range`) all accept `envs_idx` for per-environment values. Genesis added an official **domain randomization example** and `set_friction_ratio` APIs, plus external force/torque APIs (for random "pushes"). Typical sim-to-real ranges: mass ±20%, friction 0.5–1.5×, damping 0.8–1.2×, plus observation/action noise.
- **Visual DR (supported):** appearance is controlled by `surface=gs.surfaces.*` (Default/Plastic/Rough/Smooth/Metal/…, with `color`/texture); the Madrona `BatchRenderer` gained **batched textures** support (PR #2077), enabling per-env textures/colors. Mesh color can be set programmatically (`mesh.set_color(...)`).
- **Lighting DR (partial / not confirmed per-env):** `scene.add_light(...)` exists with a renderer-dependent signature (BatchRenderer: `pos, dir, intensity, directional, castshadow, cutoff`; Rasterizer: unsupported). Fully dynamic per-environment lighting appears to be a roadmap item for `gs-madrona`, not confirmed shipped. The Nyx renderer is pitched for exactly this kind of sweep — "We can sweep variants … including object shapes, surface materials, lighting angles, and camera trajectories, across ~10 axes" — but that is evaluation-side Nyx, not confirmed as per-batch-index lighting in the training BatchRenderer. **Recommendation: do not force per-env lighting**; randomize global lighting per training iteration and lean on texture/color randomization for visual diversity.
- **Camera DR (supported):** randomize camera intrinsics/extrinsics per reset via the camera pose/fov and mount offset, and add pixel/gaussian noise to rendered observations in the env.

### Heterogeneous parallel environments (different tracks/colors/lighting)
- Genesis explicitly advertises **"parallel and heterogeneous environments"** as a first-class feature.
- Shipped capability, from release notes:
  - **Initial heterogeneous rigid objects** + USD stage import (PR #2202; the same release was ~4× faster for complex scenes at n_envs=4096 vs 0.3.10).
  - **Heterogeneous articulated robots** (PRs #2472/#2535).
  - **Batched textures for BatchRenderer** (PR #2077); **render each heterogeneous variant in its own environment** (PR #2958); **raise on ambiguous morph access for heterogeneous entities** (PR #2798); raycast-sensor fixes for heterogeneous entities (#2876/#2891).
  - Mechanism (from source/DeepWiki): a single entity/link holds **multiple geometry variants**, each geom carrying an **`active_envs_mask`** that binds it to specific batch indices; exposed via a heterogeneous morph path (`gs.morphs.MeshSet`) in `scene.add_entity()`. The `batch_fixed_verts` flag on file morphs "will allow setting env-specific poses to fixed geometries, at the cost of significantly increasing memory usage" — this is what lets a fixed track take env-specific poses.
- **Honest limitation:** the exact `MeshSet` constructor signature and a canonical "track A in env 0, track B in env 1" example are **not surfaced in the user tutorials** (documented mainly in source). You will likely need to inspect `genesis/options/morphs.py` (`MeshSet`) and `rigid_entity.py` directly. Per-env **lighting** variation is not confirmed. Treat heterogeneous geometry as supported-but-verify.

### Installation & dependencies
- `pip install genesis-world`; install **PyTorch separately** per the official instructions (CUDA build). Requires **Python ≥3.10** (3.10/3.11 recommended), PyTorch 2.x, NVIDIA driver + CUDA 12.x for GPU. Editable/full install: `git clone …/Genesis.git && cd Genesis && pip install -e ".[dev]"` (or `".[render]"`/`"[all]"` for renderers). `uv` is supported.
- Batch rendering: `pip install gs-madrona` (Linux + CUDA 12.4+ only).
- Headless: Genesis uses EGL offscreen by default; on cloud GPUs either rely on EGL or start `Xvfb :0 -screen 0 1024x768x24 & export DISPLAY=:0`. Misconfigured EGL silently falls back to CPU MESA rendering (severe slowdown) — verify EGL is active.
- RL deps: `pip install tensorboard rsl-rl-lib==<pinned>`.

## Details

### (a) Project architecture and file structure
```
deepracer-genesis/
├── deepracer_genesis/
│   ├── __init__.py
│   ├── envs/
│   │   ├── deepracer_env.py        # DeepRacerEnv: RSL-RL VecEnv-style class (single camera obs)
│   │   └── track_registry.py       # loads track meshes + waypoints (from simulation/ assets)
│   ├── assets/                     # copied from seresheim/deepracer-env/simulation/
│   │   ├── urdf/                   # DeepRacer car URDF (+ meshes it references)
│   │   ├── meshes/                 # car + track meshes
│   │   └── tracks/                 # per-track OBJ/GLB + waypoints (.npy/.csv)
│   ├── rewards/
│   │   └── reward_terms.py         # vectorized reward functions
│   ├── randomization/
│   │   └── domain_rand.py          # physics + visual (+ best-effort lighting) DR
│   ├── configs/
│   │   ├── env_cfg.py              # env_cfg, obs_cfg, reward_cfg, command_cfg
│   │   └── train_cfg.py            # rsl-rl PPO config dict
│   ├── validation/
│   │   └── camera_check.py         # saves paired onboard-camera + top-down images (see section i)
│   ├── train.py                    # builds env + OnPolicyRunner.learn()
│   └── eval.py                     # loads checkpoint, rolls out, records camera video
├── benchmarks/
│   ├── throughput.py               # steps/s vs n_envs, vs Gazebo baseline
│   └── results.md                  # final throughput table (see section j)
├── pyproject.toml
└── README.md
```
Mirror the structure of Genesis's `examples/locomotion` (`*_env.py`, `*_train.py`, `*_eval.py`, cfg dicts pickled to `logs/<exp>/cfgs.pkl`).

### (b) Loading the DeepRacer URDF and tracks
```python
import genesis as gs
gs.init(backend=gs.cuda)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=1/60., gravity=(0,0,-9.81)),
    rigid_options=gs.options.RigidOptions(dt=1/60., constraint_solver=gs.constraint_solver.Newton,
                                          enable_collision=True, enable_joint_limit=True),
    vis_options=gs.options.VisOptions(shadow=False),           # shadows off for speed
    renderer=gs.renderers.BatchRenderer(use_rasterizer=True),  # Madrona batch renderer
    show_viewer=False,
)
plane = scene.add_entity(gs.morphs.Plane())
car = scene.add_entity(
    gs.morphs.URDF(file="deepracer_genesis/assets/urdf/deepracer.urdf",
                   pos=(0,0,0.05), euler=(0,0,0), merge_fixed_links=True),
)
track = scene.add_entity(
    gs.morphs.Mesh(file="deepracer_genesis/assets/tracks/reinvent_base.obj",
                   fixed=True, pos=(0,0,0)),
    surface=gs.surfaces.Rough(color=(0.15,0.15,0.15,1.0)),
)
```
- Convert Gazebo `.world`/COLLADA track assets to a Genesis-loadable mesh (OBJ/GLB/USD). Keep collision meshes convex-decomposed; keep the track a **fixed** rigid entity.
- Reuse `track_geom/` waypoints (load as an `(N,2/3)` tensor per track) for the reward's centerline/progress computation.
- Identify the DeepRacer car's steering joints and drive-wheel DOFs by joint name (`car.get_joint(name).dof_idx_local`); set control gains with `set_dofs_kp/kv` and `set_dofs_force_range`. Watch the multi-DoF joint pitfall (Genesis `dof_idx_local` can be a list) seen in the go2 example.

### (c) The RSL-RL-compatible vectorized env with single-camera obs
```python
class DeepRacerEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg,
                 show_viewer=False, device="cuda"):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.num_actions = 2                     # [steering, speed]
        self.num_obs = obs_cfg["num_obs"]        # flattened image, or use CNN policy on (C,H,W)
        self.num_privileged_obs = obs_cfg.get("num_privileged_obs")  # for teacher/critic
        self.dt = env_cfg["dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)
        # build scene (as in b), add a per-env front camera:
        self.cam = self.scene.add_camera(res=(160,120), fov=90, GUI=False)
        self.scene.build(n_envs=num_envs, env_spacing=(5.0, 5.0))
        self.cam.attach(self.car.get_link("camera_link"), offset_T)   # forward-facing mount

    def step(self, actions):
        self._apply_actions(actions)                      # map -> steering + wheel velocities
        self.scene.step()
        rgb, _, _, _ = self.cam.render(rgb=True)          # (num_envs,120,160,3) via BatchRenderer
        self.obs_buf = self._to_obs(rgb)                  # normalize, permute to (N,C,H,W)
        self._compute_reward(); self._check_termination()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, self.extras

    def get_observations(self):
        return self.obs_buf, self.extras
```
- Use RSL-RL with a **CNN actor-critic** since observations are images; keep the image small (160×120 or downsampled to 84×84).
- Consider the **teacher–student** recipe: train a fast state-based teacher (waypoint-relative pose, speed, heading error) on huge `n_envs`, then distill to the camera-only student — this dodges the render bottleneck during the expensive policy-search phase.

### (d) Domain randomization implementation
```python
def randomize(env, env_ids):
    B = len(env_ids)
    # physics
    for link in env.car.links:
        base = link.get_mass()
        link.set_mass(base * gs_rand_float(0.8, 1.2, (B,), env.device), envs_idx=env_ids)
    env.car.set_friction_ratio(gs_rand_float(0.5, 1.5, (B, env.car.n_geoms), env.device),
                               link_indices=range(env.car.n_geoms), envs_idx=env_ids)
    env.car.set_dofs_damping(base_damp * gs_rand_float(0.8,1.2,(B,env.car.n_dofs),env.device),
                             envs_idx=env_ids)
    # visual: per-env track/car color + texture (BatchRenderer batched textures)
    env.set_track_color(sample_colors(B), envs_idx=env_ids)
    # camera: jitter mount offset + fov + additive pixel noise (applied in _to_obs)
```
- Apply DR at `reset_idx` (correlated per-episode) and optionally add small per-step observation/action noise (uncorrelated). Use a **curriculum**: start narrow, widen ranges as reward stabilizes (ADR-style), and retune PPO hyperparameters after adding randomization.
- Lighting: randomize the global light(s) per iteration via `scene.add_light`/light params; note per-env lighting is not confirmed — do not block on it.

### (e) Heterogeneous parallel envs (different tracks/colors/lighting)
Recommended, lowest-risk approach given current support:
1. **Colors/textures per env (safe):** fully supported via batched textures + per-env `surface`/color set with `envs_idx`. Randomize track surface, walls, and car livery per environment index.
2. **Different track geometry per env:** two viable paths —
   - **(Preferred, verify) Heterogeneous morph:** declare the track via `gs.morphs.MeshSet` (the documented "collection of meshes") so different batch indices instantiate different track meshes; Genesis binds variants via per-geom `active_envs_mask` and (per PR #2958) renders each variant in its own env. Enable `batch_fixed_verts=True` for env-specific static-track poses. **Confirm the exact `MeshSet` argument names in `genesis/options/morphs.py` before committing.**
   - **(Fallback, robust) Variant groups:** partition `n_envs` into G groups, build G scenes/processes each with one track (all envs in a group share geometry), and aggregate throughput. This trades heterogeneity granularity for guaranteed correctness and is the safe benchmark configuration.
3. **Lighting per env:** best-effort only (see caveat).

### (f) Reward function design
Port DeepRacer's semantics to a vectorized, per-env reward computed from waypoints + car pose. A sensible dense design (all tensors shape `(num_envs,)`):
```
progress_reward    = k_p * d_progress_along_centerline            # forward progress this step
center_reward      = exp(-(distance_from_center / (0.5*track_width))**2)   # stay centered
speed_reward       = k_v * clip(forward_speed, 0, v_max)          # go fast
heading_penalty    = -k_h * |yaw - track_direction|              # align to track tangent
steer_penalty      = -k_s * |steering_angle|                     # smoothness (anti-zigzag)
action_rate_penalty= -k_a * ||a_t - a_{t-1}||^2
off_track_penalty  = -K_off  where not all_wheels_on_track (also triggers done)
lap_bonus          = +K_lap  when progress crosses 100%
reward = progress_reward + center_reward*speed_reward + heading + steer + action_rate + penalties
```
This follows AWS's canonical guidance: reward proximity to center, reward speed, penalize large steering, big penalty/terminate off-track. A well-known compact AWS baseline is `reward = 1 - (distance_from_center/(track_width/2))**4`, gated by `all_wheels_on_track`, with a speed multiplier and a large `progress==100` bonus — good for sanity-checking. Termination: off-track (all wheels off), excessive tilt/flip, or `episode_length` exceeded (timeout → `extras["time_outs"]` for correct bootstrapping).

### (g) Performance optimization for benchmarking
- **Backend & headless:** `gs.init(backend=gs.cuda)`, `show_viewer=False`, verify EGL (avoid MESA CPU fallback).
- **Render path:** use `BatchRenderer`; keep camera at native 160×120 (or 84×84); RGB only; `shadow=False`; disable segmentation/depth/normal unless needed.
- **Batch sizing:** physics scales to thousands of envs; with per-env cameras, VRAM and render time cap you far lower. Benchmark a sweep (e.g., 64/256/1024/4096) and report steps/s at each; expect the sweet spot where render+CNN saturate the GPU. For reference, a 52-DoF robot at n_envs=1024 uses ~6–8 GB physics VRAM; per-env images add substantially more.
- **Keep tensors on GPU:** use `gs.device` torch tensors for actions/obs to avoid CPU↔GPU transfer; use the `Newton` constraint solver; tune `dt`/substeps to the largest stable step (DeepRacer control ~15–60 Hz).
- **Two-stage training** to dodge the render bottleneck during search (teacher on state, student on pixels).
- **Benchmark vs Gazebo:** the original is single-world, CPU-bound, ROS-mediated (RTF ~1–2×). Report (i) pure physics steps/s, (ii) steps/s with single-camera obs, (iii) wall-clock to a fixed reward, and (iv) $/1M env-steps. Genesis's advantage is entirely in parallelism, so present per-env and aggregate numbers separately, and disclose that headline 43M/27M FPS figures are idle/random-action single-plane Franka (RTX 4090) — a realistic Franka benchmark measured ~0.29M FPS — not a vision-car workload.

### (h) Training loop and configuration
```python
# train.py
gs.init(backend=gs.cuda)
env = DeepRacerEnv(num_envs=args.num_envs, env_cfg=..., obs_cfg=..., reward_cfg=..., command_cfg=...)
runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
runner.learn(num_learning_iterations=args.max_iterations)
```
`train_cfg` (RSL-RL PPO), starting point adapted from Genesis examples:
```python
train_cfg = {
  "algorithm": {"clip_param":0.2,"desired_kl":0.01,"entropy_coef":0.004,"gamma":0.99,
                "lam":0.95,"learning_rate":3e-4,"max_grad_norm":1.0,"num_learning_epochs":5,
                "num_mini_batches":4,"schedule":"adaptive","use_clipped_value_loss":True,
                "value_loss_coef":1.0},
  "policy": {"activation":"elu","actor_hidden_dims":[512,256,128],
             "critic_hidden_dims":[512,256,128],"init_noise_std":1.0},   # + CNN encoder for pixels
  "runner": {"algorithm_class_name":"PPO","num_steps_per_env":24,
             "max_iterations":args.max_iterations,"save_interval":50,"experiment_name":"deepracer"},
}
```
Launch parallel envs with `-B/--num_envs`; monitor with TensorBoard; checkpoint to `logs/<exp>/model_<i>.pt`; evaluate with `eval.py` (small num_envs, `show_viewer` or record camera video with `cam.start_recording()/stop_recording()`). Note the known Genesis bug where `show_viewer=True` during eval can raise "Buffer model does not exist" — run eval headless and record instead if you hit it.

### (i) End goal 1 — camera-feed validation (headless, image-based)
Training runs **fully headless** (`show_viewer=False`) — a human-visible viewer is never required. Instead, the camera pipeline is validated by saving images from two independent viewpoints and comparing them:

- **Two cameras per validation run:**
  1. The **onboard front camera** (the actual policy observation, 160×120 RGB, attached to the chassis `camera_link`).
  2. A **top-down validation camera** placed above the track (e.g. `scene.add_camera(res=(640,480), pos=(cx, cy, h), lookat=(cx, cy, 0), fov=...)`, or `cam.follow_entity(car, offset=(0,0,3), fix_orientation=True)` for a bird's-eye chase view). This camera exists **only in validation/eval**, never in the training scene, so it costs nothing during training.
- **`validation/camera_check.py` procedure:**
  1. Build a small env (`n_envs` = 4–16), reset, and drive with random or scripted actions for N steps.
  2. Every K steps, save paired snapshots per env: `logs/validation/env{i}_step{t}_onboard.png` and `env{i}_step{t}_topdown.png` (convert the returned `(n_envs, H, W, 3)` tensors with `torchvision.utils.save_image` or PIL).
  3. Also record short videos of both viewpoints via `cam.start_recording()` / `cam.stop_recording(save_to_filename=...)`.
- **Automated sanity checks** (so validation doesn't depend on a human eyeballing every image):
  - Onboard frames are **not degenerate**: per-env pixel std > threshold (catches black/uniform frames from a broken EGL/Madrona setup), and frames **change between steps** (mean abs frame-diff > threshold — catches a frozen camera mount).
  - Frames **differ across envs** when domain randomization is on (catches the batch renderer returning env 0's image for all envs).
  - **Cross-view consistency:** the car's pose extracted from sim state matches what the top-down image shows — e.g. project the car's `(x, y)` world position through the top-down camera intrinsics and assert the car's pixels (or a colored marker on its roof) appear at that image location.
  - The onboard camera moves with the car: after a scripted turn, the onboard view's optical flow / frame difference is consistent with the commanded steering direction (a coarse check is fine).
- **Acceptance criterion:** the run produces paired onboard + top-down images/videos, all automated checks pass, and a human spot-check of a handful of pairs confirms the onboard feed shows the track from the car's perspective while the top-down view shows the same car at the same location. This is the definition of "the vision pipeline works."
- Run this validation **at three points**: (1) right after the camera is first wired up (random actions), (2) mid-training on a checkpoint, (3) on the final policy (this doubles as the qualitative demo — top-down video of the trained car lapping the track alongside its onboard feed).

### (j) End goal 2 — final throughput table (max steps/s per agent × parallel agents)
`benchmarks/throughput.py` must sweep `n_envs` and emit a machine-readable CSV plus a rendered markdown table in `benchmarks/results.md`. **The required final deliverable is this table**, reporting for each configuration the max steps/s per agent, the number of agents running in parallel, and their product (aggregate steps/s):

| Configuration | Obs type | n_agents (parallel envs) | Max steps/s per agent | Aggregate steps/s (per-agent × n_agents) | GPU VRAM | Notes |
|---|---|---|---|---|---|---|
| Physics only | state | 64 / 256 / 1024 / 4096 / … | *measured* | *measured* | *measured* | no rendering |
| Vision 160×120 | RGB camera | 64 / 256 / 1024 / … | *measured* | *measured* | *measured* | BatchRenderer, random actions |
| Vision + PPO update | RGB camera | best n_envs from sweep | *measured* | *measured* | *measured* | full training loop (render + CNN fwd/bwd) |
| Vision + DR + heterogeneous | RGB camera | best n_envs | *measured* | *measured* | *measured* | per-env colors (+ MeshSet or variant groups) |
| Gazebo baseline (reference) | RGB camera | 1 | ~15–60 (RTF 1–2×) | same | n/a | single-world, CPU, ROS |

Measurement rules:
- **Per-agent steps/s** = aggregate env-steps/s ÷ n_envs (report both; the "max steps per agent" is the per-agent rate at the aggregate-optimal n_envs, and also report the single-env n_envs=1 rate as the true per-agent maximum).
- Warm up (JIT compile + first render) before timing; time ≥ 1000 steps; report median of 3 runs.
- Sweep until aggregate steps/s stops improving (the GPU saturation point) and mark the peak row **bold** — that row is the headline number: *max steps/s per agent × number of agents running in parallel*.
- Record hardware (GPU model, driver, CUDA), Genesis/gs-madrona/rsl-rl-lib versions, and resolution in the table header so the numbers are reproducible.

## Recommendations
1. **Stand up the physics-only port first** (car URDF + one track, ROS-free, `n_envs` sweep, state-based observation, port the reward). Benchmark pure steps/s vs Gazebo. *Threshold to proceed: a stable driving policy on one track and ≥100× aggregate step throughput over the Gazebo baseline.*
2. **Add the single front camera via BatchRenderer** and switch to a CNN policy at 160×120. **Immediately run the camera-validation harness (section i)** — paired onboard + top-down snapshots with the automated degenerate-frame/cross-view checks — before spending any GPU-hours on vision training. Re-benchmark with vision. *If vision throughput collapses (to only a few thousand aggregate steps/s), adopt the teacher–student two-stage recipe before scaling.*
3. **Layer in domain randomization** (physics first — mass/friction/damping/push; then visual color/texture; camera noise). Use a curriculum. *Widen ranges only when mean episode reward plateaus; retune PPO if training destabilizes.* Re-run camera validation after visual DR to confirm per-env differences show up in the frames.
4. **Introduce heterogeneity last:** per-env colors/textures (safe) → per-env track geometry via `MeshSet` (verify API) or the variant-group fallback. *Decision point: if `MeshSet` per-env geometry is not confirmed working within ~1 day of testing, ship the variant-group configuration for the benchmark and document it.*
5. **Do not force per-env lighting.** Randomize global lighting per iteration; revisit if a future `gs-madrona` release ships dynamic per-env lights.
6. **Pin versions** (`genesis-world`, `rsl-rl-lib` — currently 5.4.0, `gs-madrona`, PyTorch/CUDA) and validate the `VecEnv` tuple arity against your installed rsl-rl-lib to avoid the known unpack error. Because rsl-rl-lib jumped from 2.x (Genesis-example era) to 5.4.0, budget time to adapt the `reset`/`get_observations`/`step` signatures and the TensorDict observation-group convention.
7. **Close with the two end-goal deliverables:** (a) final camera-validation run on the trained policy — onboard-feed video + top-down video + paired snapshot images, all checks green; (b) the completed throughput table in `benchmarks/results.md` with the peak *max steps/s per agent × n_agents* row highlighted.

## Caveats
- **Version flux:** Genesis's API changes across minor releases and its docs are versioned inconsistently (0.3.x/0.4.x/1.0.0 pages coexist under `/en/latest/`). Verify every API call against the exact version you pin; the "1.0 heterogeneous API" is documented mostly in source, not tutorials.
- **Heterogeneous geometry per env** is shipped but under-documented; the exact `MeshSet`/heterogeneous-morph signature must be confirmed in `genesis/options/morphs.py`. **Per-env lighting variation is not confirmed** (roadmap item for gs-madrona).
- **Vision is the bottleneck, not physics.** Headline FPS numbers (43M/27M) are idle/random-action single-plane Franka on RTX 4090; a realistic Franka benchmark measured ~0.29M FPS (~150× lower). Batch-renderer cameras must share resolution and are RGB-only. Benchmark honestly and report the vision-car configuration explicitly.
- **Batch-renderer cameras share one resolution** — this affects the validation harness too: if the top-down validation camera runs in the same BatchRenderer scene as the 160×120 onboard camera, it is constrained to the same resolution. Workarounds: run validation in a separate small build using the standard Rasterizer/Nyx camera path (fine, since validation needs only a handful of envs), or accept 160×120 for the top-down snapshots.
- **Platform constraints:** the Madrona batch renderer is Linux x86-64 + NVIDIA CUDA 12.4+ only (no Windows/Mac); EGL misconfiguration silently falls back to slow CPU rendering.
- **Asset conversion risk:** Gazebo track/world assets may need manual conversion (COLLADA→OBJ/GLB/USD), convex decomposition for collision, and scale/axis fixes; the car URDF may reference ROS-package mesh paths (`package://…`) that must be rewritten to local paths.
- **RSL-RL interface drift:** older Genesis examples predate current rsl-rl-lib (now 5.4.0) observation-group/TensorDict conventions; expect to adapt `reset`/`get_observations`/`step` return signatures and the teacher/critic (`num_privileged_obs`, "critic" obs group) plumbing.
- Some throughput/setup specifics come from third-party sources (Spheron blog, Medium sensor comparisons, Stone Tao's analysis) rather than primary Genesis docs; treat those numbers as indicative, not guaranteed, and re-measure on your own hardware.
