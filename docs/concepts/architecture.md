# Architecture: how the Gazebo → Genesis port is wired

This is the "how does it all fit together" map of the repo: how a DeepRacer
step flows, where the car comes from, how control is wired, how tracks and the
scene are built, and where the DeepRacer "rules" (off-track, reward, laps)
actually live. Every claim here is anchored to a `file:line` you can open.

> Mental model in one sentence: the waypoint `.npy` files **are** DeepRacer's
> rulebook — every rule is torch math on those arrays — while the URDF is only
> the car's *body*, the track meshes are *cosmetic*, and the whole thing is
> N-batched so PPO trains hundreds of cars at once.

---

## 1. The two training front-ends over one simulator

The port is a single batched Genesis environment wrapped for two different
training stacks:

```
rsl-rl PPO ─┐
            ├─▶ DeepRacerEnv (Genesis, N cars in parallel) ─▶ scene.step()
TorchRL   ──┘        envs/deepracer_env.py:58
```

- `DeepRacerEnv` (`envs/deepracer_env.py:58`) speaks the **rsl-rl-lib 5.x
  VecEnv contract** natively: there is no external `reset()` — done envs
  respawn *inside* `step()` (`deepracer_env.py:417-420`).
- `TorchRLDeepRacerEnv` (`envs/torchrl_env.py:21`) is a thin `EnvBase` adapter
  over the *same* sim object. It sets `_torchrl_native_autoreset = True`
  (`torchrl_env.py:55`) precisely because the sim already auto-resets.

So TorchRL and rsl-rl are two faces on one simulator, not two simulators.
Everything the sim exposes is `(N, …)` GPU tensors — N parallel cars stepping
together. That batching is the essence of the port: Gazebo ran one car in a ROS
process; here N cars are tensors and the ROS control + domain logic are
reimplemented in torch.

A single control step (`deepracer_env.py:374-431`):

1. Map normalized action `[steer, throttle] ∈ [-1,1]` to physical commands.
2. Push commands to the Genesis DOF controllers.
3. Step physics `decimation` times.
4. `_post_physics` — refresh kinematics + localize on the track.
5. `_compute_reward` → `_check_termination` → respawn done envs.

### 1.1 The wrapper handshake (why you don't "see" them working together)

`TorchRLDeepRacerEnv` does not run *alongside* `DeepRacerEnv` — it **wraps** it,
and only one front-end is live per run:

- **rsl-rl** (`train.py`): `OnPolicyRunner` calls `DeepRacerEnv.step()` directly;
  the wrapper is never imported.
- **TorchRL** (`experiment/trainer.py`): the collector drives
  `TorchRLDeepRacerEnv._step()`, which on `torchrl_env.py:65` calls
  `self.sim.step()` — and `self.sim` **is** the `DeepRacerEnv` (injected in
  `builder.py:151`). That one line is the entire coupling.

Division of labour: `DeepRacerEnv.step` is the *real simulator* (controls →
physics → reward → termination → **auto-reset of done envs**), returning the
rsl-rl VecEnv tuple. `TorchRLDeepRacerEnv._step` is a *translation adapter*: it
calls `sim.step`, then repackages the result into TorchRL's TensorDict.

The subtle part is the **pre-reset snapshot**. Because `step` auto-resets done
envs *inside itself* (overwriting their state), it first stashes `self.step_info`
(`deepracer_env.py:406-413`) — `offtrack`/`flipped`/`time_out`/`terminal_state`
for the step that just happened. The adapter reads `sim.step_info`
(`torchrl_env.py:66-69`) to split `terminated` (crash/off-track → value
bootstrap killed) from `truncated` (timeout → bootstrap kept), and
`_torchrl_native_autoreset = True` (`torchrl_env.py:55`) tells TorchRL not to
issue its own reset. **`step_info` is the whole contract between the two files.**

---

## 2. The car: URDF gives the body, **not** the controller

The car is loaded from `assets/urdf/deepracer/deepracer_processed.urdf`
(`deepracer_env.py:136-144`).

**What the URDF contains** (verified — grep for `pid|gain|dynamics|damping`
returns nothing): 7 links + joints (4 wheel joints `type="continuous"`, 2
steering-hinge joints `continuous` with `lower=-1 upper=1`, camera/body fixed
joints), `<transmission>` blocks declaring `hardware_interface/
EffortJointInterface`, and `<limit effort="10" velocity="100"/>` per joint.

**What the URDF does NOT contain: the control gains.** That is the original
Gazebo actuation *contract* — "these joints are effort-controlled" — but in the
real DeepRacer/Gazebo stack the actual PID gains lived in a **separate
`ros_control` YAML** loaded by `controller_manager`, never in the URDF. That
YAML did not come across in the port.

### 2.1 kp / kv are hand-wired here

Because the gains didn't port, they are re-authored in two places:

- **Defaults** (`configs/cfgs.py:16-19`):
  ```python
  "steer_kp": 25.0,
  "steer_kv": 5.0,   # heavy damping needed: low values cause front-wheel shimmy
  "wheel_kv": 5.0,
  "wheel_max_torque": 3.0,
  ```
- **Applied to Genesis DOFs** (`deepracer_env.py:264-276`):
  - steering DOFs get **both** `set_dofs_kp` + `set_dofs_kv` → PD position control.
  - wheel DOFs get **only** `set_dofs_kv` → pure velocity (P) control.
  - wheels also get `set_dofs_force_range(±wheel_max_torque)` to cap drive
    torque near the traction limit (avoids wheel-slip limit cycles at speed —
    see the comment at `deepracer_env.py:270`).

So the URDF gives the *mechanism*; the *controller feel* (kp/kv, torque cap) is
authored in this repo and tuned by comment-documented trial. Under domain
randomization these gains are re-scaled per-env **once per run**, from the env's
`__init__` (`randomization/physics.py:61-66`; `domain_rand.py` is a back-compat
shim) — not each episode: per-reset genesis setters sporadically crash even on
genesis 1.2.3 (`base_env.py:326-337`), so DR bodies stay fixed for the run.

### 2.2 Wheels and steering are driven separately (and simplified)

In `step()` (`deepracer_env.py:389-395`):

```python
steer = actions[:,0:1] * radians(max_steering_deg)          # one angle
speed = min_speed + (actions[:,1:2]+1)*0.5*(max_speed-min_speed)
wheel_omega = (speed / self.wheel_radius).repeat(1, 4)
self.car.control_dofs_position(steer.repeat(1,2), self.steer_dofs)   # 2 hinges
self.car.control_dofs_velocity(wheel_omega,       self.wheel_dofs)   # 4 wheels
```

Two deliberate simplifications versus the real chassis:

- **Steering** drives both hinges with the *same* angle (`steer.repeat(1,2)`) —
  there is **no Ackermann `<mimic>`** in the URDF, so it is parallel steering.
- **Drive** commands all 4 wheels the *same* omega (`.repeat(1,4)`) — no
  differential, no rear-drive/front-steer split.

DOF indices are resolved by joint name (`deepracer_env.py:262-263`) from the
`WHEEL_DOFS` / `STEER_DOFS` lists (`deepracer_env.py:43-45`). `wheel_radius` is
*measured* from the wheel STL at load time (`deepracer_env.py:278-281`), not
hardcoded.

---

## 3. Tracks: geometry (`.npy`) vs. mesh (visuals) are separate

Each track has **two independent representations**.

### 3.1 The `.npy` route = geometry and the physics of the *rules*

`envs/track.py`. Shape `(W, 6)` = `[center_xy, inner_xy, outer_xy]` per
waypoint, straight from the original AWS simapp routes (`track.py:3-6`). From
this, `Track.__init__` (`track.py:61-82`) precomputes on GPU: centerline,
per-waypoint `half_width` (from inner/outer spread), tangents, left-normals,
`track_yaw`, cumulative arclength `cum_len`, `total_len`, and signed
`curvature`. **All DeepRacer rule logic runs off these arrays, not the mesh.**

`MultiTrack` (`track.py:135`) pads several tracks to a common waypoint count
and gathers per-env by a `variant_idx`, so different parallel envs can run
different tracks (feature mode only — see §4).

### 3.2 The mesh = what you *see*

Two flavors:

- **Original tracks** (`reinvent_base`, `reInvent2019_track`,
  `2022_reinvent_champ`, registered at `track.py:19-26`) are real DAE/OBJ
  meshes with `textures/` folders — those PNGs (`road.png`, `centerLine.png`,
  `field.png`, `wall.png`, `background.png`, `startline.png`) are the "sprites".
- **Generated tracks** (`assets/tracks/generated/*`) are built *procedurally
  from the route* by `tools/track_builder.py`. Only a handful of Gazebo meshes
  ever existed, so the builder generates a road mesh from the waypoints for
  every official route. These are plain **OBJ + MTL with tiny solid-color
  PNGs** (`track_builder.py` `_PALETTE`) — deliberately textureless-ish to
  dodge Madrona texture bugs. New folders with `route.npy` + `track.obj`
  auto-register on import (`track.py:31-36`).

### 3.3 How the mesh enters the scene

`deepracer_env.py:155-160`:

```python
track_morphs = [gs.morphs.Mesh(file=p, fixed=True, collision=False) for p in mesh_paths]
self.track_entity = self.scene.add_entity(track_morphs if len>1 else track_morphs[0])
```

Two things to internalize:

- `collision=False` → **the track mesh is purely visual.** There are no
  physical walls. The car is physically supported only by the ground plane;
  "off track" is a *geometric* computation off the waypoints (§5), not a
  collision.
- A *list* of morphs makes the entity heterogeneous (one variant per env). But
  heterogeneous **camera** training is blocked (`envs/scene.py:113-118`):
  Genesis 1.2.1 did not feed `active_envs_mask` to Madrona, so all variants
  render superimposed. Genesis 1.3.2 (installed) now passes it upstream as
  `geom_env_mask` (`genesis/vis/batch_renderer.py:112-116`), but the repo still
  routes camera multi-track through Part O **spatial tiling** (each variant on
  its own world tile, `base_env.py:149-154`) — relaxing the heterogeneous-morph
  guard is a separate follow-up. Feature (non-vision) multi-track uses the
  heterogeneous morphs directly.

---

## 4. The rest of the scene: field, sky, lights, cameras

Built in `_build_scene` (`deepracer_env.py:84-257`):

- **Ground = the field.** A green `gs.morphs.Plane` at z=-0.001 with a solid
  `Rough` surface color (`deepracer_env.py:127-131`). It is a colored primitive,
  not a textured DAE, because some DAE grounds render transparent under Madrona
  (see the comment there).
- **Sky = `background_color`** in `VisOptions` (`deepracer_env.py:109`), default
  a bluish `(0.55, 0.72, 0.9)`.
- **Vestigial field overlay:** `track.py:43-46` still parses a `field_rel`
  overlay into `self.field_path`, but it is **never consumed** anywhere in the
  `.py` code — the green plane replaced it.
- **Lighting** depends on the renderer: one directional light for the Madrona
  batch path (`deepracer_env.py:221-225`), a Nyx "sun" for the path-tracer path
  (`deepracer_env.py:188-207`), nothing extra for feature-only training.
- **Cameras** are all optional: onboard `cam` attached to `camera_link` with a
  hand-built mount transform (`_camera_offset_T`, `deepracer_env.py:356-371`), a
  per-env top-down validation cam, and a high-res spectator cam. The vision path
  has three renderer modes (`batch`/Madrona, `nyx` path tracer, rasterizer) with
  many documented Madrona-quirk workarounds (R↔G swap, world-color remap).

---

## 5. The "laws of DeepRacer" — where the domain logic lives

None of this has a Genesis equivalent; it is entirely reimplemented in torch,
all in `deepracer_env.py`, driven by the arrays from `track.py`. Three
functions:

### 5.1 Localize — `_post_physics` (`deepracer_env.py:434-527`)

`track.localize(pos_xy)` (`track.py:84 / 179`) does nearest-waypoint by
`cdist → argmin`, then projects the car's offset onto the local tangent/normal
to get **signed lateral offset** (+ = left of center), **progress in meters**
along arclength, and local `track_yaw`. From that it derives `heading_err`,
forward/lateral velocity, `up_z` (flip detection), and lap counting.

- **Progress & laps** (`deepracer_env.py:461-470`): `d_progress` is the change
  in arclength, wrapped across the finish line, multiplied by `dir_sign` so it
  is positive whenever the car moves the intended way. Crossing the finish while
  moving forward increments `laps`.
- **Driving direction** (`dir_sign`, `deepracer_env.py:304`, reset at 614-619):
  optionally coin-flipped per episode; every track-frame quantity is expressed
  in the env's own direction, so a reversed car still sees `heading_err ≈ 0`.

### 5.2 Termination — `_check_termination` (`deepracer_env.py:570-594`)

There are no walls; leaving the track is a geometric test:

```python
off      = lateral.abs() > (half_width + off_track_margin)   # off the road
flipped  = up_z < 0.3                                        # tipped over
reset_buf = off | flipped | time_out                        # episode ends
rew_buf  += (off | flipped) * crash_penalty                 # −10 terminal hit
```

So going off track **does not push the car back** — the episode terminates,
takes `crash_penalty = −10` (`cfgs.py:28`), and `reset_idx` respawns it at a
fresh random waypoint. There is also an alternate **CMDP/constrained** framing
(`deepracer_env.py:575-589`, gated by `emit_cost`): off-track becomes a *cost*
signal instead of a termination+penalty; only flips or far-off-road terminate.
This feeds the PPO-Lagrangian variant.

### 5.3 Reward — `_compute_reward` (`deepracer_env.py:556-567`) + `envs/rewards.py`

The reward is a dict of **named per-step terms** times a `reward_scales` dict.
The default `deepracer` reward (`rewards.py:58-70`) returns:

| term          | value                                   | default scale |
|---------------|-----------------------------------------|---------------|
| `progress`    | `d_progress` (meters this step)         | 10.0 (dominant) |
| `speed`       | `clamp(v_forward)·dt`                    | 0.5 |
| `centered`    | `exp(-(lateral/half_width)²)·dt`        | 0.5 |
| `heading`     | `-|heading_err|·dt`                     | 0.5 |
| `steering`    | `-|steer|·dt`                           | 0.3 |
| `action_rate` | `-Δaction²·dt`                          | 0.05 |
| `off_track`   | `(wheel off road)·dt`                   | 2.0 |

Every term is a batched `(N,)` tensor summed with its scale
(`deepracer_env.py:559-566`) and accumulated per-episode for logging
(`Episode/rew_<name>`). The reward's `off_track` uses a *tighter* `wheel_margin`
(all-wheels-on-track) than the termination's `off_track_margin`, so the car is
penalized for drifting to the edge before it is actually terminated.

---

## 6. Observations / feature vectors — `envs/features.py`

Two feature sets ship:

- **classic** (`features.py:131`) — the original vector: normalized velocities,
  track-relative pose, last action, and K look-ahead waypoints rotated into the
  body frame. Width `8 + 2·lookahead_k`.
- **perception** (`features.py:178`) — sim2real-oriented: CNN-predictable
  targets (lateral offset, heading error, speed, yaw-rate, sideslip, look-ahead
  curvature) + action-conditioned error channels (speed/steer/yaw error vs a
  fixed nominal bicycle model). Everything normalized by **fixed constants**,
  never episodic statistics.

---

## 7. The declarative experiment layer — `experiment/`

Above the raw sim sits a declarative DSL. `ExperimentSpec` and its slices
(`EnvSpec`, `PolicySpec`, `AlgorithmSpec`, `ObsDRSpec`, …) are **frozen
dataclasses** (`experiment/spec.py`). A spec is **content-hashed**
(`spec.id()`, `spec.py:127-136`) so identical configs share a cached run
directory. `Builder` (`experiment/builder.py`) is the only layer that imports
the heavy libraries; `Builder.sim_cfg()` (`builder.py:71-103`) translates the
dataclass spec into the flat `env_cfg` dict the sim consumes (by calling
`get_env_cfg()` and overwriting keys).

This means there are effectively **two config representations** today:

1. flat dicts from `configs/cfgs.py` (`get_env_cfg` / `get_train_cfg`) — the
   sim and rsl-rl consume these directly;
2. the frozen-dataclass `ExperimentSpec` — the DSL authoring surface, translated
   *down* into (1) by the Builder.

---

## 8. Extension points (pass code directly — no registries)

There are **no string-keyed registries**. You plug in behavior by passing the
Python object (a callable or a class) directly into the spec:

| what          | how you plug it in                                   | selected by |
|---------------|------------------------------------------------------|-------------|
| reward fn     | a `RewardFn` callable                                | `EnvSpec.reward` (None = built-in `deepracer`) |
| feature set   | a `FeatureSet` **class** (or `SelectFeatures` + blocks) | `EnvSpec.feature_set` |
| algorithm     | an `Algorithm` **class** (`requires_cost` flag)      | `AlgorithmSpec.cls` (None = `PPO`) |
| experiment    | an `Experiment` **subclass**                         | the class itself — `run(MyExp)` / `MyExp().run()` |

An experiment is referenced by its **class**, not a name: run it with
`run(MyExperiment)`, `MyExperiment().run()`, a `__main__` block, or the CLI
`module:ClassName` path. The spec's `to_dict()` records callables/classes by
`__qualname__` for the run record (display only), and the run id is a content
hash of the config — no cache, so a run always retrains.
