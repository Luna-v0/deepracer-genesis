# Domain randomization

Domain randomization (DR) perturbs the simulator so a policy trained in Genesis
transfers to the real DeepRacer. This repo keeps **all DR *definitions* in one
folder** — `deepracer_genesis/randomization/` — even though each knob is *applied*
at a different layer.

> Mental model in one sentence: `randomization/` is the single home for *what can
> be randomized* (the `CATALOG`) and *how each knob samples* (a `Space`), while
> *where* it acts stays at its native layer — physics before stepping, visual in
> the renderer, actuation/image env-side in the sim step.

---

## The catalog — one table of every knob

`randomization/catalog.py` is documentation-as-data: importing it has no runtime
effect. Each `Knob` (`catalog.py:43`) records its name, a suggested `Space`, the
**layer** it acts at, the `cfg` key its value lands in, the **signal(s)** it
perturbs (the Part K vocabulary — see [Feature vectors](features.md)), and its
**compatibility** (`modalities` / `renderers` — see the
[compatibility matrix](#compatibility-matrix) below):

```python
Knob("friction", FloatRange(0.6, 1.4), "physics", "rand.friction_range",
     ("v_forward", "lateral"), "per-link friction ratio")
```

| Layer | Applied where | Schedule | Example knobs |
|-------|---------------|----------|---------------|
| `physics` | env `__init__`, before any stepping (`randomization/physics.py`) | once per run, per env | friction, mass_shift, com_shift, steer_kp_scale, wheel_kv_scale, armature |
| `geometry` | the rulebook width at reset (`base_env.reset_idx`); feature mode only | per episode | track_width_scale |
| `visual` | in the renderer (`randomization/visual.py`) | world_color per episode; mount jitter once per run; env_map baked at build | world_color (YIQ remap), camera_pitch_jitter, camera_pos_jitter, pixel_noise, env_map_tint, env_map_multiplier |
| `actuation` | env-side in the sim step (`base_env._apply_action_dr`) | per step | steer_noise, speed_noise, delay_steps |
| `image` | env-side on the camera obs (`vision_env._observe_camera` → `apply_image_aug` in `randomization/image_aug.py`) | per step, per sub-env | brightness, contrast, saturation, hue, blur, cutout, noise, gamma, white_balance, vignette, distortion, crop, shot_noise |
| `image` (temporal) | env-side, stateful (`FrameLatency` in `randomization/latency.py`) | advanced once per step | latency_steps, frame_drop |

`CATALOG` (`catalog.py:69`) is the full list; `by_layer(layer)` filters it and
`BY_NAME` indexes it. DR, HPO, and the build-time learnability check all read this
one table.

## Search-space types (shared with HPO)

Ranges are declared once as `Space` objects in `randomization/spaces.py` and reused
by both DR (sampled batched on GPU, at each layer's schedule — see the table above)
and [HPO](../guides/hpo.md) (searched as a scalar per trial):

- `FloatRange(lo, hi, log=False)` — continuous; `suggest()` for HPO, `sample(n, device)` for DR.
- `IntRange(lo, hi)` — integer, both paths.
- `SymRange(m)` — samples `[-m, m]`; DR-native (`suggest()` raises — no scalar to freeze).
- `Choice(values)` — HPO-only categorical; `sample()` raises (no batched-GPU categorical DR).

See the plan's Part H (`REFACTOR_PLAN.md`) for why one *type* is shared but the
declaration *sites* stay separate.

## What each layer perturbs

- **Physics** (`randomize_physics` + `randomize_armature`, applied **once per
  run** from the env's `__init__` — *not* per episode): per-link friction, base
  mass, COM offset, steering `kp`/`kv` scale, wheel `kv` scale, and `armature`
  (reflected rotor inertia added to the joint mass matrix). Once per run because
  per-reset genesis setters sporadically crash even on genesis 1.2.3
  (`randomization/physics.py:3-15`, `base_env.py:326-337`) — so each env keeps
  its own randomized body for the run and only spawn/state randomizes per
  episode.
- **Geometry** (`track_width_scale`, applied in the rulebook at reset, feature
  mode only): a per-episode scale on the rulebook half-width — torch-only, no
  crash-prone genesis setter. Under camera the *visible* width must vary
  instead: bake width-variant tracks with
  `tools.track_builder.width_variants()` (see [Tracks](tracks.md)).
- **Visual** (in the [renderer](renderers.md)): a per-episode **world-color YIQ
  remap** (hue/saturation/value shift in chroma space), camera-mount
  pitch/position jitter (Madrona + rasterizer, **once per run** alongside the
  physics DR), additive pixel noise, and the per-env Nyx sky env-map
  tint/multiplier (baked at build → per-env-fixed for the run).
- **Actuation** (`base_env._apply_action_dr`, env-side in the sim step): k-step
  command latency then per-channel Gaussian noise on `[steer, speed]`.
- **Image** (env-side on the camera obs, `vision_env._observe_camera`):
  `apply_image_aug` (`randomization/image_aug.py`) resamples the 13 stateless
  per-frame effects — brightness/contrast/saturation/hue/blur/cutout/noise plus
  the Part P.2 sensor block (gamma, white_balance, vignette, distortion, crop,
  shot_noise) — per step per sub-env; the stateful temporal knobs
  (latency_steps, frame_drop) live in `FrameLatency`
  (`randomization/latency.py`), advanced once per control step.

## Compatibility matrix

Every knob declares the env **modalities** it has any effect in and, for camera
envs, the **renderers** that actually apply it (`Knob.modalities` /
`Knob.renderers` in `randomization/catalog.py`). `ExperimentSpec.validate()`
enforces the matrix (`_validate_knob_compat`): an activated knob either acts or
**refuses to build** — never a silent no-op.

| Knob(s) | Modalities | Renderers |
|---------|------------|-----------|
| friction, mass_shift, com_shift, steer_kp_scale, wheel_kv_scale, armature | feature + camera | all |
| steer_noise, speed_noise, delay_steps | feature + camera | all |
| track_width_scale | feature only | — (feature envs render nothing) |
| world_color, pixel_noise, all 15 image knobs | camera only | all |
| camera_pitch_jitter, camera_pos_jitter | camera only | madrona, rasterizer |
| env_map_tint, env_map_multiplier | camera only | nyx |

The renderer column is checked against `EnvSpec.effective_renderer` — the single
source of truth for which renderer will actually run: feature envs render
nothing; camera envs on the CPU backend always fall to the per-env rasterizer;
otherwise `render` picks Nyx or Madrona. What refuses to build:

- **`track_width` under a camera env** — the rendered mesh is baked, so scaling
  only the rulebook would desync what the camera sees from what the rules
  enforce. Use `tools.track_builder.width_variants()` to bake real
  width-variant tracks instead ([Tracks](tracks.md)).
- **Camera-mount jitter under Nyx** — Nyx has one shared sensor offset, no
  per-env mounts (Madrona batches the attach offset; the rasterizer holds one
  camera per env).
- **`env_map` under Madrona / rasterizer** — per-env sky maps exist only as Nyx
  `EnvironmentMapAsset`s baked at build; Madrona replicates its lights
  identically into every world.
- **Unknown `image_aug` / `physics` keys** — a typo would be silently ignored
  at runtime, so it is a build error (`SpecError`).

Neutral physics values (`NEUTRAL_PHYSICS` in `experiment/spec.py`) do not count
as activations — the physics stage always emits every key, so e.g. its default
`track_width=(1.0, 1.0)` still builds under a camera env.

## What is NOT randomized

- **Lighting** is fixed at build time (the Nyx sky env-map tint/multiplier above
  is per-env randomized, but also baked at build — fixed for the run).
- The **offline texture bake** in `randomization/appearance.py` (sha1-cached track/
  field variants) is a rasterizer tool and is **not wired into camera training**;
  train-time appearance DR is the world-color remap above.

## Authoring DR in an experiment

DR is added with `>>` stages (see [Experiments](experiments.md)):

```python
CameraEnvironment(render="madrona", num_envs=128)
    >> DomainRandomizationTrackAppearance(strength=0.6)      # world-color
    >> DomainRandomizationCamera(brightness=(0.7, 1.3), hue=0.05, blur=0.3,
                                 camera_jitter=True)          # image + mount
    >> DomainRandomizationPhysics()                           # friction/mass/...
    >> AsymmetricCameraPolicy(actor_keys=("camera",), critic_keys=("camera", "state"))
    >> DomainRandomizationActions(steer_noise=0.02, speed_noise=0.05, delay_steps=1)
```

Each stage accepts either a raw range/scalar or a `Space` (a suggested default
range lives in the catalog).
