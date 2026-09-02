# Renderers

Rendering is a **strategy**: the env holds one `Renderer` chosen from config, and
the strategy owns every vision decision. Feature-vector training uses no renderer;
camera training uses a batched GPU renderer.

> Mental model in one sentence: `make_renderer(vision_cfg)` picks `NullRenderer`
> (no camera), `MadronaRenderer` (fast batched rasterizer), `NyxRenderer` (path
> tracer), or `RasterizerObsRenderer` (the CPU fallback) — and the same strategy
> also owns the world-color DR and the debug views.

---

## Choosing a renderer

`make_renderer` (`envs/renderers.py`) dispatches on config:

- `vision=False` → **`NullRenderer`** — state observations only; `obs()` returns None.
- `vision=True`, `vision_renderer="batch"` → **`MadronaRenderer`** — Genesis
  `BatchRenderer`, a camera attached to the car's `camera_link`. Fast; the default
  for camera training.
- `vision=True`, `vision_renderer="nyx"` → **`NyxRenderer`** — Nyx path-tracer
  sensors, true texture colors, slower.
- `vision=True`, `vision_renderer="rasterizer"` → **`RasterizerObsRenderer`** — the
  CPU path (`backend="cpu"`), where Madrona and Nyx are unavailable. One camera
  shared across envs, plus `env_separate_rigid` so each env renders only its own
  car. A debug / small-`num_envs` path, not a throughput one.

The camera is DeepRacer-native `(160, 120)` RGB by default, pitched down for a
forward-looking view (`camera_offset_T`).

## World-color domain randomization

Both camera renderers keep a per-env color transform (`color_mat` `(N,3,3)` +
`color_bias`) and apply it to each frame. `resample_appearance()` redraws it every
reset via `sample_world_color()` (in `randomization/visual.py`) — a hue rotation
plus saturation/value scaling in YIQ chroma space. See
[Domain randomization](domain-randomization.md).

## Camera-mount jitter (Madrona)

`MadronaRenderer.randomize_mount()` perturbs the camera's pitch and position per
episode (`camera_pitch_jitter_deg`, `camera_pos_jitter_m`) via
`sample_mount_transforms()`. Nyx does not jitter the mount.

## Debug views (independent of the obs renderer)

Two human-facing views work even in a `NullRenderer` (feature-only) env, because
they use their own cameras:

- **Spectator** (`render_spectator()`) — one high-resolution bird's-eye image of the
  whole track with every car, from a single static camera. `env_separate_rigid`
  (see `RasterizerObsRenderer`) batches that camera too and draws one car per
  frame, so the strategy recomposes the batch into one image: the median across
  envs is the empty track, and each car is pasted back where its own frame
  departs from it. Consumers always get a single `(H, W, 3)` fleet view.
- **Top-down** (`render_topdown()`) — an optional per-env bird's-eye view for
  validation. Madrona poses it per track variant and returns `(N, H, W, 3)`; Nyx
  shares one pose across envs.

`NyxRenderer` sets `merge_fixed_links = False` (the Nyx exporter rejects merged
links); Madrona merges them. Every strategy also declares `env_separate_rigid`
(default `False`), which `build_scene` reads straight off the port into
`VisOptions`.
