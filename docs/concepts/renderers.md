# Renderers

Rendering is a **strategy**: the env holds one `Renderer` chosen from config, and
the strategy owns every vision decision. Feature-vector training uses no renderer;
camera training uses a batched GPU renderer.

> Mental model in one sentence: `make_renderer(vision_cfg)` picks `NullRenderer`
> (no camera), `MadronaRenderer` (fast batched rasterizer), or `NyxRenderer` (path
> tracer) — and the same strategy also owns the world-color DR and the debug views.

---

## Choosing a renderer

`make_renderer` (`envs/renderers.py`) dispatches on config:

- `vision=False` → **`NullRenderer`** — state observations only; `obs()` returns None.
- `vision=True`, `vision_renderer="batch"` → **`MadronaRenderer`** — Genesis
  `BatchRenderer`, a camera attached to the car's `camera_link`. Fast; the default
  for camera training.
- `vision=True`, `vision_renderer="nyx"` → **`NyxRenderer`** — Nyx path-tracer
  sensors, true texture colors, slower.

The camera is DeepRacer-native `(160, 120)` RGB by default, pitched down for a
forward-looking view (`camera_offset_T`).

## World-color domain randomization

Both camera renderers keep a per-env color transform (`color_mat` `(N,3,3)` +
`color_bias`) and apply it to each frame. `resample_appearance()` redraws it every
reset via `sample_world_color()` (in `randomization/visual.py`) — a hue rotation
plus saturation/value scaling in YIQ chroma space. See
[Domain randomization](domain-randomization.md).

## Camera-mount jitter (Madrona + rasterizer)

`randomize_mount()` perturbs each env's camera pitch and position
(`camera_pitch_jitter_deg`, `camera_pos_jitter_m`) via
`sample_mount_transforms()` — applied **once per run**, from the env's
`__init__` alongside the physics DR, not per episode. It works on Madrona
(batched attach offsets) and on the rasterizer path (`RasterizerObsRenderer`
holds one camera per env, so each attach offset is rewritten individually).
Nyx is excluded: a single batched sensor with one shared offset, so there are
no per-env mounts to jitter.

## Debug views (independent of the obs renderer)

Two human-facing views work even in a `NullRenderer` (feature-only) env, because
they use their own cameras:

- **Spectator** (`render_spectator()`) — one high-resolution bird's-eye image of the
  whole track with every car, from a single static camera.
- **Top-down** (`render_topdown()`) — an optional per-env bird's-eye view for
  validation. Madrona poses it per track variant and returns `(N, H, W, 3)`; Nyx
  shares one pose across envs.

`NyxRenderer` sets `merge_fixed_links = False` (the Nyx exporter rejects merged
links); Madrona merges them.
