# Domain randomization catalog

Every randomizable knob in the system, in one table. This page mirrors the
single source of truth — `deepracer_genesis/randomization/catalog.py` — which
also carries the **compatibility matrix**: which env modalities a knob has
any effect in, and (for camera envs) which renderers actually apply it.
`ExperimentSpec.validate()` enforces that matrix, so an experiment that
declares a knob its modality or renderer cannot apply **refuses to build**
instead of silently sampling nothing.

Suggested ranges are starting points (the catalog's `Space` defaults), not
live config values. To browse programmatically:

```python
from deepracer_genesis.randomization.catalog import CATALOG, BY_NAME, by_layer

BY_NAME["blur"].space        # suggested range
by_layer("physics")          # all physics knobs
```

Which pipeline stage turns each layer on:

| Catalog layer | Pipeline stage |
|---------------|----------------|
| Physics | `DomainRandomizationPhysics(...)` |
| Image + camera-mount jitter | `DomainRandomizationCamera(...)` |
| Actuation | `DomainRandomizationActions(...)` |
| Visual `world_color` | `DomainRandomizationTrackAppearance(strength=...)` |
| World/scene (tracks, widths, looks, walls) | the [track zoo](../guides/track-zoo.md) track list + `random_direction` |

## Physics (dynamics before stepping)

| Knob | Suggested range | Modalities | Renderers | Note |
|------|-----------------|------------|-----------|------|
| `friction` | 0.6 – 1.4 | camera + feature | all | per-link friction ratio |
| `mass_shift` | ±0.05 | camera + feature | all | +/- kg per link |
| `com_shift` | ±0.01 | camera + feature | all | +/- m per link, 3 axes |
| `steer_kp_scale` | 0.8 – 1.2 | camera + feature | all | scales steering kp and kv |
| `wheel_kv_scale` | 0.8 – 1.2 | camera + feature | all |  |
| `armature` | 0.0 – 0.01 | camera + feature | all |  |

## Geometry (track shape)

| Knob | Suggested range | Modalities | Renderers | Note |
|------|-----------------|------------|-----------|------|
| `track_width_scale` | 0.9 – 1.15 | feature | all | per-episode scale on rulebook half-width; feature mode only (the mesh is fixed at build, so a camera env would see a road that contradicts the rules). Visible width under camera = width-variant tracks: tools.track_builder.width_variants() |

## Visual (world & camera mount)

| Knob | Suggested range | Modalities | Renderers | Note |
|------|-----------------|------------|-----------|------|
| `world_color` | 0.0 – 0.6 | camera | all | per-episode YIQ remap strength |
| `camera_pitch_jitter` | ±2.0 | camera | madrona + rasterizer | +/- deg mount pitch (per env, once per run) |
| `camera_pos_jitter` | ±0.01 | camera | madrona + rasterizer | +/- m mount position (per env, once per run) |
| `pixel_noise` | 0.0 – 0.05 | camera | all | gaussian pixel noise scale |
| `env_map_tint` | 0.35 – 0.75 | camera | nyx | per-env HDRI sky tint (Nyx; baked at build -> per-env-fixed) |
| `env_map_multiplier` | 0.5 – 2.0 | camera | nyx | per-env sky exposure multiplier (Nyx; per-run) |

## Actuation (action-side)

| Knob | Suggested range | Modalities | Renderers | Note |
|------|-----------------|------------|-----------|------|
| `steer_noise` | 0.0 – 0.05 | camera + feature | all | gaussian steering-command noise scale |
| `speed_noise` | 0.0 – 0.05 | camera + feature | all | gaussian speed-command noise scale |
| `delay_steps` | 0 – 3 | camera + feature | all | command latency in steps |

## Image (post-render pixel pipeline)

| Knob | Suggested range | Modalities | Renderers | Note |
|------|-----------------|------------|-----------|------|
| `brightness` | 0.7 – 1.3 | camera | all |  |
| `contrast` | 0.7 – 1.3 | camera | all |  |
| `saturation` | 0.7 – 1.3 | camera | all |  |
| `hue` | 0.0 – 0.1 | camera | all | IQ-plane rotation fraction |
| `blur` | 0.0 – 0.5 | camera | all | max gaussian sigma (one sigma per batch draw; a per-env coin picks which envs get the blurred frame) |
| `cutout` | 0.0 – 0.5 | camera | all | per-env probability of one random occlusion patch |
| `noise` | 0.0 – 0.05 | camera | all | additive gaussian pixel sigma (intensity-independent; see shot_noise for the intensity-dependent one) |
| `gamma` | 0.7 – 1.5 | camera | all | exposure/tone curve (render has no auto-exposure) |
| `white_balance` | ±0.1 | camera | all | per-channel gain magnitude; colour cast + R<->G insurance |
| `vignette` | 0.0 – 0.4 | camera | all | max radial corner darkening |
| `distortion` | ±0.15 | camera | all | wide-angle barrel/pincushion coefficient |
| `crop` | 0.0 – 0.2 | camera | all | max crop fraction, resized back (FOV / principal-point jitter) |
| `shot_noise` | 0.0 – 0.05 | camera | all | brightness-dependent (sqrt-intensity) sensor noise |
| `latency_steps` | 0 – 2 | camera | all | camera pipeline delay in control steps (likely the largest untreated sim2real gap for a 4 m/s car) |
| `frame_drop` | 0.0 – 0.1 | camera | all | per-step probability of repeating the previous frame |

---

*This page is generated from the catalog — to refresh it after editing
`randomization/catalog.py`, rerun the table script in the docs commit that
added this page (the tables above are its verbatim output).*
