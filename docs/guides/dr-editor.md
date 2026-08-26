# The DR editor

Tuning domain randomization by launching training runs is slow and blind: you
cannot see what a knob does to the frames the policy consumes, and a
misconfigured knob can silently do nothing for GPU-hours. The DR editor
(`deepracer_genesis/tools/dr_editor/`) closes that loop with three jobs —
**inspect** (render what a config looks like), **tune** (sweep a knob, pick a
range by eye, get pasteable `>>` stage code back), and **prove** (assert a knob
has a measurable effect on the frames the policy sees, on the axis it claims).
It is driven by the same [catalog](../concepts/domain-randomization.md) the DR
stages read.

> Mental model in one sentence: one command → one labeled PNG → one pasteable
> stage line — the editor renders a small DR-stripped scene (or replays a
> recorded frame bank with no Genesis at all), applies each pipeline stage as
> an explicit tensor, and emits the `DomainRandomization*` code that reproduces
> what you picked.

---

## Two tiers, three liveness classes

Everything after the raw render — world colour, pixel noise, policy-res
downscale, image aug, latency, frame stack — is pure torch
(`envs/renderers.py` + `envs/vision_env.py`), so the editor splits its work:

- **Offline tier** (`pipeline.py`): replays those stages on a raw frame batch
  with the *same functions in the same order* the env uses. Needs torch + PIL
  only — no Genesis, no GPU. Feed it frames from a recorded
  [bank](#recording-a-frame-bank-bank) and sweeps become instant.
- **Live tier** (`session.py`): one small inspection scene (default 12 envs,
  onboard + top-down + spectator cameras), built **DR-stripped** and teleported
  so every env sits at one exact pose at one sim instant.

Each knob declares how the editor realizes a new value — its **liveness**:

| Liveness | Meaning | Knobs |
|----------|---------|-------|
| `offline` | pure-torch replay on a raw frame | all 15 image-aug knobs, latency/frame-drop, world_color, pixel_noise |
| `reroll` | poke the live env and re-render | camera_pitch_jitter, camera_pos_jitter, steer_noise, speed_noise, delay_steps |
| `rebuild` | a new scene is required | physics knobs, env_map_tint/multiplier, track_width_scale, all static scene knobs |

Rebuild-class knobs are handled by injecting their build-time cfg before the
session builds (the CLI does this automatically for `grid`/`prove`). One live
gotcha is handled internally: the batch renderer caches frames by sim tick
(`scene.t`), so a mount re-roll alone would return the cached pre-jitter
frames — the session forces one scene step before re-rendering.

## The knob registry (`knobs`)

```bash
python -m deepracer_genesis.tools.dr_editor knobs
python -m deepracer_genesis.tools.dr_editor knobs --layer image
```

Prints the truth table for all 41 knobs — the 31 catalog knobs annotated with
their **schedule** (per_step / per_episode / per_run / build), **liveness**,
**kind**, supported renderers, and cfg path, plus 10 **static** scene knobs
(camera FOV, light intensity, track, …) that a scene editor edits but the
catalog rightly excludes because they are never randomized. `--layer` filters
by catalog layer (`image`, `visual`, `physics`, `actuation`, `geometry`,
`static`).

## Sweeping a knob (`sweep`)

```bash
# live: build a small scene, teleport, filmstrip 12 values
python -m deepracer_genesis.tools.dr_editor sweep brightness

# GPU-free: replay against a recorded bank instead (no Genesis import at all)
python -m deepracer_genesis.tools.dr_editor sweep gamma --bank datasets/editor_bank

# pick the range you liked and save it as a preset
python -m deepracer_genesis.tools.dr_editor sweep brightness --pick 0.6:1.4 \
    --save-preset bright_v1
```

One offline knob, one env, k values across the knob's suggested space (12 by
default; `--points`, or an explicit `--values 0.5,1.0,1.5`) → one filmstrip
PNG under `--out` (default `logs/dr_editor`) plus a pasteable stage line:

```
wrote logs/dr_editor/sweep_brightness_s0-i0.png

pasteable stage code:
  DomainRandomizationCamera(brightness=(0.6, 1.4))
```

Range-shaped aug knobs (brightness, contrast, saturation, gamma) sweep as the
degenerate range `(v, v)` — fully deterministic tiles; scalar-magnitude knobs
(blur, cutout, noise, …) are sampled under the seed, so the strip shows real
draws. `sweep` handles offline knobs only and exits with code 2 for
reroll/rebuild-class knobs, pointing you to `grid` or `prove`.

`--save-preset NAME` writes `configs/dr_presets/NAME.json`;
`emit.load_preset(NAME)` turns it back into ready-to-`>>` stage objects, and
identical preset values produce an identical spec — and therefore an identical
content-hashed run id.

## The per-env grid (`grid`)

```bash
python -m deepracer_genesis.tools.dr_editor grid --knob world_color
python -m deepracer_genesis.tools.dr_editor grid --knob env_map_tint --renderer nyx
python -m deepracer_genesis.tools.dr_editor grid            # bare scene check
```

A contact sheet of N envs at **one shared pose, one sim instant** — an onboard
row and a top-down row, one column per env. This is the layout for
per-episode/per-run knobs (world colour, env maps, mount jitter, physics):
they vary across *envs*, not across steps, so a filmstrip of one env shows
nothing. The same-pose teleport is what makes the sheet honest: when every car
sits at the same waypoint facing the same tangent, the scenes are
pixel-identical *except* for the knob — any difference between columns IS the
knob, not a pose difference masquerading as one. Rebuild-class knobs
(`env_map_*` under `--renderer nyx`, physics) get their build-time cfg
injected into the session automatically.

## The stage strip (`stages`)

```bash
python -m deepracer_genesis.tools.dr_editor stages --target experiments.camera:CamDR
python -m deepracer_genesis.tools.dr_editor stages --preset bright_v1 --bank datasets/editor_bank
```

One strip, seven tiles: `raw → world_color → pixel_noise → policy_res →
image_aug → latency → stack` — the exact application order of the env's
post-render pipeline, ending at what the policy actually consumes.

Why the editor can show this and a training env cannot: in a DR-on env,
`image_buf` is **not** raw — world colour and pixel noise are applied inside
`render()` itself, so no truly-raw buffer ever exists. The editor therefore
builds its live scene **DR-stripped** (with `--target`, the experiment's DR is
carried as replay parameters while the env is built from the stripped spec)
and replays every stage offline from the raw grab. Each stage becomes an
explicit tensor, exact by construction — same functions, same order.

At a static instant the latency tile is an identity by definition (the
k-steps-ago frame of a parked car is the current frame) and the stack tile
shows the fresh-episode priming contract. Without `--target` or `--preset`
there are no DR params and the CLI says so — the tiles would all equal raw.

## Proving knobs (`prove`, `dr_check`)

```bash
python -m deepracer_genesis.tools.dr_editor prove world_color,brightness,frame_drop
```

The automated check suite: every check runs at a shared teleported pose (so
any difference is the knob), prints one PASS/FAIL line per check, writes the
full verdicts to `logs/dr_editor/dr_prove.json`, and exits non-zero if
anything failed. Checks per knob kind:

- **offline knobs** — has-effect at the knob's own stage *and* at the policy
  `stack` endpoint; axis conformance (per-step resample / per-episode palette
  constancy / per-env spread); range sanity (NaN scan + clip fraction at the
  endpoints); sampler coverage of the declared space. Temporal knobs
  (latency_steps, frame_drop) are proven on a synthetic moving sequence, since
  a static instant is identity by definition.
- **mount jitter** — a live re-roll must move the frames, per env.
- **env maps** — cross-env sky difference at one pose (the session is built
  with the knob enabled — rebuild-class).
- **physics** — cross-env divergence of the knob's *declared signals*
  (`Knob.signals` in the catalog) under identical actions from identical
  poses: the cars step with the same command, and e.g. a live friction knob
  must spread `v_forward`/`lateral` across envs.
- **action noise** — one shared command, post-DR actions must differ across envs.
- **declared refusal** — for a knob the session's renderer/modality does *not*
  support, the check is that `spec.validate()` **refuses** to build it. This
  validates the
  [compatibility matrix](../concepts/domain-randomization.md#compatibility-matrix)
  end to end: "inert but loudly declared" is the honesty contract, and matrix
  drift (inert *and* accepted) is a FAIL.

The CI twin is exit-coded and lives beside `camera_check` with the same
contract (verdict lines on stdout, a JSON artifact under `logs/validation`,
exit 0 only when every check passes). Run it on the GPU box before spending
training hours on a DR config:

```bash
python -m deepracer_genesis.validation.dr_check --knobs world_color,brightness
python -m deepracer_genesis.validation.dr_check --knobs env_map_tint --renderer nyx
```

## Recording a frame bank (`bank`)

```bash
python -m deepracer_genesis.tools.dr_editor bank --bank-out datasets/editor_bank \
    --waypoints 0,5,10,20,40
```

Renders once on the GPU box, iterate anywhere: a bank is a directory of raw
(DR-stripped) frames captured over a teleport pose grid (`frames.npz` +
`meta.json` with track/renderer/resolution provenance and a `raw: true`
marker). `sweep --bank` and `stages --bank` then run the whole image tier
instantly with no Genesis import and no GPU.

## Determinism

Every random draw in the editor runs inside a forked, seeded RNG bubble
(`rng.seeded` — `torch.random.fork_rng` over the CPU generator *and* the
frame's CUDA device, since one shared image-aug draw is CPU-side even on CUDA
runs), so the global RNG that training uses is never disturbed and any frame
is reproducible from `--seed`. Artifact filenames carry their seed stamp
(`sweep_brightness_s0-i0.png`), so "show me sample #7 again" is always
answerable.

## Common flags

All commands share `--target module:ClassName` (inspect an experiment's own
config), `--track` (default `reinvent_base`), `--renderer batch|nyx|rasterizer`,
`--num-envs` (default 12), `--res` (default `160x120`), `--waypoint`
(default 5), `--seed` (default 0), and `--out` (default `logs/dr_editor`).

## Deferred

Two pieces are deliberately not built yet: a **renderer-parity compare** (the
same pose rendered under Madrona / Nyx / rasterizer side by side) and a
**notebook widget layer** over the session API — the CLI + PNG loop is the
supported surface today.
