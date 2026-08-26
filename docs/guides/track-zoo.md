# The track zoo

Camera policies steer by what they see, and most of what they see is baked:
road colour, line colour, track width, and the ground around the road all live
in the track mesh, so none of them can be a runtime DR knob the way brightness
or pixel noise can. The track zoo (`deepracer_genesis/tools/zoo.py`) turns
that constraint into the design: declare a population of track variants — base
shape × width × road/line palette × field colour — **compile** it down to
baked, registered tracks once, **see** the compiled result before spending
GPU-hours, then **plumb** the same assets into training, where the
randomization simply *is* cars living on those variants.

> Mental model in one sentence: manifest → compile → see → plumb — a `Zoo` of
> `TrackVariant`s is linted and baked into registered tracks, previewed as a
> bare-scene tile grid (`view`) or under the real env with driving cars
> (`watch`), then handed to training as `CameraEnvironment(tracks=names)`,
> where each env gets one variant, fixed for the run.

---

## Pre-baked variants ARE the DR

There is no sampler and no schedule knob for scene appearance: the zoo is the
distribution. Each env is assigned one variant through the existing
[multi-track tiling](../concepts/tracks.md#width-variants-camera-mode-track-width-dr)
(every variant sits on its own world tile; envs are balanced across tiles), so
the schedule is **per env, fixed for the run** — and variety comes from the
zoo's *size*, not from re-rolls. Six variants means six looks in the batch,
every step of the run.

Everything downstream of the raw render is unchanged: the obs-side knobs from
the [DR catalog](../concepts/domain-randomization.md) — world colour, image
aug, latency, mount jitter — stack on top of whichever variant an env lives
on, exactly as they do on a single track. The zoo replaces nothing; it feeds
the same pipeline a wider set of raw frames.

[Width variants](../concepts/tracks.md#width-variants-camera-mode-track-width-dr)
are the one-axis special case of this: `width_variants()` predates the zoo and
still works. The zoo adds the palette and field axes, the manifest, the lint,
and the see-it-first tooling on top of the same bake machinery
(`scale_route_width` + `install_track`).

## The manifest

A zoo is config-as-code: two frozen dataclasses, no registry, no YAML.

```python
from deepracer_genesis.tools.zoo import TrackVariant, Zoo

MY_ZOO = Zoo("roadsim_v1", (
    TrackVariant("reinvent_base"),                       # the base look
    TrackVariant("reinvent_base", width=0.9),
    TrackVariant("reinvent_base", palette="dusk"),
    TrackVariant("reinvent_base", palette="asphalt_light", field="sand"),
    TrackVariant("reinvent_base", width=1.15, palette="faded", field="concrete"),
))
```

- **`base`** — any registered track name; its route supplies the centerline.
- **`width`** — scale about the centerline (borders move, centerline doesn't,
  so spawn poses and arclength are identical across variants).
- **`palette`** — road/border/centerline colours: a named entry from
  `PALETTES` (`classic`, `asphalt_light`, `dusk`, `faded`), an explicit
  0-255 RGB dict (any subset of `road`/`border`/`centerline`, merged over the
  default), or `None` for the stock look.
- **`field`** — a per-tile ground colour: a named entry from `FIELDS`
  (`grass`, `sand`, `concrete`), an explicit 0-255 RGB tuple, or `None` for
  the global ground plane. A field is baked **into the mesh** as a ground quad
  under the road (track bounding box plus a margin), so it travels with the
  asset and tiles correctly under every renderer.

Compiled names are deterministic: an all-default variant reuses its base name
(and its base assets — no duplicate bake); anything else appends `_wNNN` for
width, the palette name (or `p<6-hex>` for a custom dict), and `f<name>` (or
`f<6-hex>`) for a field — e.g.
`reinvent_base_w115_faded_fconcrete`. Identical manifest → identical names →
identical assets, so a zoo is reproducible from its code alone.

One axis is deliberately **inexpressible**: sky and lighting. They are
scene-global under Madrona and the rasterizer — every tile shares them — so
the manifest has no field to ask for them per variant. Illegal states are
unrepresentable rather than silently ignored; per-env sky variation is Nyx's
`env_map_*` DR territory (see [Renderers](../concepts/renderers.md)).

## The bake-time lint

The compiler refuses a variant whose look would destroy the driving signal.
For the resolved palette, the relative luminance
(`0.2126 R + 0.7152 G + 0.0722 B`, 0-255 scale) of the **centerline** and the
**border** must each sit at least `MIN_LINE_CONTRAST = 60` away from the
road's, and the width must be positive — otherwise `compile` raises
`ValueError` before anything is written.

Why lint at bake time rather than trust the author: the lines are what the
policy steers by. A palette where the centerline melts into the road doesn't
*harden* the policy the way aggressive image aug does — it erases the signal
itself, and the failure would surface as GPU-hours of a policy that cannot
learn, not as an error. A light-grey road under the default amber centerline
is enough to trip it — `TrackVariant("reinvent_base", palette={"road": (180,
180, 180)})` fails with:

```
ValueError: palette rejected: |luminance(centerline) - luminance(road)| = 8 < 60
— the centerline must stay visible against the road or the variant erases the
signal the policy steers by
```

## The manifest file

**What to build lives in a manifest file — config-as-code, the project rule —
and the file is also the program: give it a `__main__` and run it directly.**

```bash
uv run examples/zoos.py        # compile + watch, no CLI involved
```

```python
if __name__ == "__main__":
    watch(population, num_envs=32)     # or view(population), or compile_zoo(...)
```

A manifest is an ordinary Python file declaring one or more `Zoo` objects
(see `examples/zoos.py` for a commented set), mixing three source kinds that
expand in declaration order:

```python
# my_zoos.py
from deepracer_genesis.tools.zoo import OfficialSample, RandomShapes, TrackVariant, Zoo

population = Zoo("population", (
    OfficialSample(24, seed=7, jitter=0.4, looks=True),   # real circuits, noised
    RandomShapes(8, seed=7),                              # synthetic wildcards
    TrackVariant("Monaco", wall="white"),                 # one exact pick
))
```

- **`OfficialSample`** — a seeded sample of the ORIGINAL DeepRacer library
  (~126 tracks, listed once from the community race-data repository and
  cached; missing tracks download on first use, `fetch=False` = offline).
  `jitter` adds smooth waypoint noise: a low-frequency periodic displacement
  along the local normals, original per-waypoint width profile preserved,
  gated RELATIVE to each original's own metrics (≥ 80% of its minimum turn
  radius and corridor clearance, no self-intersection), halving amplitude on
  failure down to the untouched original — never worse to drive. `looks`
  draws a contrast-linted palette, a per-tile field (~50%), and a perimeter
  wall (~60%) per clone.
- **`RandomShapes`** — fully synthetic circuits (convex-hull construction
  with inward punches; drivability-linted, size/width/look randomized).
- **`TrackVariant`** — one explicit variant: any base track with `width`,
  `palette`, `field`, `wall` axes.

Everything is deterministic in the declared seeds — the same manifest
compiles to the same population byte for byte, which matters because the
names enter the run id. Baking is cached by the deterministic names;
`--force` rebakes. **Without a manifest, the CLI uses the recommended
default: `Zoo("default", (OfficialSample(32),))`** — 32 real circuits,
noised and look-randomized.

## The three commands

All three take the manifest as `path/to/zoos.py[:name]` or `module[:name]`
(a bare module with one `Zoo`, or one named `zoo`, resolves automatically);
every subcommand compiles first — `view` and `watch` operate on the
compiled names. The remaining flags are strictly operational (how to run,
never what to build).

### `compile` — manifest → linted, baked, registered tracks

```bash
python -m deepracer_genesis.tools.zoo compile                          # the default population
python -m deepracer_genesis.tools.zoo compile my_zoos.py:population
python -m deepracer_genesis.tools.zoo compile examples/zoos.py:showcase
```

Prints one registered name per variant and ends with the pasteable plumbing
line (see below).

### `view` — the bare-scene testing ground

```bash
python -m deepracer_genesis.tools.zoo view my_zoos.py:population

# headless box: render one top-down overview PNG instead of the viewer
python -m deepracer_genesis.tools.zoo view my_zoos.py:population \
    --screenshot logs/zoo/overview.png
```

No env, no cars, no RL — a bare `gs.Scene` with every variant on its world
tile, for judging scene composition by eye. By default the grid
**auto-compacts** to the tracks' footprint: training's 100 m spacing exists to
keep onboard cameras isolated, but a bare-scene view has no cameras to
isolate, and at 100 m small tracks are specks. Pass the training value to
preview the real layout instead:

```bash
python -m deepracer_genesis.tools.zoo view my_zoos.py:population --grid-spacing 100
```

`--seconds N` auto-closes the interactive viewer.

### `watch` — cars driving on the zoo

```bash
python -m deepracer_genesis.tools.zoo watch my_zoos.py:population --num-envs 32

# headless: fixed-length run, photos only
python -m deepracer_genesis.tools.zoo watch my_zoos.py:population \
    --no-gui --steps 400 --photos-every 100
```

The training world, watchable: the real `DeepRacerEnv` is built on the
compiled tracks (spawn randomization ON, exactly as training has it), every
car is driven by the scripted centerline follower, and every `--photos-every`
steps a few envs' **car-view photos** (`--photo-envs`, default 4) are saved
under `--out` (default `logs/zoo`). The photos are `obs_image_buf` — the
policy's own frames — so they carry the full obs-side DR that the viewer
window cannot show: world colour and pixel noise are applied inside `render()`
itself, and only the policy buffer sees them. `--randomize` additionally draws
per-run physics DR, to watch cars with different bodies diverge on identical
tiles.

Tiles pack **compactly by default**: the onboard batch cameras clip at a
20 m far plane, so a neighboring tile is invisible as soon as its geometry
sits beyond that — the automatic spacing is `footprint + 26 m` (verified:
same pose renders **bit-identically** at that spacing and at the
conservative 100 m). Override per-run with `--grid-spacing M`, or pin it in
the manifest (`Zoo(..., grid_spacing=100.0)`). The compacted world is much
easier to take in from the viewer — dozens of tracks in one glance.

## Plumbing into training

Every run of the CLI ends by printing the exact line training needs:

```
plumb into training:
  CameraEnvironment(tracks=('reinvent_base', 'reinvent_base_w090', 'reinvent_base_dusk', ...))
```

Or from code, in an [experiment](../concepts/experiments.md):

```python
from deepracer_genesis.tools.zoo import compile_zoo

CameraEnvironment(tracks=compile_zoo(MY_ZOO), ...)
```

There is no separate training path: `compile_zoo` returns registered track
names, and the assets training loads are byte-for-byte the ones `view` and
`watch` showed. What you saw is what the policy gets.

## Renderer support and cost

Tiled multi-track works under **Madrona and Nyx** alike: tiles are plain
meshes on separate world tiles, ordinary scene content — no per-env visibility
masking needed, which is what the old Nyx multi-track guard was actually
about. `scripts/verify_nyx_tiling.py` proves it end to end on Nyx (rulebook
per variant, same-tile determinism, cross-variant visible difference), and
`scripts/verify_width_variants.py` does the same on Madrona. Camera-on-CPU
remains single-track.

The cost model to keep in mind: tiles are scene content, so **all K variant
meshes load into every env's world**. Scene build time, GPU memory (Madrona's
device heap in particular), and render cost all grow with zoo size — a
handful of variants is cheap, but benchmark before scaling to a large zoo,
e.g. a short `watch --no-gui --steps 200` run at your training `--num-envs`.

## Worked example: the invisible donut

Why "see before training" is a real step and not ceremony: `donut_track`, a
~1.5 m room-scale ring, compiles cleanly, lints cleanly, and drives cleanly
under the feature env. Under the onboard camera it is **nearly invisible** —
at the stock 10-degree mount pitch the visible ground starts beyond the whole
ring, so the policy's frames contain almost no track at all. Nothing in the
manifest, the lint, or the rulebook can catch that; the `watch` photos caught
it immediately, because they are the policy's own frames. This is why
`demo_zoo` defaults to reinvent-sized variants (they fill the onboard frame),
and why a zoo should be watched once before it is trained on.
