# Tracks

A track is a `.npy` array of waypoints — it **is** DeepRacer's rulebook. Every
track-relative quantity (lateral offset, heading error, progress, curvature) is
torch math on these arrays; the mesh is cosmetic.

> Mental model in one sentence: `.npy` waypoints define the centerline and road
> width; `MultiTrack` turns a raw `(x, y)` into track-frame coordinates
> (`localize`) and answers look-ahead / curvature queries, all batched across envs.

---

## Waypoint loading

Each `.npy` has rows `[center_x, center_y, inner_x, inner_y, outer_x, outer_y]`
(`envs/track.py`). On load the `Track` container:

- drops a duplicated closing waypoint (AWS routes repeat the first point),
- stores `center` `(W, 2)`, computes `half_width = 0.5·‖outer − inner‖`,
- precomputes `tangent`, `normal`, `track_yaw`, `curvature`, `cum_len`, `total_len`.

## `Track` vs `MultiTrack`

- **`Track`** is a pure geometry container — tensors only, no query methods.
- **`MultiTrack`** owns all query logic over padded, batched tensors and maps each
  env to a track variant (`variant_idx`, contiguous blocks), so a batch can span
  several tracks. Per-env views (`total_len_env`, `n_wps_env`) broadcast by variant.

## Query methods

### `localize(pos_xy)` → dict

Nearest-waypoint projection returning `{wp_idx, lateral, half_width, progress_m,
track_yaw}`:

- `wp_idx = argmin ‖center − pos‖`
- `lateral = (pos − c) · normal` (signed offset)
- `progress_m = (cum_len[wp_idx] + (pos − c)·tangent) mod total_len` (wraps at the finish line)

### `lookahead(wp_idx, k, stride, dir_sign)` → `(N, k)`

Indices of the next `k` waypoints at fixed `stride`; `dir_sign` flips the walk for
reversed-direction episodes. `lookahead_points(idx)` gathers their `(N, k, 2)` world
positions.

### `curvature_ahead(progress_m, distances, dir_sign)` → `(N, H)`

Signed curvature sampled at fixed arc-length distances ahead; for reversed envs both
the probe distance and the curvature sign flip.

### `spawn_pose(env_ids, random_start, lateral_noise, yaw_noise)` → `(pos_xy, yaw)`

Spawn at a random (or fixed) waypoint, perturbed laterally along the normal and
rotationally about the tangent. The env may additionally coin-flip the driving
direction (see `dir_sign` in [Feature vectors](features.md#25-dir_sign-the-companion-convention)).

## Adding a track

The `notebooks/track_designer.ipynb` notebook builds a route from waypoints,
`install_track(name, route)` registers it, and you drive it with a scripted
controller to sanity-check before training. Reference the track by name in an
environment stage: `FeatureEnvironment(tracks=("my_track",))`.

## Width variants (camera-mode track-width DR)

The `track_width_scale` DR knob is **feature-only**: it scales the rulebook,
which a baked mesh cannot follow, so a camera env would see a road that
contradicts the rules. Under camera the width itself must vary —
`tools/track_builder.py` bakes it:

- **`scale_route_width(route, scale)`** scales a `(W, 6)` route's border
  columns about the centerline (the centerline is untouched), so waypoint
  count, arclength, and spawn poses are identical across variants.
- **`width_variants(track, scales)`** installs one generated track per scale
  (named `<track>_wNNN`, e.g. `tight_oval_w090`; a scale of 1.0 reuses the
  source track) and returns the names, skipping already-baked variants unless
  `force=True`.

Pass the returned names straight to an environment stage:

```python
CameraEnvironment(tracks=width_variants("tight_oval", (0.9, 1.0, 1.15)), ...)
```

Each env is assigned one variant via the existing multi-track tiling, so the
schedule is **per env, fixed for the run**. Because `Track.half_width` derives
from the same route borders the mesh is built from, the rulebook follows the
mesh **by construction** — no desync possible. Variant meshes use the
procedural generated-track look (road ribbon + border lines + dashed
centerline), also when the source track is an official DAE mesh.
Tiled variants work under **Madrona and Nyx** alike (tiles are plain meshes on
separate world tiles — no per-env visibility needed; camera-on-CPU remains
single-track). `scripts/verify_width_variants.py` proves the pipeline end to
end on Madrona, `scripts/verify_nyx_tiling.py` on Nyx.

Width variants are the one-axis special case of the
[track zoo](../guides/track-zoo.md), which adds palette and field axes on the
same bake machinery plus a compile → see → plumb workflow.
