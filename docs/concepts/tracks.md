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
- drops any waypoint less than 1 mm from its successor — 54 of the 62 shipped
  routes also repeat one mid-loop, and a zero-length segment has no direction:
  `arctan2(0, 0)` poisons `track_yaw`, which then spikes `curvature`,
- stores `center` `(W, 2)`, computes `half_width = 0.5·‖outer − inner‖`,
- precomputes `tangent`, `normal`, `track_yaw`, `curvature`, `cum_len`, `total_len`.

`W` is the **post-filter** waypoint count — usually one or two below the row count
of the raw `.npy`. Both drops live in one loader, `envs.track.load_route()`, which
`deepracer_genesis.tracks.info()` also calls, so `info(name).num_waypoints` is
always the `W` the env actually drives.

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
