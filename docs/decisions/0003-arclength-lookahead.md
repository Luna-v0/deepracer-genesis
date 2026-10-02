# 0003 — Lookahead sampled by arc-length meters, not waypoint indices

- Status: accepted
- Date: 2026-09-08

## Context

`Track.lookahead` walked raw waypoint indices, so the feature vector's
lookahead horizon in meters was `k × stride × spacing` — and waypoint spacing
varies ~48× across the 298 registered tracks (0.017 m on `donut_track` to
0.888 m on `H_track`). With the defaults the horizon ranged roughly 0.5 m to
26.6 m depending on track; within one eval set the holdouts spanned ~5×. Both
feature-vector actors and camera critics (whose privileged state uses
`ClassicFeatures`) were affected, so cross-track comparisons partly measured
track discretization, not driving. (PROBLEMS.md P1.)

## Decision

Add `MultiTrack.lookahead_points_m(progress_m, distances, dir_sign)` — the
same searchsorted-on-`cum_len` pattern `curvature_ahead` already used — and
switch both feature call sites (`ClassicFeatures.compute`, the
`lookahead_xy` block) to it. Points are sampled at fixed meters ahead
(`obs.lookahead_spacing_m`, replacing `obs.lookahead_stride`) and linearly
interpolated along the waypoint tangent, which also bridges coarse segments
such as `Straight_track`'s 5.7 m closing hop.

**Default calibrated for continuity:** `lookahead_spacing_m = 0.45` reproduces
`reinvent_base`'s pre-fix horizon exactly (0.15 m spacing × stride 3), so
features on the reference track are essentially unchanged while every other
track now sees the same 4.5 m (k=10) horizon.

The index-based `Track.lookahead` remains public (tools may want raw indices)
with a docstring caution; nothing in the package observes it anymore.

## Alternatives

- **Per-track stride `round(target_m / spacing)`** — keeps quantization to
  waypoint resolution (useless on 0.8 m-spaced tracks) and still varies the
  realized horizon.
- **Resample all routes to uniform spacing at bake time** — heaviest option:
  regenerates 298 assets and invalidates waypoint counts the catalog and
  tests pin; interpolation at read time gets the same effect for free.
- **A spec field for the spacing** — deferred; the cfg knob
  (`obs.lookahead_spacing_m`) matches the status the old stride had, and spec
  exposure can ride the prune-when-default pattern later if HPO wants it.

## Consequences

- Feature semantics changed for every non-reference track: runs trained
  before this ADR are not feature-compatible with runs after it (same
  precedent as the `off_track` sign fix — content hashes do not encode code
  behavior; runs always retrain).
- The `state` vector's meaning is now track-independent, removing a hidden
  train/holdout distribution shift.
- `tests/test_lookahead_arclength.py` pins horizon constancy across an 8×
  spacing spread, interpolation (no snapped duplicates), reverse-direction
  walks, and finish-line wrap.
