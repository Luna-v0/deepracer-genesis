"""The track zoo: pre-compiled scene variants as the domain randomization.

The model (decided 2026-08-26): declare a zoo of track variants — base shape x
width x road/line palette x field colour — **compile** it down to baked assets
once, **see** the compiled result as a grid of track instances in one Genesis
scene, then **plumb** the same assets into the real simulator
(``CameraEnvironment(tracks=zoo_names)``), where the randomization simply IS
cars living on those instances (per env, fixed per run; variety comes from the
zoo's size). Runtime obs-side knobs stack on top unchanged.

Three commands (``python -m deepracer_genesis.tools.zoo <cmd>``):

- ``compile`` — manifest -> linted, baked, registered tracks (cached).
- ``view``    — a bare ``gs.Scene`` testing ground: every variant on its world
  tile, interactive viewer (or ``--screenshot`` for headless boxes). No env,
  no RL — scene composition only.
- ``watch``   — the same zoo under the real ``DeepRacerEnv``: cars driven by
  the scripted centerline follower, optional GUI viewer, periodic saved
  car-view photos (which carry the full obs DR the viewer cannot show).

Constraints are enforced at the earliest layer that can know them: the
manifest cannot express per-tile sky/lighting (scene-global under Madrona —
illegal states are unrepresentable), the compiler lints palette contrast
(a centerline that vanishes against the road destroys the driving signal),
and ``spec.validate()`` + the verify scripts cover the rest downstream.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

from .. import ASSETS_DIR
from .track_builder import (GENERATED_DIR, _PALETTE, install_track,
                            scale_route_width)

# named palettes (0-255 RGB per material); "classic" is the repo default
PALETTES: dict[str, dict] = {
    "classic": dict(_PALETTE),
    "asphalt_light": {"road": (96, 99, 106), "border": (245, 245, 245),
                      "centerline": (250, 205, 60)},
    "dusk": {"road": (30, 30, 38), "border": (200, 200, 210),
             "centerline": (255, 140, 40)},
    "faded": {"road": (70, 68, 62), "border": (190, 185, 170),
              "centerline": (200, 170, 90)},
}

# named per-tile field (ground quad) colours, 0-255 RGB
FIELDS: dict[str, tuple] = {
    "grass": (77, 122, 82),
    "sand": (194, 178, 128),
    "concrete": (150, 150, 150),
}

# named perimeter-wall colours, 0-255 RGB (real DeepRacer venues run white
# barriers — the camera learns them, which is exactly why they belong in DR)
WALLS: dict[str, tuple] = {
    "white": (240, 240, 240),
    "grey": (150, 150, 150),
    "dark": (60, 60, 65),
    "red": (185, 60, 50),
}

# minimum luminance separation (0-255 scale) between the road and each line
# colour — below this, DR destroys the driving signal instead of hardening
# the policy (the REFACTOR_PLAN palette caution, enforced at bake time)
MIN_LINE_CONTRAST = 60.0

# minimum RGB colour distance between the road and a baked per-tile field —
# below this the road camouflages against its own ground (measured: a
# grey-on-grey pair at distance ~24 was invisible; clear pairs start ~60)
MIN_GROUND_CONTRAST = 50.0


@dataclass(frozen=True)
class TrackVariant:
    """One pre-compiled track instance of a zoo.

    Only per-tile-capable axes exist here BY CONSTRUCTION — sky and lighting
    are scene-global under Madrona/rasterizer, so the manifest cannot ask for
    them per tile.

    Attributes:
        base: Source track name (its route supplies the centerline).
        width: Width scale about the centerline (1.0 = as authored).
        palette: Road/border/centerline colours — a :data:`PALETTES` name, an
            explicit 0-255 RGB dict, or None for the default look.
        field: Per-tile ground colour baked into the mesh — a :data:`FIELDS`
            name, an explicit 0-255 RGB tuple, or None for the global plane.
        wall: Perimeter wall around the tile — a :data:`WALLS` name, an
            explicit 0-255 RGB tuple, or None for no wall. Visual-only (the
            rulebook is the actual fence), standing at the field-quad edge.
    """

    base: str
    width: float = 1.0
    palette: Union[str, dict, None] = None
    field: Union[str, tuple, None] = None
    wall: Union[str, tuple, None] = None


@dataclass(frozen=True)
class OfficialSample:
    """Declarative source: a seeded sample of ORIGINAL DeepRacer tracks.

    Expanded by :func:`compile_zoo` (declaration stays pure; fetching and
    baking happen at compile time, like the ``>>`` DSL's build step).

    Attributes:
        n: How many tracks to sample (None = the whole library).
        seed: Sampling + noise + look seed (the population's identity).
        jitter: Waypoint-noise amplitude in metres (0 = pristine geometry).
        clones: Noised clones per sampled track.
        looks: Randomize each clone's palette/field/wall.
        keep_original: Also include each pristine original.
        loops_only: Skip open (non-loop) courses like ``Straight_track``.
        fetch: Download missing tracks (False = locally installed only).
        names: Explicit name pool (None = the full official library).
    """

    n: Optional[int] = None
    seed: int = 0
    jitter: float = 0.4
    clones: int = 1
    looks: bool = True
    keep_original: bool = False
    loops_only: bool = True
    fetch: bool = True
    names: Optional[tuple] = None


@dataclass(frozen=True)
class RandomShapes:
    """Declarative source: fully synthetic procedural circuits.

    Attributes:
        n: Number of shapes to spawn.
        seed: Determinism seed (part of every variant's name).
        size: Approximate track diameter in metres.
        half_width_range: Road half-width sampled per variant (metres).
        field_prob: Probability of a per-tile field.
        wall_prob: Probability of a perimeter wall.
    """

    n: int
    seed: int = 0
    size: float = 14.0
    half_width_range: tuple = (0.42, 0.62)
    field_prob: float = 0.5
    wall_prob: float = 0.6


#: anything a Zoo manifest may declare in ``variants``
Source = Union[TrackVariant, OfficialSample, RandomShapes]


@dataclass(frozen=True)
class Zoo:
    """A named collection of track sources (the compiled DR population).

    A manifest is config-as-code: declare it in a Python file and hand the
    file to the CLI — ``python -m deepracer_genesis.tools.zoo watch
    my_zoos.py:population``. Sources may mix explicit variants, official
    samples, and synthetic shapes; expansion order is declaration order.

    Attributes:
        name: Zoo name (bookkeeping only; variants get their own names).
        variants: Explicit :class:`TrackVariant` entries and/or declarative
            sources (:class:`OfficialSample`, :class:`RandomShapes`).
        grid_spacing: Metres between world tiles for ``watch``; None = the
            compact-but-safe automatic spacing (:func:`near_spacing`).
    """

    name: str
    variants: tuple[Source, ...]
    grid_spacing: Optional[float] = None


def _luminance(rgb: Sequence[float]) -> float:
    """Relative luminance of a 0-255 RGB colour (0-255 scale).

    Args:
        rgb: The colour.

    Returns:
        ``0.2126 R + 0.7152 G + 0.0722 B``.
    """
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _resolve(variant: TrackVariant) -> tuple[Optional[dict], Optional[tuple],
                                             Optional[tuple]]:
    """Resolve a variant's palette/field/wall references to explicit values.

    Args:
        variant: The variant being compiled.

    Returns:
        ``(palette_dict_or_None, field_rgb_or_None, wall_rgb_or_None)``.

    Raises:
        ValueError: On an unknown palette/field/wall name or malformed values.
    """
    pal = variant.palette
    if isinstance(pal, str):
        if pal not in PALETTES:
            raise ValueError(f"unknown palette {pal!r}; known: {sorted(PALETTES)}")
        pal = PALETTES[pal]
    if pal is not None:
        bad = set(pal) - set(_PALETTE)
        if bad:
            raise ValueError(f"palette keys {sorted(bad)} unknown; "
                             f"allowed: {sorted(_PALETTE)}")
    field = variant.field
    if isinstance(field, str):
        if field not in FIELDS:
            raise ValueError(f"unknown field {field!r}; known: {sorted(FIELDS)}")
        field = FIELDS[field]
    wall = variant.wall
    if isinstance(wall, str):
        if wall not in WALLS:
            raise ValueError(f"unknown wall {wall!r}; known: {sorted(WALLS)}")
        wall = WALLS[wall]
    return (dict(pal) if pal else None,
            tuple(field) if field is not None else None,
            tuple(wall) if wall is not None else None)


def lint_variant(variant: TrackVariant) -> None:
    """Refuse a variant whose look would destroy the driving signal.

    Args:
        variant: The variant being compiled.

    Raises:
        ValueError: If the width is not positive, or the (resolved) palette's
            centerline/border luminance sits closer than
            :data:`MIN_LINE_CONTRAST` to the road — the lines the policy
            steers by must stay visible under every variant.
    """
    if not variant.width > 0:
        raise ValueError(f"width must be > 0; got {variant.width}")
    pal, field, _wall = _resolve(variant)
    merged = {**_PALETTE, **(pal or {})}
    road_l = _luminance(merged["road"])
    for line in ("centerline", "border"):
        d = abs(_luminance(merged[line]) - road_l)
        if d < MIN_LINE_CONTRAST:
            raise ValueError(
                f"palette rejected: |luminance({line}) - luminance(road)| = "
                f"{d:.0f} < {MIN_LINE_CONTRAST:.0f} — the {line} must stay "
                "visible against the road or the variant erases the signal "
                "the policy steers by")
    if field is not None:
        d = float(np.linalg.norm(np.asarray(merged["road"], dtype=float)
                                 - np.asarray(field, dtype=float)))
        if d < MIN_GROUND_CONTRAST:
            raise ValueError(
                f"field rejected: road-vs-field colour distance {d:.0f} < "
                f"{MIN_GROUND_CONTRAST:.0f} — a road camouflaged against its "
                "own ground erases the track (empirical: grey-on-grey at "
                "distance ~24 is invisible)")


def variant_name(variant: TrackVariant) -> str:
    """Deterministic registered-track name for a variant.

    Args:
        variant: The variant.

    Returns:
        The base name for an all-default variant; otherwise the base plus
        ``_wNNN`` (width), a palette tag (its :data:`PALETTES` name or a
        6-hex content hash), a field tag, and/or a wall tag.
    """
    pal, field, wall = _resolve(variant)
    parts = [variant.base]
    if abs(variant.width - 1.0) >= 1e-9:
        parts.append(f"w{round(variant.width * 100):03d}")
    if pal is not None:
        if isinstance(variant.palette, str):
            parts.append(variant.palette)
        else:
            digest = hashlib.sha1(
                json.dumps(sorted(pal.items())).encode()).hexdigest()[:6]
            parts.append(f"p{digest}")
    if field is not None:
        if isinstance(variant.field, str):
            parts.append(f"f{variant.field}")
        else:
            digest = hashlib.sha1(repr(tuple(field)).encode()).hexdigest()[:6]
            parts.append(f"f{digest}")
    if wall is not None:
        if isinstance(variant.wall, str):
            parts.append(f"b{variant.wall}")
        else:
            digest = hashlib.sha1(repr(tuple(wall)).encode()).hexdigest()[:6]
            parts.append(f"b{digest}")
    return "_".join(parts)


def compile_zoo(zoo: Zoo, *, force: bool = False) -> tuple[str, ...]:
    """Lint and bake every variant; return the registered track names.

    Baking is cached by the deterministic names (rebake with ``force``);
    an all-default variant reuses its base track without duplicating assets.
    The returned tuple plumbs straight into training:
    ``CameraEnvironment(tracks=compile_zoo(my_zoo))``.

    Args:
        zoo: The zoo manifest.
        force: Rebake variants whose assets already exist.

    Returns:
        One registered track name per variant, in manifest order.

    Raises:
        ValueError: From the lint (bad width, unknown/low-contrast palette).
        KeyError: If a variant's base track is not registered.
    """
    from ..envs.track import TRACKS

    names: list[str] = []
    for item in zoo.variants:
        if isinstance(item, OfficialSample):
            names.extend(_expand_official(item, force=force))
            continue
        if isinstance(item, RandomShapes):
            names.extend(_expand_random(item, force=force))
            continue
        variant = item
        lint_variant(variant)
        pal, field, wall = _resolve(variant)
        name = variant_name(variant)
        if name == variant.base:
            names.append(name)
            continue
        _mesh, route_rel, _f = TRACKS[variant.base]
        track_dir = os.path.join(GENERATED_DIR, name)
        if force or not os.path.exists(os.path.join(track_dir, "route.npy")):
            route = np.load(os.path.join(ASSETS_DIR, route_rel))
            install_track(name, scale_route_width(route, variant.width),
                          palette=pal, field=field, wall=wall)
        elif name not in TRACKS:
            rel = os.path.relpath(track_dir, ASSETS_DIR)
            TRACKS[name] = (f"{rel}/track.obj", f"{rel}/route.npy", None)
        names.append(name)
    return tuple(names)


def zoo_extent(names: Sequence[str]) -> float:
    """Largest track footprint (metres) across a set of registered tracks.

    Args:
        names: Registered track names.

    Returns:
        The max bounding-box side over all outer borders.
    """
    from ..envs.track import TRACKS

    extent = 0.0
    for name in names:
        r = np.load(os.path.join(ASSETS_DIR, TRACKS[name][1]))
        extent = max(extent, float(max(r[:, 4:6].max(0) - r[:, 4:6].min(0))))
    return extent


def near_spacing(names: Sequence[str]) -> float:
    """The compact-but-safe tile spacing for a zoo under the real env.

    The onboard batch cameras clip at a 20 m far plane, so a neighboring
    tile is invisible as soon as its geometry sits beyond that — the default
    100 m is far more conservative than isolation needs. Verified empirically
    (same pose renders bit-identically at this spacing and at 100 m).

    Args:
        names: Registered track names of the zoo.

    Returns:
        ``extent + 26`` metres (neighbor gap ≥ the 20 m far plane + margin).
    """
    return zoo_extent(names) + 26.0


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    """Convex hull of 2D points (Andrew monotone chain), CCW order.

    Args:
        pts: ``(M, 2)`` points.

    Returns:
        ``(K, 2)`` hull vertices, counter-clockwise.
    """
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def _chain(seq):
        h: list = []
        for p in seq:
            while len(h) >= 2:
                a, b = h[-1] - h[-2], p - h[-2]
                if a[0] * b[1] - a[1] * b[0] > 0:      # 2D cross, left turn
                    break
                h.pop()
            h.append(p)
        return h[:-1]

    return np.array(_chain(pts) + _chain(pts[::-1]))


def _self_intersects(poly: np.ndarray) -> bool:
    """Whether a closed polygon's non-adjacent edges cross.

    Chaikin corner-cutting preserves simplicity, so checking the control
    polygon is enough to guarantee a non-crossing smoothed centerline.

    Args:
        poly: ``(K, 2)`` closed control polygon.

    Returns:
        True if any two non-adjacent edges intersect.
    """
    k = len(poly)
    a, b = poly, np.roll(poly, -1, axis=0)

    def _ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])

    for i in range(k):
        for j in range(i + 2, k):
            if i == 0 and j == k - 1:
                continue                      # adjacent around the loop
            p1, p2, p3, p4 = a[i], b[i], a[j], b[j]
            if (_ccw(p1, p3, p4) != _ccw(p2, p3, p4)
                    and _ccw(p1, p2, p3) != _ccw(p1, p2, p4)):
                return True
    return False


def _corridor_clear(center: np.ndarray, half_width: float) -> bool:
    """Whether far-apart track sections keep road-width clearance.

    Args:
        center: ``(W, 2)`` centerline waypoints.
        half_width: Road half-width in metres.

    Returns:
        True when no two sections more than an eighth of the lap apart come
        closer than ``2.9 x half_width`` (roads must never overlap).
    """
    w = len(center)
    d = np.linalg.norm(center[:, None] - center[None], axis=2)
    idx = np.abs(np.arange(w)[:, None] - np.arange(w)[None])
    far = np.minimum(idx, w - idx) > w // 8
    return bool(d[far].min() >= 2.9 * half_width)


def _random_look(rng: np.random.Generator, *, field_prob: float = 0.5,
                 wall_prob: float = 0.6) -> tuple:
    """One seeded appearance draw: palette + maybe field + maybe wall.

    The shared look-randomization used by the random AND official zoos —
    hue-preserving palette jitter (asphalt stays asphalt), a per-tile field
    with ``field_prob``, a perimeter wall with ``wall_prob``.

    Args:
        rng: Seeded generator (identity: same seed = same look).
        field_prob: Probability of a per-tile ground quad.
        wall_prob: Probability of a perimeter wall.

    Returns:
        ``(palette_dict, field_rgb_or_None, wall_rgb_or_None)`` — always a
        combination the lint accepts (road never camouflaged against its
        field; a stubborn draw drops the field rather than shipping one).
    """
    wall = None
    if rng.uniform() < wall_prob:
        wall = _jitter_color(
            rng, WALLS[sorted(WALLS)[int(rng.integers(len(WALLS)))]],
            value=(0.85, 1.15), tint=8)
    for _ in range(30):
        pal = _random_palette(rng)
        field = None
        if rng.uniform() < field_prob:
            field = _jitter_color(
                rng, FIELDS[sorted(FIELDS)[int(rng.integers(len(FIELDS)))]],
                value=(0.8, 1.25), tint=10)
        try:
            lint_variant(TrackVariant("_probe", palette=pal, field=field))
            return pal, field, wall
        except ValueError:
            continue
    return dict(PALETTES["classic"]), None, wall


def _random_route(rng: np.random.Generator, size: float,
                  half_width: float) -> np.ndarray:
    """One random drivable closed CIRCUIT (straights, bends, hairpins).

    The classic procedural-track construction: scatter points in a random-
    aspect box, take the convex hull (its edges become straights), punch a
    random subset of edge midpoints inward (bends up to hairpins), reject
    crossings and road-overlap, smooth, and gate on drivability. An adaptive
    schedule tames the inward punches over the retries so convergence is
    guaranteed (the plain hull always passes).

    Args:
        rng: Seeded generator (determinism = zoo identity).
        size: Approximate track diameter in metres.
        half_width: Half road width in metres.

    Returns:
        A ``(W, 6)`` route array.

    Raises:
        RuntimeError: If no drivable shape is found in 80 draws.
    """
    from .track_builder import build_route, track_metrics

    for attempt in range(80):
        t = attempt / 79.0
        aspect = float(rng.uniform(0.55, 1.0))
        rot = float(rng.uniform(0.0, np.pi))
        pts = rng.uniform(-0.5, 0.5, (int(rng.integers(12, 22)), 2)) \
            * np.array([size, size * aspect])
        hull = _convex_hull(pts)
        if len(hull) < 5:
            continue
        centroid = hull.mean(0)
        amp_hi = 0.40 - 0.32 * t              # inward punch depth (tames)
        punch_p = 0.65 - 0.45 * t
        poly: list = []
        for i in range(len(hull)):
            a, b = hull[i], hull[(i + 1) % len(hull)]
            poly.append(a)
            edge = float(np.linalg.norm(b - a))
            if edge > 4.5 * half_width and rng.uniform() < punch_p:
                mid = (a + b) / 2.0
                inward = centroid - mid
                inward /= max(float(np.linalg.norm(inward)), 1e-9)
                poly.append(mid + inward * edge * rng.uniform(0.12, max(amp_hi, 0.13)))
        control = np.array(poly)
        # drop control points that crowd (they force undrivable kinks)
        keep = [0]
        for i in range(1, len(control)):
            if np.linalg.norm(control[i] - control[keep[-1]]) > 3.0 * half_width:
                keep.append(i)
        control = control[keep]
        if len(control) < 5 or _self_intersects(control):
            continue
        c, s = np.cos(rot), np.sin(rot)
        control = control @ np.array([[c, -s], [s, c]]).T
        # smoothing scales with the schedule: hull corners can be arbitrarily
        # pointy (unlike ellipse samples), and each Chaikin pass roughly
        # doubles a corner's radius
        route = build_route(control, half_width=half_width, n_waypoints=150,
                            smooth_passes=2 + int(3 * t))
        if track_metrics(route)["min_turn_radius_m"] >= 2.2 * half_width \
                and _corridor_clear(route[:, 0:2], half_width):
            return route
    # guaranteed fallback: a gently jittered near-circle always clears the
    # bar for sane (size, half_width); keeps population spawning unstoppable
    for _ in range(20):
        ang = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        radius = (size / 2.0) * rng.uniform(0.82, 1.0, 12)
        pts = np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)
        route = build_route(pts, half_width=half_width, n_waypoints=150,
                            smooth_passes=4)
        if track_metrics(route)["min_turn_radius_m"] >= 2.2 * half_width:
            return route
    raise RuntimeError(
        f"no drivable random circuit (size={size:.1f} m, "
        f"half_width={half_width:.2f} m needs min turn radius "
        f">= {2.2 * half_width:.2f} m) — widen size or narrow half_width")


def _jitter_color(rng: np.random.Generator, rgb: Sequence[int], *,
                  value: tuple = (0.75, 1.3), tint: int = 12) -> tuple:
    """Hue-preserving colour jitter: shared brightness scale + a small tint.

    Independent per-channel jitter turns dark greys into saturated purples
    and navies (large jitter relative to small base values); scaling all
    three channels together and adding only a small tint keeps asphalt
    looking like asphalt while still varying brightness and cast.

    Args:
        rng: Seeded generator.
        rgb: Base 0-255 colour.
        value: Brightness-scale range shared by all channels.
        tint: Max per-channel additive cast.

    Returns:
        The jittered 0-255 colour.
    """
    v = float(rng.uniform(*value))
    cast = rng.integers(-tint, tint + 1, 3)
    return tuple(int(np.clip(c * v + cast[j], 0, 255))
                 for j, c in enumerate(rgb))


def _random_palette(rng: np.random.Generator) -> dict:
    """A contrast-safe random palette (a named palette, realistically varied).

    Args:
        rng: Seeded generator.

    Returns:
        A 0-255 RGB palette dict passing the line-contrast lint.
    """
    names = sorted(PALETTES)
    for _ in range(40):
        base = PALETTES[names[int(rng.integers(len(names)))]]
        pal = {mat: _jitter_color(rng, rgb) for mat, rgb in base.items()}
        try:
            lint_variant(TrackVariant("_probe", palette=pal))
            return pal
        except (ValueError, KeyError):
            continue
    return dict(PALETTES["classic"])


def random_zoo(n: int, *, seed: int = 0, size: float = 14.0,
               half_width_range: tuple = (0.42, 0.62),
               field_prob: float = 0.5, wall_prob: float = 0.6) -> Zoo:
    """Spawn a large zoo of random track variations, deterministically.

    Every variant gets its own random SHAPE (drivability-linted), road width,
    contrast-linted jittered palette, (sometimes) a per-tile field, and
    (sometimes) a perimeter wall — baked directly at install, so the returned
    zoo compiles instantly. Same seed = same zoo, byte for byte; assets are
    cached by name.

    Args:
        n: Number of variants to spawn.
        seed: Determinism seed (part of every variant's name).
        size: Approximate track diameter in metres (14 = reinvent-ish; keep
            camera-scale — tiny room tracks are nearly invisible to the
            onboard camera at the stock mount pitch).
        half_width_range: Road half-width sampled per variant (metres).
        field_prob: Probability a variant gets a random per-tile field.
        wall_prob: Probability a variant gets a random perimeter wall.

    Returns:
        The zoo (all-default variants over the pre-baked shape tracks —
        ``compile_zoo`` registers without rebaking).
    """
    src = RandomShapes(n, seed=seed, size=size,
                       half_width_range=half_width_range,
                       field_prob=field_prob, wall_prob=wall_prob)
    return Zoo(f"random{seed}",
               tuple(TrackVariant(name) for name in _expand_random(src)))


def _expand_random(src: RandomShapes, *, force: bool = False) -> list[str]:
    """Expand a :class:`RandomShapes` source into registered track names.

    Args:
        src: The declarative source.
        force: Rebake shapes whose assets already exist.

    Returns:
        The ``rz{seed}_{i}`` track names, in index order.
    """
    from ..envs.track import TRACKS

    rng = np.random.default_rng(src.seed)
    names: list[str] = []
    for i in range(src.n):
        name = f"rz{src.seed}_{i:02d}"
        # draw EVERYTHING regardless of cache so variant i is seed-stable
        sz = src.size * float(rng.uniform(0.75, 1.3))  # per-variant footprint
        hw = float(rng.uniform(*src.half_width_range))
        pal, field, wall = _random_look(rng, field_prob=src.field_prob,
                                        wall_prob=src.wall_prob)
        shape_rng = np.random.default_rng((src.seed, i))
        track_dir = os.path.join(GENERATED_DIR, name)
        if force or not os.path.exists(os.path.join(track_dir, "route.npy")):
            install_track(name, _random_route(shape_rng, sz, hw),
                          palette=pal, field=field, wall=wall)
        elif name not in TRACKS:
            rel = os.path.relpath(track_dir, ASSETS_DIR)
            TRACKS[name] = (f"{rel}/track.obj", f"{rel}/route.npy", None)
        names.append(name)
    return names


def _perturb_route(route: np.ndarray, rng: np.random.Generator,
                   amplitude: float) -> np.ndarray:
    """A waypoint-noised clone of a real route, no worse to drive.

    Displaces the centerline along its local normals by a smooth, periodic
    Fourier noise (3 low-frequency components along arclength — white noise
    would make jagged, undrivable centerlines), rebuilds the borders with the
    ORIGINAL per-waypoint width profile, and gates the result RELATIVE to the
    original track's own metrics (real hairpins may already be tighter than
    any absolute bar): the clone must keep >= 80% of the original's minimum
    turn radius and corridor clearance, and must not self-intersect. On
    failure the amplitude halves; at zero the original itself returns, so
    this can never exhaust.

    Args:
        route: ``(W, 6)`` original route.
        rng: Seeded generator (determinism = variant identity).
        amplitude: Max lateral displacement in metres.

    Returns:
        The perturbed ``(W, 6)`` route (or the original when even tiny noise
        breaks the gates).
    """
    from .track_builder import track_metrics

    r = np.asarray(route, dtype=np.float64)
    if np.allclose(r[0, :2], r[-1, :2], atol=1e-6):     # drop closing dup
        r = r[:-1]
    center = r[:, 0:2]
    hw = 0.5 * np.linalg.norm(r[:, 4:6] - r[:, 2:4], axis=1)
    w = len(center)
    seg = np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg[:-1])]) / max(seg.sum(), 1e-9)

    def _metrics(rt: np.ndarray) -> tuple[float, float]:
        m = track_metrics(rt)["min_turn_radius_m"]
        c = rt[:, 0:2]
        d = np.linalg.norm(c[:, None] - c[None], axis=2)
        idx = np.abs(np.arange(len(c))[:, None] - np.arange(len(c))[None])
        far = np.minimum(idx, len(c) - idx) > len(c) // 8
        return float(m), float(d[far].min())

    base_radius, base_corridor = _metrics(r)
    # draw the noise ONCE (identity: same seed = same clone), tame by halving
    freqs = rng.integers(1, 6, 3)
    phases = rng.uniform(0.0, 2.0 * np.pi, 3)
    amps = rng.uniform(0.4, 1.0, 3)
    noise = sum(a * np.sin(2.0 * np.pi * f * s + p)
                for f, p, a in zip(freqs, phases, amps))
    noise = noise / max(np.abs(noise).max(), 1e-9)

    amp = amplitude
    for _ in range(8):
        if amp < 1e-3:
            return route
        tangent = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
        c2 = center + normal * (noise * amp)[:, None]
        t2 = np.roll(c2, -1, axis=0) - np.roll(c2, 1, axis=0)
        t2 /= np.maximum(np.linalg.norm(t2, axis=1, keepdims=True), 1e-9)
        n2 = np.stack([-t2[:, 1], t2[:, 0]], axis=1)
        cand = np.concatenate([c2, c2 + n2 * hw[:, None],
                               c2 - n2 * hw[:, None]], axis=1)
        radius, corridor = _metrics(cand)
        if radius >= 0.8 * base_radius and corridor >= 0.8 * base_corridor \
                and not _self_intersects(c2[:: max(1, w // 80)]):
            return cand
        amp *= 0.5
    return route


def official_zoo(n: Optional[int] = None, *, seed: int = 0,
                 names: Optional[Sequence[str]] = None,
                 fetch: bool = True, loops_only: bool = True,
                 jitter: float = 0.0, clones: int = 1,
                 keep_original: bool = False, looks: bool = False) -> Zoo:
    """A zoo of ORIGINAL official DeepRacer tracks (no generated shapes).

    The long list the physical car's world actually comes from. The listing
    and per-track metadata live in the track catalog
    (:mod:`deepracer_genesis.tracks` — ``official()``, ``info()``); this
    function only samples, fetches on demand, and filters. Tracks whose
    fetch fails are skipped with a warning rather than sinking the
    population.

    Args:
        n: Sample this many tracks (seeded, without replacement); None = all.
        seed: Sampling seed (irrelevant when ``n`` is None).
        names: Explicit name list (default: the full official library).
        fetch: Download missing tracks; False = use only locally installed
            ones (offline mode).
        loops_only: Skip open (non-loop) courses like ``Straight_track`` —
            lap logic and training zoos usually want closed circuits.
        jitter: Max waypoint-noise displacement in metres (> 0 bakes a
            smoothly-perturbed clone of each sampled track via
            :func:`_perturb_route`; 0 = the pristine originals).
        clones: Noised clones per sampled track (only when ``jitter`` > 0).
        keep_original: Also include each pristine original next to its
            clones.
        looks: Also randomize each clone's appearance (jittered palette,
            probabilistic per-tile field and perimeter wall — the shared
            :func:`_random_look` draw), folded into the same seed identity.

    Returns:
        The zoo of plain :class:`TrackVariant` entries — add width/palette/
        wall axes by wrapping names in your own manifest if wanted.
    """
    src = OfficialSample(n=n, seed=seed, jitter=jitter, clones=clones,
                         looks=looks, keep_original=keep_original,
                         loops_only=loops_only, fetch=fetch,
                         names=tuple(names) if names is not None else None)
    return Zoo(f"official{n if n is not None else ''}",
               tuple(TrackVariant(name) for name in _expand_official(src)))


def _expand_official(src: OfficialSample, *, force: bool = False) -> list[str]:
    """Expand an :class:`OfficialSample` into registered track names.

    Args:
        src: The declarative sample.
        force: Rebake clones whose assets already exist.

    Returns:
        Track names, sample-sorted (skips announced on stdout).
    """
    from .. import tracks as track_catalog
    from .track_builder import fetch_official_track

    pool = (list(src.names) if src.names is not None
            else list(track_catalog.official()))
    if src.n is not None and src.n < len(pool):
        rng = np.random.default_rng(src.seed)
        pool = sorted(rng.choice(pool, size=src.n, replace=False).tolist())
    usable: list[str] = []
    for name in pool:
        if not track_catalog.exists(name):
            if not src.fetch:
                print(f"  (skipping {name}: not installed, fetch disabled)")
                continue
            try:
                fetch_official_track(name)
            except Exception as e:                      # noqa: BLE001
                print(f"  (skipping {name}: {e})")
                continue
        if src.loops_only and not track_catalog.info(name).closed:
            print(f"  (skipping {name}: open course, not a loop)")
            continue
        usable.append(name)
    if src.jitter > 0 or src.looks:
        noised: list[str] = []
        for name in usable:
            if src.keep_original:
                noised.append(name)
            base_route = None
            for c in range(max(1, src.clones)):
                tag = f"{name}_n{src.seed}" + (f"c{c}" if src.clones > 1 else "")
                if force or not track_catalog.exists(tag):
                    if base_route is None:
                        base_route = np.load(track_catalog.info(name).route_path)
                    key = int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)
                    clone_rng = np.random.default_rng((src.seed, key, c))
                    route = (_perturb_route(base_route, clone_rng, src.jitter)
                             if src.jitter > 0 else base_route)
                    pal = field = wall = None
                    if src.looks:
                        look_rng = np.random.default_rng((src.seed, key, c, 7))
                        pal, field, wall = _random_look(look_rng)
                    install_track(tag, route, palette=pal, field=field,
                                  wall=wall)
                noised.append(tag)
        usable = noised
    return usable


def demo_zoo(base: str = "reinvent_base") -> Zoo:
    """The built-in demo: widths x palettes plus a field variant.

    Camera-scale by default: reinvent-sized variants fill the onboard frame.
    (A tiny room track like ``donut_track`` is nearly invisible to the onboard
    camera at the stock 10-degree mount pitch — the visible ground starts
    beyond the whole ring; the zoo's watch mode is how that gets caught.)

    Args:
        base: Source track for every variant.

    Returns:
        A 6-variant zoo (2x3 tile grid when viewed).
    """
    return Zoo("demo", (
        TrackVariant(base),
        TrackVariant(base, width=0.9, wall="white"),
        TrackVariant(base, width=1.15),
        TrackVariant(base, palette="dusk", wall="dark"),
        TrackVariant(base, palette="asphalt_light", field="sand", wall="white"),
        TrackVariant(base, width=1.15, palette="faded", field="concrete"),
    ))


# ----------------------------------------------------------------- see / run
def view_zoo(names: Sequence[str], *, grid_spacing: Optional[float] = None,
             screenshot: Optional[str] = None, seconds: Optional[float] = None,
             res: tuple = (1280, 960)) -> None:
    """Open the bare-scene testing ground: every variant on its world tile.

    No env, no cars, no RL — a ``gs.Scene`` with the compiled meshes on a
    tile grid, either in the interactive viewer or rendered top-down to a PNG
    for headless use. The view grid auto-compacts to the tracks' footprint
    (training's 100 m spacing exists to keep onboard cameras isolated; a
    bare-scene view has none, and at 100 m small tracks are specks).

    Args:
        names: Registered track names (from :func:`compile_zoo`).
        grid_spacing: Metres between tiles; None = auto (1.5x the largest
            track footprint). Pass the training value to preview that layout.
        screenshot: PNG path — render one top-down overview headlessly
            instead of opening the viewer window.
        seconds: Auto-close the viewer after this long (None = until Ctrl+C).
        res: Screenshot/overview camera resolution.

    Raises:
        KeyError: If a name is not a registered track.
    """
    import genesis as gs

    from .._gs import ensure_init
    from ..envs.track import TRACKS, grid_offsets

    ensure_init("gpu")
    if grid_spacing is None:
        # room for the baked field quads/walls (track + margin) plus a gap
        grid_spacing = zoo_extent(names) * 1.45 + 0.5
    offsets = grid_offsets(len(names), grid_spacing, "cpu").tolist()
    lo = np.array([np.inf, np.inf])
    hi = -lo
    for k, name in enumerate(names):
        r = np.load(os.path.join(ASSETS_DIR, TRACKS[name][1]))
        lo = np.minimum(lo, r[:, 4:6].min(0) + offsets[k])
        hi = np.maximum(hi, r[:, 4:6].max(0) + offsets[k])
    c = (lo + hi) / 2.0
    half = float(max(hi - lo) / 2.0) + 5.0
    # open the interactive viewer already ABOVE the grid, looking at all
    # tiles at once (slight tilt so orbiting starts from a stable pose)
    vh = half / 0.466                      # fov 50: half-extent = h*tan(25 deg)
    scene = gs.Scene(
        vis_options=gs.options.VisOptions(
            shadow=False, ambient_light=(0.35, 0.35, 0.35),
            background_color=(0.55, 0.72, 0.9)),
        show_viewer=screenshot is None,
        viewer_options=gs.options.ViewerOptions(
            realtime_factor=1.0, camera_fov=50,
            camera_pos=(float(c[0]), float(c[1]) - 0.15 * vh, vh),
            camera_lookat=(float(c[0]), float(c[1]), 0.0)),
    )
    scene.add_entity(gs.morphs.Plane(pos=(0, 0, -0.001)),
                     surface=gs.surfaces.Rough(color=(0.30, 0.48, 0.32, 1.0)))
    for k, name in enumerate(names):
        mesh_rel, _route, _f = TRACKS[name]
        scene.add_entity(gs.morphs.Mesh(
            file=os.path.join(ASSETS_DIR, mesh_rel),
            pos=(offsets[k][0], offsets[k][1], 0.0),
            fixed=True, collision=False))
    cam = None
    if screenshot is not None:
        c = (lo + hi) / 2.0
        half = float(max(hi - lo) / 2.0) + 5.0
        height = half / 0.577            # fov 60: half-extent = h * tan(30°)
        cam = scene.add_camera(
            res=tuple(res), pos=(float(c[0]), float(c[1]), height),
            lookat=(float(c[0]), float(c[1]), 0.0), up=(0.0, 1.0, 0.0),
            fov=60, GUI=False, far=height + 50.0)
    scene.build()
    if cam is not None:
        from PIL import Image
        rgb = np.asarray(cam.render(rgb=True)[0])
        os.makedirs(os.path.dirname(screenshot) or ".", exist_ok=True)
        Image.fromarray(rgb).save(screenshot)
        print(f"zoo overview -> {screenshot}")
        return
    import time
    print(f"viewing {len(names)} tiles — Ctrl+C to close"
          + (f" (auto-close in {seconds:.0f}s)" if seconds else ""))
    t0 = time.time()
    try:
        while seconds is None or time.time() - t0 < seconds:
            scene.step()
    except KeyboardInterrupt:
        pass


def watch_zoo(names: Sequence[str], *, num_envs: int = 12, gui: bool = True,
              steps: Optional[int] = None, photos_every: int = 100,
              photo_envs: int = 4, out: str = "logs/zoo",
              randomize: bool = False,
              grid_spacing: Optional[float] = None) -> None:
    """Run cars on the zoo — the training world, watchable.

    Builds the real ``DeepRacerEnv`` on the compiled tracks (spawn
    randomization ON, exactly as training has it), drives every car with the
    scripted centerline follower, and periodically saves a few car-view
    photos (``obs_image_buf`` — the policy's frames, which carry any obs DR
    the viewer window cannot show).

    Args:
        names: Registered track names (from :func:`compile_zoo`).
        num_envs: Parallel cars, balanced across the tiles.
        gui: Open the interactive viewer window (needs a display).
        steps: Stop after this many control steps (None = until Ctrl+C).
        photos_every: Save photos every N steps (0 disables).
        photo_envs: How many envs' car views to save each time.
        out: Photo output directory.
        randomize: Also enable physics DR (per-run draws), to watch cars
            with different bodies diverge.
        grid_spacing: Tile spacing in metres (None keeps the training
            default, 100; :func:`near_spacing` gives the compact-but-safe
            value).
    """
    import torch
    from PIL import Image

    from .._gs import ensure_init
    from ..agents.scripted import CenterlineFollower
    from ..configs.cfgs import get_env_cfg
    from ..envs import DeepRacerEnv

    os.environ.setdefault("MADRONA_MWGPU_DEVICE_HEAP_SIZE", str(1 << 30))
    ensure_init("gpu")
    cfg = get_env_cfg(vision=True, track=list(names), randomize=randomize,
                      view="gui" if gui else "none")
    if grid_spacing is not None:
        cfg["sim"]["track_grid_spacing"] = float(grid_spacing)
        print(f"tile spacing: {grid_spacing:.0f} m")
    env = DeepRacerEnv(num_envs=num_envs, env_cfg=cfg)
    if gui:
        # start the viewer camera ABOVE the grid, all tiles in frame
        try:
            from ..envs.track import grid_offsets

            spacing = float(grid_spacing) if grid_spacing is not None \
                else cfg["sim"]["track_grid_spacing"]
            offs = grid_offsets(len(names), spacing, "cpu").numpy()
            ext = zoo_extent(names)
            lo, hi = offs.min(0) - ext / 2, offs.max(0) + ext / 2
            c = (lo + hi) / 2.0
            vh = (float(max(hi - lo)) / 2.0 + 5.0) / 0.466
            env.scene.viewer.set_camera_pose(
                pos=(float(c[0]), float(c[1]) - 0.15 * vh, vh),
                lookat=(float(c[0]), float(c[1]), 0.0))
        except Exception as e:                       # noqa: BLE001
            print(f"(viewer camera not repositioned: {e})")
    env.reset_idx(torch.arange(num_envs, device=env.device))
    env._post_physics(torch.arange(num_envs, device=env.device))
    controller = CenterlineFollower()
    os.makedirs(out, exist_ok=True)
    if num_envs < len(names):
        print(f"note: {num_envs} cars < {len(names)} variants — "
              f"{len(names) - num_envs} tiles stay empty (raise num_envs "
              "for full coverage; training wants num_envs >= variants too)")

    def _save_photos(t: int) -> None:
        """One tiled contact sheet + individual car-view frames at step t."""
        k = min(photo_envs, num_envs)
        tiles = []
        for i in range(k):
            frame = (env.obs_image_buf[i].permute(1, 2, 0) * 255
                     ).byte().cpu().numpy()
            Image.fromarray(frame).save(
                os.path.join(out, f"watch_env{i}_step{t:05d}.png"))
            tiles.append(Image.fromarray(frame).resize(
                (frame.shape[1] * 2, frame.shape[0] * 2), Image.NEAREST))
        cols = max(1, int(np.ceil(np.sqrt(k))))
        rows = int(np.ceil(k / cols))
        w, h = tiles[0].width + 2, tiles[0].height + 2
        sheet = Image.new("RGB", (cols * w, rows * h), (10, 10, 10))
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % cols) * w, (i // cols) * h))
        sheet_path = os.path.join(out, f"watch_sheet_step{t:05d}.png")
        sheet.save(sheet_path)
        print(f"step {t}: {k} car views -> {sheet_path}")

    print(f"{num_envs} cars on {len(names)} tiles — Ctrl+C to stop"
          + (f" ({steps} steps)" if steps else ""))
    if photos_every:
        _save_photos(0)          # the cameras' view exists from step 0 —
        # save immediately so a short/interrupted run still leaves photos
    t = 0
    try:
        while steps is None or t < steps:
            env.step(controller.act(env))
            t += 1
            if photos_every and t % photos_every == 0:
                _save_photos(t)
    except KeyboardInterrupt:
        pass
    if photos_every:
        _save_photos(t)
    print(f"stopped after {t} steps")


# --------------------------------------------------- run-the-file entry points
def _prepare(zoo_or_names, *, force: bool = False,
             grid_spacing: Optional[float] = None) -> tuple[tuple, float]:
    """Compile a manifest (or accept ready names) and resolve tile spacing.

    Args:
        zoo_or_names: A :class:`Zoo` manifest, or already-compiled names.
        force: Rebake assets that already exist.
        grid_spacing: Explicit spacing; None falls back to the manifest's,
            then to the compact-but-safe automatic spacing.

    Returns:
        ``(names, spacing_m)`` — with the compiled population and the
        plumbing line printed on the way (the file IS the program; running
        it should tell you what it built).
    """
    if isinstance(zoo_or_names, Zoo):
        names = compile_zoo(zoo_or_names, force=force)
        if grid_spacing is None:
            grid_spacing = zoo_or_names.grid_spacing
        label = zoo_or_names.name
    else:
        names = tuple(zoo_or_names)
        label = f"{len(names)} tracks"
    if grid_spacing is None:
        grid_spacing = near_spacing(names)
    print(f"zoo '{label}': {len(names)} variants")
    print(f"plumb into training:\n  CameraEnvironment(tracks={names!r})")
    return names, float(grid_spacing)


def view(zoo_or_names, *, screenshot: Optional[str] = None,
         seconds: Optional[float] = None, force: bool = False,
         grid_spacing: Optional[float] = None) -> None:
    """Compile and open the bare-scene view — from a manifest, in one call.

    The run-the-file idiom: a manifest ends with
    ``if __name__ == "__main__": view(my_zoo)`` and is executed directly —
    no CLI (see ``examples/zoos.py``).

    Args:
        zoo_or_names: The :class:`Zoo` manifest (or compiled names).
        screenshot: Render one top-down overview PNG instead of the viewer.
        seconds: Auto-close the viewer after this long (None = Ctrl+C).
        force: Rebake assets that already exist.
        grid_spacing: Tile spacing override (default: manifest, then the
            view's auto-compact fit).
    """
    names, _ = _prepare(zoo_or_names, force=force, grid_spacing=grid_spacing)
    view_zoo(names, grid_spacing=grid_spacing, screenshot=screenshot,
             seconds=seconds)


def watch(zoo_or_names, *, num_envs: int = 12, gui: bool = True,
          steps: Optional[int] = None, photos_every: int = 100,
          photo_envs: int = 4, out: str = "logs/zoo",
          randomize: bool = False, force: bool = False,
          grid_spacing: Optional[float] = None) -> None:
    """Compile and run cars on the zoo — from a manifest, in one call.

    The run-the-file idiom: a manifest ends with
    ``if __name__ == "__main__": watch(my_zoo, num_envs=32)`` and is
    executed directly — no CLI (see ``examples/zoos.py``).

    Args:
        zoo_or_names: The :class:`Zoo` manifest (or compiled names).
        num_envs: Parallel cars, balanced across the tiles.
        gui: Open the interactive viewer window (needs a display).
        steps: Stop after this many control steps (None = until Ctrl+C).
        photos_every: Save car-view photos every N steps (0 disables).
        photo_envs: How many envs' car views to save each time.
        out: Photo output directory.
        randomize: Also draw per-run physics DR (watch cars diverge).
        force: Rebake assets that already exist.
        grid_spacing: Tile spacing (default: manifest, then compact-but-safe
            auto).
    """
    names, spacing = _prepare(zoo_or_names, force=force,
                              grid_spacing=grid_spacing)
    watch_zoo(names, num_envs=num_envs, gui=gui, steps=steps,
              photos_every=photos_every, photo_envs=photo_envs, out=out,
              randomize=randomize, grid_spacing=spacing)


def default_zoo() -> Zoo:
    """The out-of-the-box population when no manifest is given.

    Returns:
        32 seeded official tracks, waypoint-noised and look-randomized —
        equivalent to a manifest declaring ``(OfficialSample(32),)``.
    """
    return Zoo("default", (OfficialSample(32),))


def load_manifest(ref: str) -> Zoo:
    """Load a Zoo manifest from a Python file or module.

    Config-as-code: the manifest is an ordinary Python file declaring one or
    more :class:`Zoo` objects (see ``examples/zoos.py``).

    Args:
        ref: ``path/to/zoos.py``, ``path/to/zoos.py:name``, ``module``, or
            ``module:name`` — ``name`` may be a ``Zoo`` or a callable
            returning one. Without ``:name``, a module defining exactly one
            Zoo (or one named ``zoo``) resolves automatically.

    Returns:
        The declared zoo.

    Raises:
        SystemExit: If the module defines no unambiguous Zoo.
    """
    import importlib
    import importlib.util
    import sys

    path, _, attr = ref.partition(":")
    if path.endswith(".py"):
        spec = importlib.util.spec_from_file_location(
            os.path.splitext(os.path.basename(path))[0], path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    else:
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        mod = importlib.import_module(path)
    if attr:
        obj = getattr(mod, attr)
        return obj() if callable(obj) else obj
    zoos = {k: v for k, v in vars(mod).items() if isinstance(v, Zoo)}
    if "zoo" in zoos:
        return zoos["zoo"]
    if len(zoos) == 1:
        return next(iter(zoos.values()))
    raise SystemExit(
        f"{ref}: defines {sorted(zoos) if zoos else 'no'} Zoo objects — "
        f"pick one explicitly, e.g. {path}:{next(iter(sorted(zoos)), 'my_zoo')}")


def main(argv: Optional[list] = None) -> int:
    """CLI entry point: ``compile`` / ``view`` / ``watch`` over a manifest.

    What to build lives in the MANIFEST FILE (config-as-code — the repo
    rule); the flags below only control how this run executes (headless
    screenshot, step counts, photo cadence, cache forcing).

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    import argparse

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser(
        prog="python -m deepracer_genesis.tools.zoo", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("manifest", nargs="?", default=None,
                       help="zoo manifest: path/to/zoos.py[:name] or "
                            "module[:name] (default: 32 officials, noised + "
                            "look-randomized — see examples/zoos.py)")
        p.add_argument("--force", action="store_true",
                       help="rebake assets that already exist")
        p.add_argument("--grid-spacing", type=float, default=None,
                       dest="grid_spacing",
                       help="tile spacing override in metres (default: the "
                            "manifest's, else compact-but-safe auto)")

    p = sub.add_parser("compile", help="lint + bake the manifest; print names")
    common(p)

    p = sub.add_parser("view", help="bare-scene grid of the zoo (GUI or PNG)")
    common(p)
    p.add_argument("--screenshot", default=None,
                   help="render a top-down overview PNG instead of the viewer")
    p.add_argument("--seconds", type=float, default=None)

    p = sub.add_parser("watch", help="cars driving on the zoo (env + viewer + photos)")
    common(p)
    p.add_argument("--num-envs", type=int, default=12, dest="num_envs")
    p.add_argument("--no-gui", action="store_true")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--photos-every", type=int, default=100, dest="photos_every")
    p.add_argument("--photo-envs", type=int, default=4, dest="photo_envs")
    p.add_argument("--out", default="logs/zoo")
    p.add_argument("--randomize", action="store_true",
                   help="also draw per-run physics DR (watch cars diverge)")

    args = ap.parse_args(argv)
    zoo = load_manifest(args.manifest) if args.manifest else default_zoo()
    names = compile_zoo(zoo, force=args.force)
    print(f"zoo '{zoo.name}': {len(names)} variants")
    for n in names:
        print(f"  {n}")
    print(f"\nplumb into training:\n  CameraEnvironment(tracks={names!r})")
    if args.cmd == "view":
        view_zoo(names, grid_spacing=args.grid_spacing,
                 screenshot=args.screenshot, seconds=args.seconds)
    elif args.cmd == "watch":
        # spacing: explicit flag > manifest > compact-but-safe auto
        spacing = args.grid_spacing
        if spacing is None:
            spacing = zoo.grid_spacing
        if spacing is None:
            spacing = near_spacing(names)
        watch_zoo(names, num_envs=args.num_envs, gui=not args.no_gui,
                  steps=args.steps, photos_every=args.photos_every,
                  photo_envs=args.photo_envs, out=args.out,
                  randomize=args.randomize, grid_spacing=spacing)
    return 0


if __name__ == "__main__":
    # `python -m deepracer_genesis.tools.zoo` executes this file as
    # ``__main__``; a manifest importing the module would then see a SECOND
    # copy of the classes and isinstance dispatch would fail. Re-dispatch
    # into the canonical module so there is exactly one class identity.
    import sys

    from deepracer_genesis.tools.zoo import main as _canonical_main

    sys.exit(_canonical_main())
