"""Build renderable road meshes procedurally from DeepRacer waypoint routes,
and fetch/install any official track so it becomes drivable and renderable."""

from __future__ import annotations

import os
import urllib.request

from typing import Optional, Sequence

import numpy as np

from .. import ASSETS_DIR

RACE_DATA_RAW = ("https://raw.githubusercontent.com/aws-deepracer-community/"
                 "deepracer-race-data/main/raw_data/tracks/npy/{name}.npy")

GENERATED_DIR = os.path.join(ASSETS_DIR, "tracks", "generated")

# Solid-color materials delivered as tiny TEXTURES: Madrona's textureless
# material path misassigns per-submesh Kd colors (parsed correctly by
# genesis, rendered scrambled), while its textured path is well-exercised
# by the original DAE tracks. 4x4 solid PNGs cost nothing.
_PALETTE = {"road": (41, 43, 51), "border": (235, 235, 235),
            "centerline": (242, 166, 26)}


# ----------------------------------------------------------------- geometry
def _as_ccw(pts: np.ndarray) -> np.ndarray:
    """Return the closed polygon oriented counterclockwise.

    The border offsets and mesh winding downstream assume a CCW loop, so a
    clockwise input is reversed (order-reversed, not polar-sorted, which would
    reorder the path and wreck concave tracks). Driving direction follows the
    canonical CCW order.

    Args:
        pts: (P, 2) polygon vertices in traversal order.

    Returns:
        The same vertices, reversed iff the loop was clockwise.
    """
    x, y = pts[:, 0], pts[:, 1]
    signed_area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return pts[::-1] if signed_area < 0 else pts


def build_route(points_xy, half_width: float, n_waypoints: int = 150,
                smooth_passes: int = 3) -> np.ndarray:
    """Turn a rough closed polygon into a (W, 6) DeepRacer route via Chaikin
    smoothing, even arclength resampling, and half_width border offsets.

    Args:
        points_xy: Any (P, 2) sequence of corner points (P >= 3), traversed
            in order and closed automatically.
        half_width: Border offset from the centerline in meters (half the
            track width).
        n_waypoints: Number of output waypoints.
        smooth_passes: Chaikin corner-cutting iterations.

    Returns:
        (W, 6) float array of [center_xy, inner_xy, outer_xy] per waypoint.

    Raises:
        ValueError: If points_xy is not (P, 2) with P >= 3.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise ValueError("points_xy must be (P, 2) with P >= 3")
    pts = _as_ccw(pts)                        # canonical winding for offsets

    # Chaikin corner cutting on the CLOSED polygon
    for _ in range(smooth_passes):
        nxt = np.roll(pts, -1, axis=0)
        pts = np.stack([0.75 * pts + 0.25 * nxt,
                        0.25 * pts + 0.75 * nxt], axis=1).reshape(-1, 2)

    # arclength-uniform resampling to n_waypoints
    seg = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    samples = np.linspace(0.0, total, n_waypoints, endpoint=False)
    idx = np.searchsorted(cum, samples, side="right") - 1
    idx = np.clip(idx, 0, len(pts) - 1)
    frac = (samples - cum[idx]) / np.maximum(seg[idx], 1e-9)
    nxt = np.roll(pts, -1, axis=0)
    center = pts[idx] * (1 - frac[:, None]) + nxt[idx] * frac[:, None]

    # left normals of the tangent -> inner/outer borders
    tangent = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    inner = center + normal * half_width
    outer = center - normal * half_width
    return np.concatenate([center, inner, outer], axis=1)


def route_from_waypoints(waypoints_xy, width: float,
                         n_waypoints: Optional[int] = None,
                         waypoint_spacing: float = 0.3) -> np.ndarray:
    """Build a (W, 6) route from unsmoothed centerline waypoints and a width,
    densifying the polyline so downstream localization features stay valid.

    Args:
        waypoints_xy: (P, 2) centerline points in meters, in driving order.
            The loop closes automatically (a repeated last point is dropped).
        width: full track width in meters (official tracks: ~1.06).
        n_waypoints: exact number of output waypoints. Overrides
            `waypoint_spacing` when given.
        waypoint_spacing: target distance between output waypoints in meters
            (used when `n_waypoints` is None; output count is capped at 5000).

    Returns:
        (W, 6) float array of [center_xy, inner_xy, outer_xy] per waypoint.

    Raises:
        ValueError: fewer than 3 points or a malformed array.
    """
    pts = np.asarray(waypoints_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise ValueError("waypoints_xy must be (P, 2) with P >= 3")
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]                        # closed automatically
    pts = _as_ccw(pts)                        # canonical winding for offsets

    seg = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if n_waypoints is None:
        n_waypoints = int(np.clip(round(cum[-1] / waypoint_spacing), 32, 5000))
    samples = np.linspace(0.0, cum[-1], n_waypoints, endpoint=False)
    idx = np.clip(np.searchsorted(cum, samples, side="right") - 1,
                  0, len(pts) - 1)
    frac = (samples - cum[idx]) / np.maximum(seg[idx], 1e-9)
    nxt = np.roll(pts, -1, axis=0)
    pts = pts[idx] * (1 - frac[:, None]) + nxt[idx] * frac[:, None]

    tangent = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    inner = pts + normal * (width / 2)
    outer = pts - normal * (width / 2)
    return np.concatenate([pts, inner, outer], axis=1)


def scale_route_width(route: np.ndarray, scale: float) -> np.ndarray:
    """Scale a route's width about its centerline; the centerline is untouched.

    Only the border columns move, so waypoint count, arclength, spawn poses,
    and localization geometry are identical across width variants — and the
    rulebook width (``Track.half_width`` is derived from these same borders)
    tracks the rendered mesh by construction. Per-waypoint width variation in
    the source route is preserved (each border offset scales individually).

    Args:
        route: (W, 6) route array ([center, inner, outer] per waypoint).
        scale: Width multiplier (> 0); 1.0 returns an unchanged copy.

    Returns:
        A new (W, 6) route with both borders scaled by ``scale``.

    Raises:
        ValueError: If the route is not (W, 6) or ``scale`` is not positive.
    """
    r = np.asarray(route, dtype=np.float64).copy()
    if r.ndim != 2 or r.shape[1] != 6:
        raise ValueError(f"route must be (W, 6) [center,inner,outer]; got {r.shape}")
    if not scale > 0:
        raise ValueError(f"width scale must be > 0; got {scale}")
    center = r[:, 0:2]
    r[:, 2:4] = center + (r[:, 2:4] - center) * scale
    r[:, 4:6] = center + (r[:, 4:6] - center) * scale
    return r


def track_metrics(route: np.ndarray) -> dict:
    """Physical measurements for building the track in real life.

    Args:
        route: (W, 6) route array ([center, inner, outer] per waypoint).

    Returns:
        Dict with:
            length_m: centerline lap length.
            width_m: mean track width.
            road_area_m2: paved surface area (shoelace of the outer border
                polygon minus the inner one — exact for the ribbon).
            footprint_m: (x, y) bounding-box size — the floor space you
                need, borders included.
            min_turn_radius_m: tightest centerline turn (1/max curvature);
                AWS suggests keeping physical turns manageable for the car
                (~0.4 m radius is already tight at DeepRacer scale).
    """
    center, inner, outer = route[:, 0:2], route[:, 2:4], route[:, 4:6]
    seg = np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1)
    length = float(seg.sum())

    def shoelace(poly):
        x, y = poly[:, 0], poly[:, 1]
        return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))

    width = float(np.linalg.norm(outer - inner, axis=1).mean())
    lo = np.minimum(inner, outer).min(axis=0)
    hi = np.maximum(inner, outer).max(axis=0)

    yaw = np.arctan2(*(np.roll(center, -1, axis=0) - center).T[::-1])
    dyaw = np.abs((np.diff(yaw, append=yaw[:1]) + np.pi) % (2 * np.pi) - np.pi)
    curvature = dyaw / np.maximum(seg, 1e-9)
    max_k = float(np.percentile(curvature, 99))          # robust to kinks

    return {
        "length_m": round(length, 2),
        "width_m": round(width, 3),
        "road_area_m2": round(abs(shoelace(outer) - shoelace(inner)), 2),
        "footprint_m": (round(float(hi[0] - lo[0]), 2),
                        round(float(hi[1] - lo[1]), 2)),
        "min_turn_radius_m": round(1.0 / max_k, 2) if max_k > 1e-6 else float("inf"),
    }


def stadium(straight: float, radius: float, arc_pts: int = 60,
            straight_pts: int = 20) -> np.ndarray:
    """Centerline of a stadium ("stretched-oval" / pill): two straights joined
    by semicircular ends, centered on the origin and wound counterclockwise.

    Footprint tips for fitting a room: with track ``width`` the centerline
    spans ``(straight + 2*radius) x (2*radius)`` and the paved footprint adds
    ``width`` to each dimension, so keep ``straight + 2*radius + width`` and
    ``2*radius + width`` within the floor. Tighter ends (smaller ``radius``,
    longer ``straight``) read as a longer pill; ``radius`` sets the centerline
    turn radius and must exceed ``width / 2`` or the inner border self-pinches.

    Args:
        straight: Length of each straight section in meters.
        radius: Semicircle end radius (the centerline turn radius) in meters.
        arc_pts: Samples per semicircular end.
        straight_pts: Samples per straight section.

    Returns:
        (P, 2) centerline points ready for :func:`route_from_waypoints`.
    """
    L, r = straight, radius
    bottom = np.stack([np.linspace(-L / 2, L / 2, straight_pts, endpoint=False),
                       np.full(straight_pts, -r)], axis=1)
    a = np.linspace(-np.pi / 2, np.pi / 2, arc_pts, endpoint=False)
    right = np.stack([L / 2 + r * np.cos(a), r * np.sin(a)], axis=1)
    top = np.stack([np.linspace(L / 2, -L / 2, straight_pts, endpoint=False),
                    np.full(straight_pts, r)], axis=1)
    a2 = np.linspace(np.pi / 2, 3 * np.pi / 2, arc_pts, endpoint=False)
    left = np.stack([-L / 2 + r * np.cos(a2), r * np.sin(a2)], axis=1)
    return np.concatenate([bottom, right, top, left], axis=0)


def _rgb(c):
    """Normalize a (0-255) RGB tuple to matplotlib 0-1 floats (pass through str)."""
    return c if isinstance(c, str) else tuple(v / 255 for v in c)


def _close(poly: np.ndarray) -> np.ndarray:
    """Repeat the first vertex so a fill/plot closes the loop cleanly."""
    return np.vstack([poly, poly[:1]])


def _dash_quads(center: np.ndarray, nrm: np.ndarray, dash_len: float,
                dash_gap: float, half_thick: float):
    """Yield (x, y) polygons for centerline dashes laid out in meters.

    Walks the closed centerline by arclength, emitting one rectangle per dash
    (length ``dash_len`` along travel, ``2*half_thick`` across it), matching how
    :func:`build_track_mesh` bakes the dashed centerline into the road mesh.

    Args:
        center: (W, 2) centerline points.
        nrm: (W, 2) unit normals (across-road direction) per point.
        dash_len: Dash length in meters.
        dash_gap: Gap between dashes in meters.
        half_thick: Half the dash thickness in meters.

    Yields:
        (xs, ys) vertex arrays for each dash quad, ready for ``ax.fill``.
    """
    seg = np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    w = len(center)

    def at(s):
        s %= total
        i = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, w - 1))
        j = (i + 1) % w
        f = (s - cum[i]) / max(seg[i], 1e-9)
        return center[i] * (1 - f) + center[j] * f, nrm[i]

    s = 0.0
    while s < total:
        pa, na = at(s)
        pb, nb = at(min(s + dash_len, total))
        quad = np.array([pa + na * half_thick, pa - na * half_thick,
                         pb - nb * half_thick, pb + nb * half_thick])
        yield quad[:, 0], quad[:, 1]
        s += dash_len + dash_gap


def plot_track(route: np.ndarray, *, out_path: Optional[str] = None,
               ax=None, room=None, show_centerline: bool = True,
               border_strip: float = 0.04, dash_len: float = 0.30,
               dash_gap: float = 0.35, centerline_width: float = 0.05,
               dpi: int = 300,
               grass=(92, 138, 92), asphalt=_PALETTE["road"],
               border=_PALETTE["border"], centerline=_PALETTE["centerline"]):
    """Render a clean, chrome-free top-down of the track for printing.

    Layers grass over the whole frame, a white border strip on each edge of the
    black asphalt ribbon, the grass infield, and an optional dashed centerline.
    No axes, ticks, title, or margins — just the track.

    Args:
        route: (W, 6) route array ([center, inner, outer] per waypoint).
        out_path: If given, save here (``.png`` raster or ``.pdf``/``.svg``
            vector for scalable printing); the grass fills the whole canvas.
        ax: Existing matplotlib Axes to draw on; a new figure is made if None.
        room: Optional (width, height) in meters. When given, the view is
            framed to the whole floor (track centered), so the grass margin
            around the track is visible; otherwise it crops tight to the track.
        show_centerline: Draw the dashed centerline.
        border_strip: Width of the white edge strips in meters (matches the
            installed mesh's border lines by default).
        dash_len: Centerline dash length in meters (matches the mesh default).
        dash_gap: Gap between centerline dashes in meters.
        centerline_width: Centerline dash thickness in meters.
        dpi: Raster resolution when saving to a pixel format.
        grass: Infield/background color (RGB 0-255 tuple or matplotlib color).
        asphalt: Road color.
        border: Border-strip color.
        centerline: Centerline dash color.

    Returns:
        (fig, ax) for further tweaking or saving by the caller.
    """
    import matplotlib.pyplot as plt

    grass, asphalt, border, centerline = (
        _rgb(grass), _rgb(asphalt), _rgb(border), _rgb(centerline))
    center, inner, outer = route[:, 0:2], route[:, 2:4], route[:, 4:6]
    nrm = outer - center
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    outer_plus = outer + nrm * border_strip
    inner_minus = inner - nrm * border_strip

    span = np.asarray(room, dtype=float) if room is not None else (
        outer_plus.max(axis=0) - outer_plus.min(axis=0))
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12 * span[1] / span[0]))
    else:
        fig = ax.figure
    fig.patch.set_facecolor(grass)
    ax.set_facecolor(grass)

    ax.fill(*_close(outer_plus).T, color=border)      # outer white strip
    ax.fill(*_close(outer).T, color=asphalt)          # asphalt ribbon
    ax.fill(*_close(inner).T, color=border)            # inner white strip
    ax.fill(*_close(inner_minus).T, color=grass)       # grass infield
    if show_centerline:
        for xs, ys in _dash_quads(center, nrm, dash_len, dash_gap,
                                  centerline_width / 2):
            ax.fill(xs, ys, color=centerline)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(0)
    if room is not None:
        rw, rh = room
        ax.set_xlim(-rw / 2, rw / 2)
        ax.set_ylim(-rh / 2, rh / 2)
    if out_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path, dpi=dpi, facecolor=grass,
                    bbox_inches="tight", pad_inches=0)
    return fig, ax


def plot_wireframe(route: np.ndarray, waypoints=None, *,
                   out_path: Optional[str] = None, ax=None, room=None,
                   dpi: int = 200):
    """Render an inspection wireframe: border/centerline outlines, waypoint
    dots, a start marker, and a driving-direction arrow, with axes in meters.

    Args:
        route: (W, 6) route array ([center, inner, outer] per waypoint).
        waypoints: Optional (P, 2) source waypoints to mark; falls back to the
            route centerline. Dots are thinned for legibility.
        out_path: If given, save the figure here.
        ax: Existing matplotlib Axes; a new figure is made if None.
        room: Optional (width, height) in meters, drawn as a dashed floor
            boundary centered on the origin.
        dpi: Raster resolution when saving.

    Returns:
        (fig, ax) for further tweaking or saving by the caller.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    center, inner, outer = route[:, 0:2], route[:, 2:4], route[:, 4:6]
    wp = center if waypoints is None else np.asarray(waypoints, dtype=float)
    step = max(1, len(wp) // 40)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6.5))
    else:
        fig = ax.figure
    ax.plot(*_close(inner).T, color=_rgb(_PALETTE["road"]), lw=1.0, label="inner border")
    ax.plot(*_close(outer).T, color=_rgb(_PALETTE["road"]), lw=1.0, label="outer border")
    ax.plot(*_close(center).T, "--", color=_rgb(_PALETTE["centerline"]), lw=1.0,
            label="centerline")
    ax.plot(wp[::step, 0], wp[::step, 1], "o", color="tomato", ms=4, label="waypoints")
    ax.plot(*wp[0], "s", color="red", ms=9, label="start")
    ax.annotate("", xy=wp[min(3, len(wp) - 1)], xytext=wp[0],
                arrowprops=dict(arrowstyle="->", color="red", lw=2))
    if room is not None:
        rw, rh = room
        ax.add_patch(Rectangle((-rw / 2, -rh / 2), rw, rh, fill=False,
                               ec="#bbb", lw=1.0, ls="--"))
    ax.set_aspect("equal")
    ax.grid(True, ls=":", alpha=0.5)
    ax.set_xlabel("meters")
    ax.set_ylabel("meters")
    ax.legend(loc="upper right", fontsize=8)
    if out_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return fig, ax


def _quad(f, base, flip):
    """Up-facing quad; `flip` swaps winding (track handedness varies)."""
    if flip:
        f.write(f"f {base + 2}/1 {base + 1}/1 {base}/1\n")
        f.write(f"f {base + 3}/1 {base + 2}/1 {base}/1\n")
    else:
        f.write(f"f {base}/1 {base + 1}/1 {base + 2}/1\n")
        f.write(f"f {base}/1 {base + 2}/1 {base + 3}/1\n")


def _write_strip(f, left, right, z, vert_offset, flip):
    """Up-facing triangle strip between two closed (W, 2) polylines."""
    w = len(left)
    for i in range(w):
        f.write(f"v {left[i, 0]:.5f} {left[i, 1]:.5f} {z}\n")
        f.write(f"v {right[i, 0]:.5f} {right[i, 1]:.5f} {z}\n")
    for i in range(w):
        j = (i + 1) % w
        a, b = vert_offset + 2 * i, vert_offset + 2 * j
        if flip:
            f.write(f"f {b + 1}/1 {a + 1}/1 {a}/1\n")
            f.write(f"f {b}/1 {b + 1}/1 {a}/1\n")
        else:
            f.write(f"f {a}/1 {a + 1}/1 {b + 1}/1\n")
            f.write(f"f {a}/1 {b + 1}/1 {b}/1\n")
    return vert_offset + 2 * w


def _wall_ring_quads(x0: float, y0: float, x1: float, y1: float,
                     t: float, h: float) -> list:
    """Quads of a rectangular perimeter wall band (single-sided faces).

    Each of the four sides gets an inner vertical face (normal toward the
    ring center), an outer vertical face (normal away), and an up-facing top
    strip; the south/north spans extend by the thickness so the corners
    close. Vertex order per quad is counter-clockwise as seen from the
    visible side (right-hand-rule normals).

    Args:
        x0: Inner-rectangle min x.
        y0: Inner-rectangle min y.
        x1: Inner-rectangle max x.
        y1: Inner-rectangle max y.
        t: Wall thickness (band grows outward).
        h: Wall height.

    Returns:
        Twelve 4-tuples of ``(x, y, z)`` vertices.
    """
    xs0, xs1 = x0 - t, x1 + t
    return [
        # south (inner +y / outer -y / top up)
        ((xs1, y0, 0), (xs0, y0, 0), (xs0, y0, h), (xs1, y0, h)),
        ((xs0, y0 - t, 0), (xs1, y0 - t, 0), (xs1, y0 - t, h), (xs0, y0 - t, h)),
        ((xs0, y0 - t, h), (xs1, y0 - t, h), (xs1, y0, h), (xs0, y0, h)),
        # north
        ((xs0, y1, 0), (xs1, y1, 0), (xs1, y1, h), (xs0, y1, h)),
        ((xs1, y1 + t, 0), (xs0, y1 + t, 0), (xs0, y1 + t, h), (xs1, y1 + t, h)),
        ((xs0, y1, h), (xs1, y1, h), (xs1, y1 + t, h), (xs0, y1 + t, h)),
        # west (inner +x / outer -x / top up)
        ((x0, y0, 0), (x0, y1, 0), (x0, y1, h), (x0, y0, h)),
        ((x0 - t, y1, 0), (x0 - t, y0, 0), (x0 - t, y0, h), (x0 - t, y1, h)),
        ((x0 - t, y0, h), (x0, y0, h), (x0, y1, h), (x0 - t, y1, h)),
        # east
        ((x1, y1, 0), (x1, y0, 0), (x1, y0, h), (x1, y1, h)),
        ((x1 + t, y0, 0), (x1 + t, y1, 0), (x1 + t, y1, h), (x1 + t, y0, h)),
        ((x1, y0, h), (x1 + t, y0, h), (x1 + t, y1, h), (x1, y1, h)),
    ]


def build_track_mesh(route: np.ndarray, out_obj: str, *,
                     line_width: float = 0.04, dash_len: float = 0.30,
                     dash_gap: float = 0.35, palette: "dict | None" = None,
                     field: "tuple | None" = None,
                     wall: "tuple | None" = None,
                     wall_height: float = 0.3) -> str:
    """Write a road-ribbon OBJ (road, border lines, dashed centerline) plus its
    .mtl and solid-color 4x4 PNG textures next to it.

    Args:
        route: (W, 6) route array ([center, inner, outer] per waypoint).
        out_obj: Output .obj path (parent directories are created).
        line_width: Border-line width in meters.
        dash_len: Centerline dash length in meters.
        dash_gap: Gap between centerline dashes in meters.
        palette: Optional 0-255 RGB overrides for any of ``road``/``border``/
            ``centerline`` (merged over the default palette).
        field: Optional 0-255 RGB — bakes a per-track ground quad (track
            bounding box + margin, just above the global plane) INTO the
            mesh, so a variant's field colour travels with its asset and
            tiles automatically under every renderer.
        wall: Optional 0-255 RGB — bakes a perimeter wall band around the
            track footprint (at the field-quad edge) into the mesh: the
            visual barrier a real DeepRacer venue has. Visual-only — the
            track entity is added without collision; the rulebook/termination
            is the actual fence.
        wall_height: Wall height in metres.

    Returns:
        The written .obj path.
    """
    pal = {**_PALETTE, **(palette or {})}
    # MESH-ONLY winding normalization: a clockwise route (official *_cw
    # variants) must not change driving direction — the route file stays
    # untouched — but the legacy flip branch baked downward-facing
    # (invisible) roads for CW input, so the baker reverses its LOCAL copy
    # of the geometry instead (a mesh is direction-agnostic).
    x, y = route[:, 0], route[:, 1]
    if 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) < 0:
        route = route[::-1]
    center, inner, outer = route[:, 0:2], route[:, 2:4], route[:, 4:6]
    # border columns may be stored travel-relative (official files) or
    # geometry-fixed; the strip writer is only verified for "inner = left of
    # travel", so decide by GEOMETRY, not labels
    t = np.roll(center, -1, axis=0) - center
    left = np.stack([-t[:, 1], t[:, 0]], axis=1)
    if np.median(np.sum((inner - center) * left, axis=1)) < 0:
        inner, outer = outer, inner
    normal = (inner - center)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    tangent = np.stack([normal[:, 1], -normal[:, 0]], axis=1)
    # winding: signed area of the centerline decides which triangle order
    # faces +z (single-sided faces; coincident double-sided faces z-fight)
    x, y = center[:, 0], center[:, 1]
    signed_area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    flip = signed_area < 0

    os.makedirs(os.path.dirname(out_obj), exist_ok=True)
    out_dir = os.path.dirname(out_obj)
    mtl_name = os.path.basename(out_obj).replace(".obj", ".mtl")
    from PIL import Image
    mats = dict(pal)
    if field is not None:
        mats["field"] = tuple(field)
    if wall is not None:
        mats["wall"] = tuple(wall)
    with open(os.path.join(out_dir, mtl_name), "w") as f:
        for mat, rgb in mats.items():
            Image.new("RGB", (4, 4), rgb).save(os.path.join(out_dir, f"{mat}.png"))
            kd = " ".join(f"{c / 255:.4f}" for c in rgb)
            f.write(f"newmtl {mat}\nKd {kd}\nKa 0 0 0\nKs 0 0 0\n"
                    f"map_Kd {mat}.png\n")

    with open(out_obj, "w") as f:
        f.write(f"mtllib {mtl_name}\n")
        # variant signature: keeps OBJ bytes unique per palette/field so
        # genesis's mesh-preprocessing cache never dedups two palette variants
        # of the same geometry onto one processed asset (the appearance.py trap)
        sig = ",".join(f"{m}:{r}.{g}.{b}" for m, (r, g, b) in sorted(mats.items()))
        f.write(f"# variant {sig}\n")
        f.write("vt 0.5 0.5\n")            # single UV; textures are solid
        v = 1
        # field quad and wall ring share the footprint-proportional margin so
        # the wall stands exactly at the field's edge (kept tight so tiles
        # can pack close in the zoo views)
        m = max(0.3, 0.18 * float(max(outer.max(0) - outer.min(0))))
        (x0, y0), (x1, y1) = outer.min(0) - m, outer.max(0) + m
        if field is not None:
            # ground quad: above the global plane (z=-0.001), below the road
            f.write("usemtl field\n")
            for px, py in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
                f.write(f"v {px:.5f} {py:.5f} 0.0005\n")
            _quad(f, v, False)
            v += 4
        if wall is not None:
            f.write("usemtl wall\n")
            for quad in _wall_ring_quads(x0, y0, x1, y1, 0.05, wall_height):
                for px, py, pz in quad:
                    f.write(f"v {px:.5f} {py:.5f} {pz:.5f}\n")
                f.write(f"f {v}/1 {v + 1}/1 {v + 2}/1\n")
                f.write(f"f {v}/1 {v + 2}/1 {v + 3}/1\n")
                v += 4
        f.write("usemtl road\n")
        v = _write_strip(f, inner, outer, 0.001, v, flip)
        f.write("usemtl border\n")
        v = _write_strip(f, inner, inner - normal * line_width, 0.002, v, flip)
        v = _write_strip(f, outer, outer + normal * line_width, 0.002, v, not flip)

        # dashed centerline: quads along arclength
        f.write("usemtl centerline\n")
        seg = np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        s, total = 0.0, cum[-1]
        while s < total - dash_len:
            a = np.searchsorted(cum, s, side="right") - 1
            b = np.searchsorted(cum, s + dash_len, side="right") - 1
            a, b = min(a, len(center) - 1), min(b, len(center) - 1)
            pa = center[a] + tangent[a] * (s - cum[a])
            pb = center[b] + tangent[b] * (s + dash_len - cum[b])
            na, nb = normal[a] * (line_width * 1.2), normal[b] * (line_width * 1.2)
            for p in (pa + na, pa - na, pb - nb, pb + nb):
                f.write(f"v {p[0]:.5f} {p[1]:.5f} 0.003\n")
            _quad(f, v, flip)
            v += 4
            s += dash_len + dash_gap
    return out_obj


# ------------------------------------------------------------ registration
def install_track(name: str, route: np.ndarray, *,
                  palette: "dict | None" = None,
                  field: "tuple | None" = None,
                  wall: "tuple | None" = None) -> str:
    """Persist a route and generated mesh under the assets tree and register it
    so the track is usable anywhere a track name is accepted.

    Args:
        name: Track name to register under.
        route: (W, 6) [center, inner, outer] route array.
        palette: Optional 0-255 RGB overrides for road/border/centerline
            (see :func:`build_track_mesh`).
        field: Optional 0-255 RGB per-track ground quad baked into the mesh.
        wall: Optional 0-255 RGB perimeter wall baked into the mesh.

    Returns:
        The track directory under the generated-assets tree.

    Raises:
        ValueError: If the route is not shaped (W, 6).
    """
    from ..envs.track import TRACKS

    route = np.asarray(route, dtype=np.float64)
    if route.ndim != 2 or route.shape[1] != 6:
        raise ValueError(f"route must be (W, 6) [center,inner,outer]; got {route.shape}")

    track_dir = os.path.join(GENERATED_DIR, name)
    os.makedirs(track_dir, exist_ok=True)
    np.save(os.path.join(track_dir, "route.npy"), route)
    build_track_mesh(route, os.path.join(track_dir, "track.obj"),
                     palette=palette, field=field, wall=wall)

    rel = os.path.relpath(track_dir, ASSETS_DIR)
    TRACKS[name] = (f"{rel}/track.obj", f"{rel}/route.npy", None)
    return track_dir


def width_variants(track: str, scales: Sequence[float], *,
                   force: bool = False) -> tuple[str, ...]:
    """Bake and install width-scaled variants of ``track``; return their names.

    The camera-mode answer to track-width DR: the ``track_width_scale`` knob
    is feature-only (it scales the rulebook, which a baked mesh cannot
    follow), so under camera the width itself must vary. Each variant here is
    a first-class generated track — same centerline, borders scaled — so
    passing the returned names as ``EnvSpec.tracks`` gives per-env *visible*
    width via the existing multi-track tiling, with the rulebook following
    each variant's own route automatically (``Track.half_width`` derives from
    the route borders). Schedule: per env, fixed for the run.

    Variant meshes use the procedural generated-track look (road ribbon +
    border lines + dashed centerline), also for official DAE source tracks.

    Args:
        track: Source track name (any registered track with a route).
        scales: Width multipliers, e.g. ``(0.9, 1.0, 1.15)``. A scale of 1.0
            reuses the source track itself (no duplicate bake).
        force: Rebake variants whose assets already exist on disk.

    Returns:
        One registered track name per scale, in the given order, e.g.
        ``("tight_oval_w090", "tight_oval", "tight_oval_w115")``.

    Raises:
        KeyError: If ``track`` is not a registered track name.
        ValueError: If a scale is not positive.
    """
    from ..envs.track import TRACKS

    _mesh_rel, route_rel, _field = TRACKS[track]
    route = np.load(os.path.join(ASSETS_DIR, route_rel))
    names = []
    for scale in scales:
        if not scale > 0:
            raise ValueError(f"width scale must be > 0; got {scale}")
        if abs(scale - 1.0) < 1e-9:
            names.append(track)
            continue
        name = f"{track}_w{round(scale * 100):03d}"
        track_dir = os.path.join(GENERATED_DIR, name)
        if force or not os.path.exists(os.path.join(track_dir, "route.npy")):
            install_track(name, scale_route_width(route, scale))
        elif name not in TRACKS:
            # assets were baked before this process imported the registry
            # (discovery runs at track-module import) — register them now
            rel = os.path.relpath(track_dir, ASSETS_DIR)
            TRACKS[name] = (f"{rel}/track.obj", f"{rel}/route.npy", None)
        names.append(name)
    return tuple(names)


def fetch_official_track(name: str, *, force: bool = False) -> str:
    """Download an official route from deepracer-race-data and install it
    (126 tracks available, e.g. Oval_track, Bowtie_track, Monaco).

    Args:
        name: Official track name, exactly as spelled in the repo.
        force: Re-download and rebuild even when already installed.

    Returns:
        The installed track directory.

    Raises:
        RuntimeError: If the download fails or the fetched route has an
            unexpected shape.
    """
    from ..envs.track import TRACKS

    track_dir = os.path.join(GENERATED_DIR, name)
    if not force and name in TRACKS and os.path.exists(
            os.path.join(track_dir, "route.npy")):
        return track_dir
    url = RACE_DATA_RAW.format(name=name)
    os.makedirs(track_dir, exist_ok=True)
    tmp = os.path.join(track_dir, "route.npy")
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception as e:
        raise RuntimeError(f"could not fetch {url}: {e}") from e
    route = np.load(tmp)
    if route.shape[1] != 6:
        raise RuntimeError(f"{name}: unexpected route shape {route.shape}")
    return install_track(name, route)
