"""Geometry invariants for the procedural track builder."""

import numpy as np
import pytest

from deepracer_genesis.tools.track_builder import (
    build_route,
    route_from_waypoints,
)

# A 3x1 rectangle listed clockwise and counterclockwise (same shape/path).
_RECT_CW = [(0.0, 0.0), (0.0, 1.0), (3.0, 1.0), (3.0, 0.0)]
_RECT_CCW = [(0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (0.0, 1.0)]


def _poly_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


@pytest.mark.parametrize("waypoints", [_RECT_CW, _RECT_CCW])
def test_inner_border_is_the_interior_regardless_of_winding(waypoints):
    """`inner` (cols 2:4) must enclose less area than `outer` for either winding.

    Regression: clockwise waypoints previously swapped the borders, hiding the
    road under the infield fill.
    """
    route = route_from_waypoints(waypoints, width=0.5)
    inner, outer = route[:, 2:4], route[:, 4:6]
    assert _poly_area(inner) < _poly_area(outer)


@pytest.mark.parametrize(
    "builder",
    [lambda w: route_from_waypoints(w, width=0.5),
     lambda w: build_route(w, half_width=0.25)],
)
def test_both_windings_yield_the_same_border_areas(builder):
    """CW and CCW inputs of one shape produce the same ribbon geometry."""
    cw, ccw = builder(_RECT_CW), builder(_RECT_CCW)
    assert _poly_area(cw[:, 2:4]) == pytest.approx(_poly_area(ccw[:, 2:4]), rel=0.02)
    assert _poly_area(cw[:, 4:6]) == pytest.approx(_poly_area(ccw[:, 4:6]), rel=0.02)


def _dash_vertices(obj_path):
    """(N, 2) xy of every centerline-dash vertex in a baked OBJ."""
    verts, in_dashes = [], False
    for line in open(obj_path):
        if line.startswith("usemtl"):
            in_dashes = line.strip() == "usemtl centerline"
        elif in_dashes and line.startswith("v "):
            x, y, _ = line.split()[1:4]
            verts.append((float(x), float(y)))
    return np.asarray(verts)


def _dist_to_polyline(points, poly):
    """Min distance of each point to a closed polyline's segments."""
    a, b = poly, np.roll(poly, -1, axis=0)          # (W, 2) segment ends
    ab = b - a
    ap = points[:, None, :] - a[None, :, :]          # (N, W, 2)
    t = np.clip((ap * ab[None]).sum(-1) / (ab * ab).sum(-1)[None], 0.0, 1.0)
    closest = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(points[:, None, :] - closest, axis=-1).min(axis=1)


def test_baked_dashes_stay_on_the_road_of_a_coarse_track(tmp_path):
    """Dash endpoints must interpolate between waypoints: tangent
    extrapolation walked dashes off coarse tracks (~0.9 m spacing)."""
    from deepracer_genesis.tools.track_builder import build_track_mesh

    n, radius, half_width = 16, 3.0, 0.15   # ~1.2 m spacing, narrow road
    theta = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    center = radius * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    inward = -np.stack([np.cos(theta), np.sin(theta)], axis=1)
    route = np.concatenate([center, center + inward * half_width,
                            center - inward * half_width], axis=1)
    obj = build_track_mesh(route, str(tmp_path / "coarse.obj"))

    dashes = _dash_vertices(obj)
    assert len(dashes) >= 4
    off = _dist_to_polyline(dashes, center)
    # dash half-thickness is line_width*1.2 = 0.048; anything beyond the
    # road half-width means the dash left the asphalt
    assert off.max() <= half_width, f"dash vertex {off.max():.3f} m off-center"
