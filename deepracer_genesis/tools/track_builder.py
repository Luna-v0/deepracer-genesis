"""Build renderable tracks from waypoint routes (and fetch official ones).

A DeepRacer route file is an (W, 6) float array of [center_xy, inner_xy,
outer_xy] per waypoint. The original Gazebo track meshes exist only for a
handful of tracks — this module generates a road mesh procedurally from the
route instead, so EVERY official route (126 on aws-deepracer-community/
deepracer-race-data) and any custom-drawn track becomes drivable + renderable:

    from deepracer_genesis.tools.track_builder import fetch_official_track
    fetch_official_track("Oval_track")        # -> usable as track="Oval_track"

The generated mesh is plain OBJ + MTL colors (road surface, white border
lines, dashed centerline) — no textures, which sidesteps every Madrona
texture quirk (alpha bleed, mipmap-less aliasing, R<->G swap) and renders
identically under Madrona / Nyx / rasterizer.

Custom tracks: see build_route() + install_track() (used by
notebooks/track_designer.ipynb).
"""

from __future__ import annotations

import os
import urllib.request

from typing import Optional

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
def build_route(points_xy, half_width: float, n_waypoints: int = 150,
                smooth_passes: int = 3) -> np.ndarray:
    """Turn a rough closed polygon into a (W, 6) DeepRacer route.

    `points_xy` is any (P, 2) sequence of corner points (P >= 3), traversed
    in order and closed automatically. Chaikin corner-cutting smooths it into
    a drivable loop, arclength resampling spaces the waypoints evenly, and
    the borders are offset `half_width` along the left/right normals.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise ValueError("points_xy must be (P, 2) with P >= 3")

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
                         n_waypoints: Optional[int] = None) -> np.ndarray:
    """The DeepRacer-native way to define a track: CENTERLINE waypoints + a
    track WIDTH. No smoothing — your waypoints ARE the centerline (optionally
    resampled to `n_waypoints` for even spacing); borders are offset half the
    width along the left/right normals.

        route = route_from_waypoints([(0,0), (5,0), (5,4), (0,4)], width=1.06)
        install_track("my_square", route)
    """
    pts = np.asarray(waypoints_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise ValueError("waypoints_xy must be (P, 2) with P >= 3")
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]                        # closed automatically
    if n_waypoints:
        seg = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
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


def build_track_mesh(route: np.ndarray, out_obj: str, *,
                     line_width: float = 0.04, dash_len: float = 0.30,
                     dash_gap: float = 0.35) -> str:
    """Write a road-ribbon OBJ (road / border lines / dashed centerline)."""
    center, inner, outer = route[:, 0:2], route[:, 2:4], route[:, 4:6]
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
    with open(os.path.join(out_dir, mtl_name), "w") as f:
        for mat, rgb in _PALETTE.items():
            Image.new("RGB", (4, 4), rgb).save(os.path.join(out_dir, f"{mat}.png"))
            kd = " ".join(f"{c / 255:.4f}" for c in rgb)
            f.write(f"newmtl {mat}\nKd {kd}\nKa 0 0 0\nKs 0 0 0\n"
                    f"map_Kd {mat}.png\n")

    with open(out_obj, "w") as f:
        f.write(f"mtllib {mtl_name}\n")
        f.write("vt 0.5 0.5\n")            # single UV; textures are solid
        v = 1
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
def install_track(name: str, route: np.ndarray) -> str:
    """Persist a route + generated mesh under the assets tree and register it.

    After this (and in every later process — generated tracks are discovered
    at import), the track is usable anywhere a track name is accepted:
    `FeatureEnvironment(tracks=(name,))`, `rollout_video(..., track=name)`...
    """
    from ..envs.track import TRACKS

    route = np.asarray(route, dtype=np.float64)
    if route.ndim != 2 or route.shape[1] != 6:
        raise ValueError(f"route must be (W, 6) [center,inner,outer]; got {route.shape}")

    track_dir = os.path.join(GENERATED_DIR, name)
    os.makedirs(track_dir, exist_ok=True)
    np.save(os.path.join(track_dir, "route.npy"), route)
    build_track_mesh(route, os.path.join(track_dir, "track.obj"))

    rel = os.path.relpath(track_dir, ASSETS_DIR)
    TRACKS[name] = (f"{rel}/track.obj", f"{rel}/route.npy", None)
    return track_dir


def fetch_official_track(name: str, *, force: bool = False) -> str:
    """Download an official route from deepracer-race-data and install it.

    126 tracks are available — e.g. Oval_track, Bowtie_track, AWS_track,
    Canada_Training, China_track, Mexico_track, New_York_Track, Spain_track,
    Tokyo_Training_track, Vegas_track, Monaco, Austin, Singapore,
    arctic_open, penbay_pro, ... (see the repo for the full list).
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
