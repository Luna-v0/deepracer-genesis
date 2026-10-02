"""Plan B width variants: scale_route_width math and width_variants baking.

Pure math plus tmp-dir filesystem checks — no sim, no genesis. The rulebook
check goes through ``envs.track.Track`` (torch/numpy only, safe to import).
"""

import os

import numpy as np
import pytest
import torch

from deepracer_genesis.envs import track as track_mod
from deepracer_genesis.envs.track import TRACKS, Track
from deepracer_genesis.tools import track_builder
from deepracer_genesis.tools.track_builder import scale_route_width, width_variants


def _ellipse_route(n: int = 64) -> np.ndarray:
    """Closed ellipse route with per-waypoint width variation.

    Args:
        n: Number of waypoints (no repeated closing point).

    Returns:
        (n, 6) float64 route of [center, inner, outer] whose half_width
        (0.5 * |outer - inner|) varies per waypoint as 0.4 + 0.15*sin(3t).
    """
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    center = np.stack([3.0 * np.cos(t), 1.5 * np.sin(t)], axis=1)
    nrm = np.stack([1.5 * np.cos(t), 3.0 * np.sin(t)], axis=1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    hw = 0.4 + 0.15 * np.sin(3.0 * t)
    inner = center + nrm * hw[:, None]
    outer = center - nrm * hw[:, None]
    return np.concatenate([center, inner, outer], axis=1)


def _half_width(route: np.ndarray) -> np.ndarray:
    """Per-waypoint half width, exactly as track.py derives it (0.5*|outer-inner|)."""
    return 0.5 * np.linalg.norm(route[:, 4:6] - route[:, 2:4], axis=1)


def _arclength(route: np.ndarray) -> float:
    """Total closed centerline arclength of a (W, 6) route."""
    center = route[:, 0:2]
    return float(np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1).sum())


# ---------------------------------------------------------- scale_route_width
@pytest.mark.parametrize("scale", [0.5, 0.9, 1.15, 2.0])
def test_half_width_scales_borders_only(scale):
    """Borders scale about the centerline; everything else is untouched."""
    route = _ellipse_route()
    out = scale_route_width(route, scale)
    assert np.allclose(_half_width(out), scale * _half_width(route), rtol=1e-12)
    assert np.array_equal(out[:, 0:2], route[:, 0:2])  # center bit-identical
    assert out.shape == route.shape
    assert _arclength(out) == _arclength(route)


def test_input_route_is_not_mutated():
    route = _ellipse_route()
    before = route.copy()
    scale_route_width(route, 1.7)
    assert np.array_equal(route, before)


def test_scale_one_returns_equal_copy_not_same_object():
    route = _ellipse_route()
    out = scale_route_width(route, 1.0)
    assert out is not route
    assert np.array_equal(out, route)


@pytest.mark.parametrize("bad", [np.zeros((10, 5)), np.zeros(12), np.zeros((4, 2, 3))])
def test_rejects_non_w6_routes(bad):
    with pytest.raises(ValueError, match=r"\(W, 6\)"):
        scale_route_width(bad, 1.1)


@pytest.mark.parametrize("scale", [0.0, -0.5])
def test_rejects_nonpositive_scale(scale):
    with pytest.raises(ValueError, match="> 0"):
        scale_route_width(_ellipse_route(), scale)


# -------------------------------------------------------------- width_variants
@pytest.fixture
def registry():
    """Snapshot the shared TRACKS registry and restore it after the test.

    Yields:
        The live TRACKS dict (tests may add entries freely).
    """
    snapshot = dict(TRACKS)
    yield TRACKS
    TRACKS.clear()
    TRACKS.update(snapshot)


@pytest.fixture
def sandbox(tmp_path, monkeypatch, registry):
    """Point every asset root at a tmp dir and register a fake source track.

    Monkeypatches ``track_builder.ASSETS_DIR``/``GENERATED_DIR`` (bake side)
    and ``envs.track.ASSETS_DIR`` (load side) to the same tmp root, then saves
    a synthetic variable-width route as registered track ``"src"``.

    Yields:
        (assets_root, generated_root, source_route) as (str, str, ndarray).
    """
    assets = tmp_path / "assets"
    generated = assets / "tracks" / "generated"
    (assets / "routes").mkdir(parents=True)
    monkeypatch.setattr(track_builder, "ASSETS_DIR", str(assets))
    monkeypatch.setattr(track_builder, "GENERATED_DIR", str(generated))
    monkeypatch.setattr(track_mod, "ASSETS_DIR", str(assets))
    route = _ellipse_route()
    np.save(assets / "routes" / "src.npy", route)
    registry["src"] = ("routes/src.obj", "routes/src.npy", None)
    yield str(assets), str(generated), route


def test_names_ordering_and_format(sandbox):
    """Names come back one per scale, in order; 1.0 reuses the source name."""
    _assets, generated, _route = sandbox
    names = width_variants("src", (0.9, 1.0, 1.15))
    assert names == ("src_w090", "src", "src_w115")
    # no duplicate bake of the source track under any name
    assert not os.path.exists(os.path.join(generated, "src"))
    assert not os.path.exists(os.path.join(generated, "src_w100"))


def test_variants_are_baked_and_registered(sandbox):
    """Each variant gets route+mesh assets, a TRACKS entry, and scaled widths."""
    _assets, generated, route = sandbox
    width_variants("src", (0.9, 1.15))
    for name, scale in (("src_w090", 0.9), ("src_w115", 1.15)):
        track_dir = os.path.join(generated, name)
        route_path = os.path.join(track_dir, "route.npy")
        assert os.path.exists(route_path)
        assert os.path.exists(os.path.join(track_dir, "track.obj"))
        rel = f"tracks/generated/{name}"
        assert TRACKS[name] == (f"{rel}/track.obj", f"{rel}/route.npy", None)
        baked = np.load(route_path)
        assert np.allclose(_half_width(baked), scale * _half_width(route),
                           rtol=1e-12)


def test_existing_variants_are_not_rebaked_unless_forced(sandbox):
    """A repeat call reuses baked assets; force=True rewrites them."""
    _assets, generated, _route = sandbox
    width_variants("src", (0.9,))
    route_path = os.path.join(generated, "src_w090", "route.npy")
    obj_path = os.path.join(generated, "src_w090", "track.obj")
    stale = os.path.getmtime(route_path) - 1000.0
    for p in (route_path, obj_path):
        os.utime(p, (stale, stale))

    assert width_variants("src", (0.9,)) == ("src_w090",)
    assert os.path.getmtime(route_path) == stale
    assert os.path.getmtime(obj_path) == stale

    assert width_variants("src", (0.9,), force=True) == ("src_w090",)
    assert os.path.getmtime(route_path) > stale
    assert os.path.getmtime(obj_path) > stale


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_width_variants_rejects_nonpositive_scale(sandbox, scale):
    with pytest.raises(ValueError, match="> 0"):
        width_variants("src", (scale,))


# ------------------------------------------------------- rulebook follows mesh
def test_track_rulebook_follows_variant_width(sandbox):
    """Track.half_width of a baked variant is scale * the source's, per waypoint."""
    scale = 1.15
    (variant,) = width_variants("src", (scale,))
    src = Track("src", device="cpu")
    var = Track(variant, device="cpu")
    assert var.n_wps == src.n_wps
    assert torch.allclose(var.half_width, scale * src.half_width, atol=1e-6)
    # centerline (and thus arclength/spawn geometry) is shared bit-for-bit
    assert torch.equal(var.center, src.center)
