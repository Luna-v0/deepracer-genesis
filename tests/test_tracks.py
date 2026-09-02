"""The track catalog API (deepracer_genesis.tracks)."""

import math

import numpy as np
import pytest
import torch

from deepracer_genesis import tracks
from deepracer_genesis.envs.track import TRACKS, Track


def test_names_lists_all_tracks_sorted():
    ns = tracks.names()
    assert ns == sorted(ns) and len(ns) >= 3
    assert {"reinvent_base", "Oval_track"} <= set(ns)


def test_base_and_generated_partition_names():
    assert set(tracks.base()) | set(tracks.generated()) == set(tracks.names())
    assert set(tracks.base()) & set(tracks.generated()) == set()
    assert "reinvent_base" in tracks.base()


def test_exists_and_require():
    assert tracks.exists("reinvent_base") and not tracks.exists("nope")
    assert tracks.require("reinvent_base") == "reinvent_base"
    with pytest.raises(KeyError, match="unknown track"):
        tracks.require("nope")


def test_info_metadata():
    i = tracks.info("reinvent_base")
    assert i.source == "base" and i.num_waypoints == 118
    assert 15 < i.length_m < 20 and i.avg_width_m > 0
    assert i.route_path.endswith(".npy")


def test_catalog_covers_every_track():
    assert {c.name for c in tracks.catalog()} == set(tracks.names())


@pytest.mark.parametrize("name", tracks.names())
def test_catalog_waypoint_count_matches_track(name):
    assert tracks.info(name).num_waypoints == Track(name, "cpu").n_wps


def _circle_route(n_wps: int, radius: float) -> np.ndarray:
    """Build a synthetic circular route of ``[cx, cy, ix, iy, ox, oy]`` rows.

    Args:
        n_wps: Number of evenly spaced centerline waypoints.
        radius: Centerline radius in metres.

    Returns:
        An ``(n_wps, 6)`` float32 route array.
    """
    theta = np.linspace(0.0, 2 * math.pi, n_wps, endpoint=False)
    center = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    return np.concatenate(
        [center, 0.9 * center, 1.1 * center], axis=1).astype(np.float32)


def test_track_drops_degenerate_segments(tmp_path, monkeypatch):
    route = _circle_route(24, 2.0)
    doubled = np.insert(route, 7, route[6], axis=0)  # repeat one mid-loop waypoint
    np.save(tmp_path / "route.npy", doubled)
    monkeypatch.setattr("deepracer_genesis.envs.track.ASSETS_DIR", str(tmp_path))
    monkeypatch.setitem(TRACKS, "_synthetic", ("mesh.obj", "route.npy", None))

    t = Track("_synthetic", "cpu")

    assert t.n_wps == 24
    assert torch.allclose(t.center, torch.tensor(route[:, 0:2]))
    assert torch.isfinite(t.track_yaw).all()
    assert torch.isfinite(t.curvature).all()
    # a kept duplicate leaves a zero-length segment, spiking curvature ~1e6x
    assert float(t.curvature.abs().max()) == pytest.approx(1 / 2.0, rel=0.05)
