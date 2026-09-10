"""The lookahead horizon is fixed in METERS, not waypoint indices (P1).

Waypoint spacing varies ~48x across tracks (0.017-0.888 m); index-based
lookahead gave 0.5-26.6 m horizons for one config. GPU-free (MultiTrack only).
"""

import pytest
import torch

from deepracer_genesis.envs.track import MultiTrack

K = 10
SPACING = 0.45          # the config default
HORIZON = K * SPACING   # 4.5 m on every track

# spacings 0.109 m -> 0.888 m: an 8x spread the old index walk turned into
# an 8x horizon spread
TRACKS = ("reinvent_base", "Bowtie_track", "thunder_hill_open", "H_track")


def _dists():
    return torch.arange(1, K + 1, dtype=torch.float32) * SPACING


def _polyline_len(pts):
    """(N, K, 2) -> (N,) length of the sampled polyline including the origin leg."""
    return (pts[:, 1:] - pts[:, :-1]).norm(dim=2).sum(dim=1)


@pytest.mark.parametrize("name", TRACKS)
def test_horizon_is_the_same_meters_on_every_track(name):
    track = MultiTrack((name,), num_envs=3, device="cpu")
    progress = torch.tensor([0.0, 1.0, 2.0])
    pts = track.lookahead_points_m(progress, _dists())
    # the polyline through the samples spans (k-1)*spacing of arc; chords cut
    # curves, so allow 10% under but no over
    span = _polyline_len(pts)
    lo, hi = 0.9 * (HORIZON - SPACING), (HORIZON - SPACING) + 1e-4
    assert ((span > lo) & (span < hi)).all(), f"{name}: spans {span.tolist()}"


def test_samples_are_interpolated_not_snapped():
    # H_track waypoints sit 0.888 m apart — nearest-waypoint snapping would
    # duplicate consecutive 0.45 m samples; interpolation keeps them distinct
    track = MultiTrack(("H_track",), num_envs=1, device="cpu")
    pts = track.lookahead_points_m(torch.tensor([0.0]), _dists())
    step = (pts[:, 1:] - pts[:, :-1]).norm(dim=2)
    assert (step > 0.2).all(), f"duplicated samples: steps {step.tolist()}"


def test_dir_sign_walks_backwards():
    track = MultiTrack(("reinvent_base",), num_envs=1, device="cpu")
    p = torch.tensor([5.0])
    d = torch.tensor([1.5])
    back = track.lookahead_points_m(p, d, dir_sign=torch.tensor([-1.0]))
    fwd_from_behind = track.lookahead_points_m(p - 1.5, torch.tensor([0.0]))
    assert torch.allclose(back, fwd_from_behind, atol=1e-5)


def test_wraps_across_the_finish_line():
    track = MultiTrack(("reinvent_base",), num_envs=1, device="cpu")
    L = float(track.total_len_env[0])
    near_end = torch.tensor([L - 0.1])
    pts = track.lookahead_points_m(near_end, _dists())
    assert torch.isfinite(pts).all()
    span = _polyline_len(pts)
    assert 0.9 * (HORIZON - SPACING) < float(span[0]) <= (HORIZON - SPACING) + 1e-4
