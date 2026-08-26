"""Telemetry loader/aggregates and trajectory plots (GPU-free slice).

The recorder itself needs a live env (GPU-verified separately); everything
downstream — parquet loading, per-episode aggregation, and the track plots —
is pure pandas/numpy/matplotlib and is pinned here on synthetic data.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from deepracer_genesis.analysis.telemetry import (COLUMNS, episode_agg,
                                                  load_telemetry)


def _synthetic(track: str = "reinvent_base", envs: int = 2, steps: int = 20,
               dt: float = 0.02, track_len: float = 10.0) -> pd.DataFrame:
    """A tiny telemetry DataFrame: env 0 laps cleanly, env 1 dies midway."""
    rows = []
    for t in range(steps):
        for e in range(envs):
            done = e == 1 and t == steps // 2
            episode = 0 if (e == 0 or t <= steps // 2) else 1
            rows.append({
                "step": t, "env": e, "episode": episode, "track": track,
                "x": float(np.cos(t / steps * 2 * np.pi)),
                "y": float(np.sin(t / steps * 2 * np.pi)),
                "yaw": 0.0, "steer": 0.1, "throttle": 0.5,
                "speed": 1.0 + 0.1 * e, "reward": 1.0,
                "progress_delta": track_len / steps,
                "progress_m": t * track_len / steps,
                "lateral": 0.0, "half_width": 0.3,
                "off_track": bool(done), "done": bool(done),
                "track_len": track_len, "dt": dt,
            })
    return pd.DataFrame(rows)[list(COLUMNS)]


def test_episode_agg_math():
    """Time, completion, off-track counts, and lap detection add up."""
    agg = episode_agg(_synthetic())
    e0 = agg[(agg["env"] == 0) & (agg["episode"] == 0)].iloc[0]
    assert e0["steps"] == 20
    assert e0["time_s"] == pytest.approx(20 * 0.02)
    assert e0["progress_total_m"] == pytest.approx(10.0)
    assert e0["completion"] == pytest.approx(1.0)
    assert bool(e0["completed_lap"])
    assert e0["off_track_steps"] == 0
    e1a = agg[(agg["env"] == 1) & (agg["episode"] == 0)].iloc[0]
    assert e1a["off_track_steps"] == 1
    assert not bool(e1a["completed_lap"])


def test_load_telemetry_directory_concatenates_with_source(tmp_path):
    """Directory mode merges every parquet and tags the file stem."""
    df = _synthetic()
    for name in ("eval_0000000001", "eval_0000000002"):
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                       tmp_path / f"{name}.parquet")
    merged = load_telemetry(str(tmp_path))
    assert len(merged) == 2 * len(df)
    assert set(merged["source"].unique()) == {"eval_0000000001",
                                              "eval_0000000002"}
    with pytest.raises(FileNotFoundError):
        load_telemetry(str(tmp_path / "empty"))


@pytest.fixture()
def _route_registry(tmp_path, monkeypatch):
    """A registered synthetic circular track the plots can resolve."""
    import deepracer_genesis.envs.track as track_mod
    import deepracer_genesis.tracks as catalog_mod
    from deepracer_genesis.tools.track_builder import build_route

    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    pts = np.stack([1.2 * np.cos(ang), 1.2 * np.sin(ang)], axis=1)
    route = build_route(pts, half_width=0.3)
    d = tmp_path / "tracks" / "generated" / "synth"
    d.mkdir(parents=True)
    np.save(d / "route.npy", route)
    monkeypatch.setattr(track_mod, "ASSETS_DIR", str(tmp_path))
    monkeypatch.setattr(catalog_mod, "ASSETS_DIR", str(tmp_path))
    monkeypatch.setitem(track_mod.TRACKS, "synth",
                        ("tracks/generated/synth/track.obj",
                         "tracks/generated/synth/route.npy", None))
    yield
    track_mod.TRACKS.pop("synth", None)


def test_trackplots_render_all_views(tmp_path, _route_registry):
    """Every plot type renders and saves on synthetic data."""
    from deepracer_genesis.analysis import trackplots as tp

    df = _synthetic(track="synth")
    fig = tp.plot_trajectories(df, "synth", color_by="speed")
    fig.savefig(tmp_path / "traj.png")
    fig = tp.plot_variant_grid(df, color_by="reward", cols=2)
    fig.savefig(tmp_path / "grid.png")
    fig = tp.plot_waypoint_heat(df, "synth", value="speed")
    fig.savefig(tmp_path / "heat.png")
    fig = tp.plot_offtrack_hotspots(df, "synth")
    fig.savefig(tmp_path / "hot.png")
    assert all((tmp_path / n).stat().st_size > 0
               for n in ("traj.png", "grid.png", "heat.png", "hot.png"))


def test_plot_trajectories_requires_track_when_ambiguous(_route_registry):
    """A multi-track DataFrame without track= is refused, not guessed."""
    from deepracer_genesis.analysis import trackplots as tp

    df = pd.concat([_synthetic(track="synth"),
                    _synthetic(track="other")], ignore_index=True)
    with pytest.raises(ValueError, match="spans 2 tracks"):
        tp.plot_trajectories(df)
