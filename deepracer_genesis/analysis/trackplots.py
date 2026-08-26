"""Trajectories on tracks: the ``PlottingUtils`` port for this simulator.

The classic deepracer-analysis views, drawn over this repo's own track
geometry (routes from the track catalog): lap trajectories colored by a
telemetry column, per-waypoint aggregates painted on the centerline, and
off-track hotspot maps. Positions in telemetry are track-local (tile
offsets already subtracted), so plots and routes line up by construction.

Matplotlib is optional and loaded lazily (the ``charts.py`` convention);
every function returns the figure for the caller to save or log to
TensorBoard via ``SummaryWriter.add_figure``.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np


def _require_mpl():
    """Import matplotlib with the headless Agg backend, or explain.

    Returns:
        The ``matplotlib.pyplot`` module.

    Raises:
        ImportError: With install guidance when matplotlib is missing.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:                      # pragma: no cover
        raise ImportError(
            "trajectory plots need matplotlib (dev group / pip install "
            "matplotlib)") from e


def _route(track: str) -> np.ndarray:
    """Load a track's (W, 6) route by registry name.

    Args:
        track: Registered track name.

    Returns:
        The route array.
    """
    from .. import tracks

    return np.load(tracks.info(track).route_path)


def _outline(ax, route: np.ndarray) -> None:
    """Draw the track borders + start marker under a trajectory plot.

    Args:
        ax: Target axes.
        route: ``(W, 6)`` route array.
    """
    for cols in (slice(2, 4), slice(4, 6)):
        border = np.vstack([route[:, cols], route[:1, cols]])
        ax.plot(border[:, 0], border[:, 1], color="0.55", lw=1.0, zorder=1)
    ax.plot(*route[0, 0:2], marker="o", color="0.3", ms=4, zorder=2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_trajectories(df, track: Optional[str] = None, *,
                      color_by: str = "speed", ax=None,
                      max_episodes: Optional[int] = None, cmap="viridis"):
    """Lap trajectories over the track, colored by a telemetry column.

    The headline deepracer-analysis view: every recorded episode drawn as a
    colored line over the borders.

    Args:
        df: Telemetry DataFrame (single track, or pass ``track`` to filter).
        track: Track name to plot (required when ``df`` spans several).
        color_by: Telemetry column coloring the line (``"speed"``,
            ``"reward"``, ``"steer"``, ...).
        ax: Existing axes (a new figure is created when None).
        max_episodes: Cap on episodes drawn (all when None).
        cmap: Matplotlib colormap name.

    Returns:
        The matplotlib figure.

    Raises:
        ValueError: If the DataFrame spans several tracks and none is named.
    """
    from matplotlib.collections import LineCollection

    plt = _require_mpl()
    tracks_present = df["track"].unique()
    if track is None:
        if len(tracks_present) != 1:
            raise ValueError(f"df spans {len(tracks_present)} tracks — pass "
                             f"track=... (one of {sorted(tracks_present)})")
        track = tracks_present[0]
    d = df[df["track"] == track]
    fig = ax.figure if ax is not None else plt.subplots(figsize=(7, 6))[0]
    ax = ax if ax is not None else fig.axes[0]
    _outline(ax, _route(track))

    vmin, vmax = float(d[color_by].min()), float(d[color_by].max())
    lc = None
    episodes = d.groupby(["env", "episode"], observed=True)
    for i, (_, g) in enumerate(episodes):
        if max_episodes is not None and i >= max_episodes:
            break
        # a done row's pose is already the RESPAWN pose (the env auto-resets
        # inside step()), so drawing it puts a teleport line across the
        # track — drop it from the drawn line (aggregates still use it)
        g = g[~g["done"]]
        if len(g) < 2:
            continue
        pts = g[["x", "y"]].to_numpy().reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=cmap, zorder=3, linewidths=1.4)
        lc.set_array(g[color_by].to_numpy()[:-1])
        lc.set_clim(vmin, vmax)
        ax.add_collection(lc)
    ax.autoscale()
    if lc is not None:
        fig.colorbar(lc, ax=ax, label=color_by, shrink=0.8)
    ax.set_title(f"{track} — trajectories by {color_by}")
    fig.tight_layout()
    return fig


def plot_variant_grid(df, *, color_by: str = "speed", cols: int = 3,
                      max_episodes: Optional[int] = None):
    """One trajectory panel per zoo variant present in the telemetry.

    Args:
        df: Telemetry DataFrame spanning any number of tracks.
        color_by: Telemetry column coloring the lines.
        cols: Panels per row.
        max_episodes: Per-panel episode cap.

    Returns:
        The matplotlib figure.
    """
    plt = _require_mpl()
    names = sorted(df["track"].unique())
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows),
                             squeeze=False)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    for name, ax in zip(names, axes.flat):
        plot_trajectories(df, name, color_by=color_by, ax=ax,
                          max_episodes=max_episodes)
        ax.set_title(name, fontsize=9)
    fig.tight_layout()
    return fig


def plot_waypoint_heat(df, track: Optional[str] = None, *,
                       value: str = "speed", agg: str = "mean",
                       cmap="plasma"):
    """A per-waypoint aggregate painted on the centerline.

    The deepracer-analysis "speed per waypoint" view: every telemetry row is
    assigned to its nearest centerline waypoint, aggregated, and drawn as a
    colored centerline.

    Args:
        df: Telemetry DataFrame (single track, or pass ``track``).
        track: Track name to plot.
        value: Column to aggregate (``"speed"``, ``"reward"``, ...).
        agg: Pandas aggregation name (``"mean"``, ``"max"``, ...).
        cmap: Matplotlib colormap name.

    Returns:
        The matplotlib figure.
    """
    plt = _require_mpl()
    if track is None:
        (track,) = df["track"].unique()
    d = df[df["track"] == track]
    route = _route(track)
    center = route[:, 0:2]
    # nearest centerline waypoint per row (numpy, chunked for memory)
    xy = d[["x", "y"]].to_numpy()
    wp = np.empty(len(xy), dtype=np.int64)
    for lo in range(0, len(xy), 65536):
        chunk = xy[lo:lo + 65536]
        d2 = ((chunk[:, None, :] - center[None]) ** 2).sum(-1)
        wp[lo:lo + 65536] = d2.argmin(1)
    series = d.assign(_wp=wp).groupby("_wp")[value].agg(agg)
    values = np.full(len(center), np.nan)
    values[series.index.to_numpy()] = series.to_numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    _outline(ax, route)
    sc = ax.scatter(center[:, 0], center[:, 1], c=values, cmap=cmap, s=14,
                    zorder=3)
    fig.colorbar(sc, ax=ax, label=f"{agg} {value}", shrink=0.8)
    ax.set_title(f"{track} — {agg} {value} per waypoint")
    fig.tight_layout()
    return fig


def plot_offtrack_hotspots(df, track: Optional[str] = None):
    """Where cars leave the track — the actionable sim2real view.

    Args:
        df: Telemetry DataFrame (single track, or pass ``track``).
        track: Track name to plot.

    Returns:
        The matplotlib figure (off-track points as red crosses, count in
        the title).
    """
    plt = _require_mpl()
    if track is None:
        (track,) = df["track"].unique()
    d = df[(df["track"] == track) & df["off_track"]]
    fig, ax = plt.subplots(figsize=(7, 6))
    _outline(ax, _route(track))
    ax.plot(d["x"], d["y"], "x", color="crimson", ms=5, mew=1.4, zorder=3)
    ax.set_title(f"{track} — {len(d)} off-track points")
    fig.tight_layout()
    return fig
