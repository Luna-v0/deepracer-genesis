"""Track catalog API: list, validate, and inspect the available tracks.

Lightweight (numpy only) — reads route .npy metadata without building a sim.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from deepracer_genesis import ASSETS_DIR


@dataclass(frozen=True)
class TrackInfo:
    """Metadata for one registered track.

    Attributes:
        name: Registry name (pass this as a track wherever a name is expected).
        source: "base" (shipped DAE) or "generated" (procedural OBJ).
        mesh_path: Absolute path to the track mesh.
        route_path: Absolute path to the centerline route .npy.
        num_waypoints: Centerline waypoint count.
        length_m: Centerline loop length in metres.
        avg_width_m: Mean track width in metres.
    """

    name: str
    source: str
    mesh_path: str
    route_path: str
    num_waypoints: int
    length_m: float
    avg_width_m: float


def names() -> list[str]:
    """Return the sorted names of every available track."""
    from deepracer_genesis.envs.track import TRACKS
    return sorted(TRACKS)


def exists(name: str) -> bool:
    """Return whether ``name`` is a registered track.

    Args:
        name: Candidate track name.

    Returns:
        True if the track is registered.
    """
    from deepracer_genesis.envs.track import TRACKS
    return name in TRACKS


def require(name: str) -> str:
    """Return ``name`` if registered, else raise listing the available tracks.

    Args:
        name: Track name to validate.

    Returns:
        The same ``name``, unchanged.

    Raises:
        KeyError: If ``name`` is not a registered track.
    """
    if not exists(name):
        raise KeyError(f"unknown track {name!r}; available: {names()}")
    return name


def _source(route_rel: str) -> str:
    """Classify a track as generated (procedural) or base by its route path."""
    return "generated" if "generated" in route_rel.split(os.sep) else "base"


def info(name: str) -> TrackInfo:
    """Return metadata for ``name`` (paths, source, length, waypoints, width).

    Args:
        name: A registered track name.

    Returns:
        The track's :class:`TrackInfo`.

    Raises:
        KeyError: If ``name`` is not a registered track.
    """
    from deepracer_genesis.envs.track import TRACKS, load_route
    require(name)
    mesh_rel, route_rel, _field = TRACKS[name]
    route_path = os.path.join(ASSETS_DIR, route_rel)

    # Same loader Track uses, so the catalog counts the waypoints the env drives.
    wps = load_route(route_path)
    center = wps[:, 0:2]
    seg = np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1)
    width = np.linalg.norm(wps[:, 4:6] - wps[:, 2:4], axis=1)  # outer - inner
    return TrackInfo(
        name=name, source=_source(route_rel),
        mesh_path=os.path.join(ASSETS_DIR, mesh_rel), route_path=route_path,
        num_waypoints=int(center.shape[0]), length_m=float(seg.sum()),
        avg_width_m=float(width.mean()),
    )


def base() -> list[str]:
    """Return the sorted names of the shipped (non-generated) tracks."""
    from deepracer_genesis.envs.track import TRACKS
    return sorted(n for n, (_m, r, _f) in TRACKS.items() if _source(r) == "base")


def generated() -> list[str]:
    """Return the sorted names of the procedurally-generated tracks."""
    from deepracer_genesis.envs.track import TRACKS
    return sorted(n for n, (_m, r, _f) in TRACKS.items() if _source(r) == "generated")


def catalog() -> list[TrackInfo]:
    """Return :class:`TrackInfo` for every track, sorted by name."""
    return [info(n) for n in names()]
