"""Track catalog API: list, validate, and inspect the available tracks.

Lightweight (numpy only) — reads route .npy metadata without building a sim.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from . import ASSETS_DIR


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
        closed: Whether the route is a closed loop (a few official tracks —
            e.g. ``Straight_track`` — are open lines; lap logic and zoos
            usually want loops only).
    """

    name: str
    source: str
    mesh_path: str
    route_path: str
    num_waypoints: int
    length_m: float
    avg_width_m: float
    closed: bool = True


def names() -> list[str]:
    """Return the sorted names of every available track."""
    from .envs.track import TRACKS
    return sorted(TRACKS)


def exists(name: str) -> bool:
    """Return whether ``name`` is a registered track.

    Args:
        name: Candidate track name.

    Returns:
        True if the track is registered.
    """
    from .envs.track import TRACKS
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
    from .envs.track import TRACKS
    require(name)
    mesh_rel, route_rel, _field = TRACKS[name]
    route_path = os.path.join(ASSETS_DIR, route_rel)

    wps = np.load(route_path).astype(np.float32)
    if np.allclose(wps[0, :2], wps[-1, :2], atol=1e-6):   # AWS routes repeat wp 0
        wps = wps[:-1]
    center = wps[:, 0:2]
    seg = np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1)
    width = np.linalg.norm(wps[:, 4:6] - wps[:, 2:4], axis=1)  # outer - inner
    # a loop's closing hop is an ordinary segment; an open line's "closing"
    # hop spans the whole course
    closed = bool(seg[-1] <= 4.0 * float(np.median(seg[:-1])))
    return TrackInfo(
        name=name, source=_source(route_rel),
        mesh_path=os.path.join(ASSETS_DIR, mesh_rel), route_path=route_path,
        num_waypoints=int(center.shape[0]), length_m=float(seg.sum()),
        avg_width_m=float(width.mean()), closed=closed,
    )


def base() -> list[str]:
    """Return the sorted names of the shipped (non-generated) tracks."""
    from .envs.track import TRACKS
    return sorted(n for n, (_m, r, _f) in TRACKS.items() if _source(r) == "base")


def generated() -> list[str]:
    """Return the sorted names of the procedurally-generated tracks."""
    from .envs.track import TRACKS
    return sorted(n for n, (_m, r, _f) in TRACKS.items() if _source(r) == "generated")


def catalog() -> list[TrackInfo]:
    """Return :class:`TrackInfo` for every track, sorted by name."""
    return [info(n) for n in names()]


OFFICIAL_LIST_URL = ("https://api.github.com/repos/aws-deepracer-community/"
                     "deepracer-race-data/contents/raw_data/tracks/npy")


def official(*, refresh: bool = False) -> tuple[str, ...]:
    """The full official DeepRacer track library (~126 names).

    The list is queried once from the community race-data repository's
    directory listing and cached on disk, so later calls (and fully offline
    use) read the cache. Falls back to locally installed officially-named
    tracks when neither network nor cache is available. Individual tracks
    are downloaded on demand by
    :func:`~deepracer_genesis.tools.track_builder.fetch_official_track`.

    Args:
        refresh: Re-query the listing even when a cache exists.

    Returns:
        Sorted official track names.
    """
    import json

    cache = os.path.join(ASSETS_DIR, "tracks", "generated",
                         "_official_tracks.json")
    if not refresh and os.path.exists(cache):
        with open(cache) as f:
            return tuple(json.load(f))
    try:
        import urllib.request
        with urllib.request.urlopen(OFFICIAL_LIST_URL, timeout=20) as r:
            entries = json.load(r)
        official_names = sorted(e["name"][:-4] for e in entries
                                if e["name"].endswith(".npy"))
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w") as f:
            json.dump(official_names, f, indent=0)
        return tuple(official_names)
    except Exception as e:                              # noqa: BLE001
        local = tuple(sorted(n for n in names()
                             if not n.startswith(("rz", "donut", "reinvent"))))
        print(f"(official listing unavailable: {e} — using {len(local)} "
              "locally installed tracks)")
        return local
