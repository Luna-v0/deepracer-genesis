"""Zoo manifests: declare exactly the world you want — then RUN THIS FILE.

The repo idiom is configurations, not command lines: a manifest is an
ordinary Python file declaring :class:`~deepracer_genesis.tools.zoo.Zoo`
objects and ending with the action to take. Execute it directly::

    uv run examples/zoos.py          # opens the viewer with cars driving

Edit the ``__main__`` block to change what happens (``view`` for the bare
scene grid, ``watch`` for cars + saved car-view photos, ``compile_zoo`` to
just bake and get the names for training).

A zoo mixes three kinds of sources, expanded in declaration order:

- :class:`OfficialSample` — a seeded sample of the ORIGINAL DeepRacer track
  library (~126 tracks, fetched + cached on first use), optionally with
  smooth waypoint noise and randomized looks (palette / field / wall).
- :class:`RandomShapes` — fully synthetic procedural circuits.
- :class:`TrackVariant` — one explicit, hand-picked variant.

Everything is deterministic in its seeds: the same manifest compiles to the
same population byte for byte, and the compiled names plumb straight into
training via ``CameraEnvironment(tracks=...)``.
"""

from deepracer_genesis.tools.zoo import (OfficialSample, RandomShapes,
                                         TrackVariant, Zoo, watch)

#: the recommended DR population: real circuits, geometrically noised,
#: visually randomized — plus a few synthetic wildcards
full_dr = Zoo("full_dr", (
    OfficialSample(24, seed=7, jitter=0.4, looks=True),
    RandomShapes(8, seed=7),
))

#: pristine originals only — no noise, no recolor (eval-style world)
originals = Zoo("originals", (
    OfficialSample(16, seed=0, jitter=0.0, looks=False),
))

#: hand-picked showcase: explicit variants, every axis exercised
showcase = Zoo("showcase", (
    TrackVariant("reinvent_base"),
    TrackVariant("reinvent_base", width=1.15, palette="dusk", wall="dark"),
    TrackVariant("reinvent_base", palette="asphalt_light", field="sand",
                 wall="white"),
    OfficialSample(names=("Monaco", "Bowtie_track"), jitter=0.0, looks=False,
                   fetch=True),
))


if __name__ == "__main__":
    watch(full_dr, num_envs=32)
