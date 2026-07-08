"""ML-style track splits: train/test + a by-name holdout.

Treat tracks like a dataset: a deterministic train/test split over every
registered track, with named tracks held out entirely (e.g. the ones you
physically printed) for post-training real-world-proxy evaluation:

    from deepracer_genesis.datasets import TrackDataset

    ds = TrackDataset()                       # default holdout: the printed ones
    ds = TrackDataset(holdout=("reinvent_base", "Oval_track", "Monaco"),
                      test_fraction=0.25, seed=7)

    ds.train      # e.g. train a multi-track curriculum / collect training data
    ds.test       # unseen-track generalization during development
    ds.holdout    # NEVER trained or tuned on — final evaluation only

The split is deterministic in (names, holdout, test_fraction, seed) and
independent of registration order.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional, Sequence

#: tracks with a physical counterpart — final-eval only, never in train/test
PRINTED_TRACKS = ("reinvent_base", "Oval_track")


@dataclass(frozen=True)
class TrackDataset:
    names: Optional[Sequence[str]] = None       # default: every registered track
    holdout: Sequence[str] = PRINTED_TRACKS
    test_fraction: float = 0.2
    seed: int = 0

    train: tuple[str, ...] = field(init=False)
    test: tuple[str, ...] = field(init=False)

    def __post_init__(self):
        if self.names is None:
            from ..envs.track import TRACKS
            names = sorted(TRACKS)
        else:
            names = sorted(self.names)

        missing = set(self.holdout) - set(names)
        if missing:
            raise ValueError(f"holdout tracks not registered: {sorted(missing)}")
        pool = [n for n in names if n not in set(self.holdout)]

        # deterministic order: hash each name with the seed, sort by digest
        # (stable under adding/removing OTHER tracks, unlike a shuffled index)
        def key(name: str) -> str:
            return hashlib.sha1(f"{self.seed}:{name}".encode()).hexdigest()

        ranked = sorted(pool, key=key)
        n_test = max(1, round(len(ranked) * self.test_fraction)) if ranked else 0
        object.__setattr__(self, "test", tuple(sorted(ranked[:n_test])))
        object.__setattr__(self, "train", tuple(sorted(ranked[n_test:])))

    def __repr__(self) -> str:
        return (f"TrackDataset(train={len(self.train)}, test={len(self.test)}, "
                f"holdout={list(self.holdout)})")
