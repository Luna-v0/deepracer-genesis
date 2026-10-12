"""Deterministic ML-style train/test track split with a by-name holdout.

Depends only on (names, holdout, test_fraction, seed), not registration order.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

#: tracks with a physical counterpart — final-eval only, never in train/test
PRINTED_TRACKS = ("reinvent_base", "Oval_track")


@dataclass(frozen=True)
class TrackDataset:
    """Deterministic train/test split over track names with a by-name holdout.

    Attributes:
        names: track names to split; defaults to every registered track.
        holdout: tracks excluded from train/test (final-eval only).
        test_fraction: share of the non-holdout pool assigned to test.
        seed: seed making the split order reproducible.
        train: tracks selected for training.
        test: tracks selected for testing.
    """

    names: Sequence[str] | None = None       # default: every registered track
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
