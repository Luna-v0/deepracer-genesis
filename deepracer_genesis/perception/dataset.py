"""Rollout frames served as k-frame stacks from a flat memmap cache.

Parquet holds decoded images in memory per worker; copying them once into a flat
file lets every worker share the OS page cache instead.
"""

from __future__ import annotations

import io
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from deepracer_genesis.perception import augment as camera_jitter

logger = logging.getLogger(__name__)

K = 4
MAX_CURVATURE = 1.0          # past this, the track polyline itself is wrong
CACHE_VERSION = 3            # bump to invalidate caches with a stale layout

# Where collected rollouts live. Defaults to ./data so a repo-root checkout
# works unchanged; override to keep datasets outside the source tree.
DATA_ROOT = Path(os.environ.get("DEEPRACER_DATA_ROOT", "data"))
CACHE = DATA_ROOT / "cache"

DATASET_TRACKS = tuple(f"{t}_v2" for t in (
    "reinvent_base", "Oval_track", "Bowtie_track", "Monaco", "Spain_track",
    "New_York_Track", "Austin", "Singapore", "Vegas_track", "China_track",
    "Mexico_track", "Tokyo_Training_track", "Canada_Training", "AWS_track",
    "reInvent2019_track", "2022_reinvent_champ", "2022_april_open",
    "2022_april_pro", "2022_august_open", "2022_august_pro",
    "2022_july_open", "2022_july_pro", "2022_june_open", "2022_june_pro",
    "2022_march_open", "2022_march_pro", "2022_may_open", "2022_may_pro",
    "2022_october_open", "2022_october_pro", "2022_september_open",
    "2022_september_pro", "2022_summit_speedway",
    "2022_summit_speedway_mini", "Albert", "AmericasGeneratedInclStart",
    "Aragon", "Belille", "FS_June2020", "H_track", "July_2020", "LGSWide",
    "arctic_open", "arctic_pro", "caecer_gp", "caecer_loop", "dubai_open",
    "dubai_pro", "hamption_open", "hamption_pro", "jyllandsringen_open",
    "jyllandsringen_pro", "morgan_open", "morgan_pro", "penbay_open",
    "penbay_pro", "red_star_open", "red_star_pro", "thunder_hill_open",
    "thunder_hill_pro",
))

# Held out from the whole pipeline: neither the CNN nor the policy trains on
# these. Ten tracks spanning every quintile of curvature spread.
HOLDOUT_TRACKS = ("2022_july_pro_v2", "2022_march_open_v2", "2022_may_pro_v2",
                  "2022_reinvent_champ_v2", "2022_summit_speedway_v2",
                  "AmericasGeneratedInclStart_v2", "Belille_v2", "dubai_open_v2",
                  "morgan_open_v2", "penbay_pro_v2")
TRAINING_TRACKS = tuple(t for t in DATASET_TRACKS if t not in HOLDOUT_TRACKS)


def track_names(tracks: Sequence[str]) -> tuple[str, ...]:
    """Strip the dataset suffix, turning ``"Monaco_v2"`` into ``"Monaco"``.

    Args:
        tracks: Dataset track names.

    Returns:
        The corresponding registry track names.
    """
    return tuple(t[:-3] for t in tracks)


def valid_curvatures(targets: np.ndarray, limit: float = MAX_CURVATURE) -> np.ndarray:
    """Return a mask of rows whose two curvature targets are both plausible.

    Near-coincident waypoints fit a circle of a few centimetres' radius.

    Args:
        targets: ``(rows, channels)`` supervision targets, curvature last.
        limit: Largest believable absolute curvature, in 1/m.

    Returns:
        A boolean mask over rows.
    """
    return np.abs(targets[:, -2:]).max(axis=1) <= limit


def _parquet_files(track: str) -> list[Path]:
    """Return a track's rollout shards in a stable order.

    Args:
        track: Dataset track name.

    Returns:
        The sorted shard paths.
    """
    return sorted((DATA_ROOT / track).glob("rollout_*.parquet"))


def _fingerprint(tracks: Sequence[str]) -> list[list]:
    """Identify the source shards by name, size and modification time.

    Args:
        tracks: Dataset track names.

    Returns:
        A sorted list of ``[name, size, mtime]`` entries.
    """
    return sorted([f.name, f.stat().st_size, int(f.stat().st_mtime)]
                  for t in tracks for f in _parquet_files(t))


def build_cache(tracks: Sequence[str] = DATASET_TRACKS) -> None:
    """Copy every image into one flat file and write the row index beside it.

    Writes nothing when the cache already matches the shards on disk.

    Args:
        tracks: Dataset track names to include.

    Raises:
        FileNotFoundError: If a track directory has no ``meta.json``.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    stamp = CACHE / "sources.json"
    state = {"version": CACHE_VERSION, "tracks": list(tracks),
             "sources": _fingerprint(tracks)}
    if stamp.exists() and json.loads(stamp.read_text()) == state:
        return

    offsets, sizes, targets = [], [], []
    env, episode, step, track_id = [], [], [], []
    position = 0
    with open(CACHE / "images.bin", "wb") as blob:
        for tid, track in enumerate(tracks):
            meta = DATA_ROOT / track / "meta.json"
            if not meta.exists():
                raise FileNotFoundError(
                    f"{meta} is missing — collect {track} before building the cache")
            lo, hi = json.loads(meta.read_text())["cnn_target_slice"]
            for path in _parquet_files(track):
                df = pd.read_parquet(path)
                for img in df["image"]:
                    blob.write(img)
                    offsets.append(position)
                    sizes.append(len(img))
                    position += len(img)
                targets.append(np.stack(df["state"].to_numpy())[:, lo:hi])
                env.append(df["env"].to_numpy())
                episode.append(df["episode"].to_numpy())
                step.append(df["t"].to_numpy())
                track_id.append(np.full(len(df), tid))
            logger.info("cached %s", track)

    np.savez(CACHE / "index.npz",
             offsets=np.array(offsets, np.int64),
             sizes=np.array(sizes, np.int32),
             targets=np.concatenate(targets).astype(np.float32),
             env=np.concatenate(env).astype(np.int32),
             episode=np.concatenate(episode).astype(np.int32),
             step=np.concatenate(step).astype(np.int64),
             track_id=np.concatenate(track_id).astype(np.int16),
             tracks=np.array(tracks))
    stamp.write_text(json.dumps(state))


class RolloutDataset(Dataset):
    """Stacks of k consecutive frames from one car, and the target of the last.

    Camera jitter is off by default so validation reads frames as rendered.

    Attributes:
        k: Frames per stack.
        index: Row indices at which a valid stack starts.
        targets: Supervision targets for every cached row.
    """

    def __init__(self, tracks: Sequence[str] = DATASET_TRACKS, k: int = K,
                 jitter: bool = False, seed: int = 0) -> None:
        """Build the cache if needed and index every valid stack start.

        Args:
            tracks: Dataset track names to draw from.
            k: Frames per stack.
            jitter: Apply camera jitter to the training frames.
            seed: Base seed; each dataloader worker derives its own stream.

        Raises:
            ValueError: If a requested track is absent from the cache.
        """
        build_cache(tracks)
        d = np.load(CACHE / "index.npz")
        all_tracks = [str(t) for t in d["tracks"]]
        missing = set(tracks) - set(all_tracks)
        if missing:
            raise ValueError(f"tracks absent from the cache: {sorted(missing)}")

        self.k = k
        self.offsets, self.sizes = d["offsets"], d["sizes"]
        self.targets = d["targets"]
        self.jitter = jitter
        self.seed = seed
        self._blob = None                  # opened per worker, never pickled
        self._rng = None                   # idem, one stream per worker

        keep = np.isin(d["track_id"], [all_tracks.index(t) for t in tracks])
        env, ep, tid, step = d["env"], d["episode"], d["track_id"], d["step"]
        curvature = valid_curvatures(self.targets)
        # a stack is usable only if all k rows come from the same car, episode
        # and track, are consecutive in time, and end on a believable target
        i = np.flatnonzero(keep)[: -(k - 1) or None]
        j = i + k - 1
        ok = ((env[i] == env[j]) & (ep[i] == ep[j]) & (tid[i] == tid[j])
              & (step[j] - step[i] == k - 1)
              & keep[j] & curvature[j])
        self.index = i[ok]

    def __len__(self) -> int:
        """Return the number of valid stacks.

        Returns:
            The stack count.
        """
        return len(self.index)

    def __getitem__(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one frame stack and the target of its newest frame.

        Args:
            n: Index into the valid-stack list.

        Returns:
            A pair ``(frames, target)`` shaped ``(3k, H, W)`` and ``(channels,)``.
        """
        i = self.index[n]
        camera = None
        if self.jitter:
            if self._rng is None:
                info = torch.utils.data.get_worker_info()
                worker = info.id if info is not None else 0
                self._rng = np.random.default_rng([self.seed, worker])
            camera = camera_jitter.sample_camera(self._rng)   # one state per stack
        frames = torch.cat([self._frame(i + o, camera) for o in range(self.k)], dim=0)
        target = torch.from_numpy(self.targets[i + self.k - 1].copy())
        return frames, target

    def _frame(self, row: int, camera: tuple | None = None) -> torch.Tensor:
        """Decode one cached frame and apply the stack's camera state.

        Args:
            row: Cache row index.
            camera: Camera state shared by the stack, or None to skip jitter.

        Returns:
            A ``(3, H, W)`` float tensor in ``[0, 1]``.
        """
        if self._blob is None:
            self._blob = np.memmap(CACHE / "images.bin", dtype=np.uint8, mode="r")
        offset, size = self.offsets[row], self.sizes[row]
        img = Image.open(io.BytesIO(self._blob[offset:offset + size].tobytes()))
        a = np.asarray(img, dtype=np.float32) / 255.0   # (H, W, 3)
        if camera is not None:
            a = camera_jitter.apply(a, camera, self._rng)
        return torch.from_numpy(a).permute(2, 0, 1)     # (3, H, W)
