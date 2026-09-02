"""Rollout dataset, served from a random-access cache.

Parquet keeps its images in memory once read, and macOS duplicates the dataset
into every worker: RAM grows with the size of the dataset. So the images are
copied once into a flat file and read back through a memmap -- the OS page cache
does the work and the workers share the same pages instead of each holding a
copy.
"""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from perception import augment

K = 4
MAX_CURVATURE = 1.0          # past this, the track polyline itself is wrong
REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE = REPO_ROOT / "data" / "cache"
CACHE_VERSION = 2      # bump when the layout of index.npz changes, so a stale
                       # cache is rebuilt instead of failing on a missing key

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

# Held out from the whole pipeline: neither the CNN nor the policy ever trains
# on these. Ten tracks spread over the full range of curvature spread -- every
# quintile is represented, so the holdout is neither all easy nor all hopeless.
HOLDOUT_TRACKS = ("2022_july_pro_v2", "2022_march_open_v2", "2022_may_pro_v2",
                  "2022_reinvent_champ_v2", "2022_summit_speedway_v2",
                  "AmericasGeneratedInclStart_v2", "Belille_v2", "dubai_open_v2",
                  "morgan_open_v2", "penbay_pro_v2")
TRAINING_TRACKS = tuple(t for t in DATASET_TRACKS if t not in HOLDOUT_TRACKS)


def track_names(tracks):
    """Drop the dataset suffix: "Monaco_v2" -> "Monaco"."""
    return tuple(t[:-3] for t in tracks)


def valid_curvatures(targets, limit=MAX_CURVATURE):
    """True for rows whose two curvature targets are both plausible.

    Some tracks have near-coincident waypoints: the circle fitted through three
    nearly identical points has a radius of a few centimetres.
    """
    return np.abs(targets[:, -2:]).max(axis=1) <= limit


def _parquet_files(track):
    return sorted((REPO_ROOT / "data" / track).glob("rollout_*.parquet"))


def _fingerprint(tracks):
    """Identify the source parquets by name, size and modification time."""
    return sorted([f.name, f.stat().st_size, int(f.stat().st_mtime)]
                  for t in tracks for f in _parquet_files(t))


def build_cache(tracks=DATASET_TRACKS):
    """Copy every image end to end into a flat file, plus an index.

    Writes nothing if the cache already matches the parquets on disk.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    stamp = CACHE / "sources.json"
    state = {"version": CACHE_VERSION, "sources": _fingerprint(tracks)}
    if stamp.exists() and json.loads(stamp.read_text()) == state:
        return

    offsets, sizes, targets, env, episode, track_id = [], [], [], [], [], []
    position = 0
    with open(CACHE / "images.bin", "wb") as blob:
        for tid, track in enumerate(tracks):
            folder = REPO_ROOT / "data" / track
            lo, hi = json.loads((folder / "meta.json").read_text())["cnn_target_slice"]
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
                track_id.append(np.full(len(df), tid))
            print(f"  cached {track}", flush=True)

    np.savez(CACHE / "index.npz",
             offsets=np.array(offsets, np.int64),
             sizes=np.array(sizes, np.int32),
             targets=np.concatenate(targets).astype(np.float32),
             env=np.concatenate(env).astype(np.int32),
             episode=np.concatenate(episode).astype(np.int32),
             track_id=np.concatenate(track_id).astype(np.int16),
             tracks=np.array(tracks))
    stamp.write_text(json.dumps(state))


class RolloutDataset(Dataset):
    """Stacks of k consecutive frames from one car, and the target of the last.

    ``augment`` adds camera jitter (see perception.augment). Off by default, so
    validation and every other caller keep reading the frames as rendered.
    """

    def __init__(self, tracks=DATASET_TRACKS, k=K, augment=False):
        build_cache()
        d = np.load(CACHE / "index.npz")
        all_tracks = list(d["tracks"])
        keep = np.isin(d["track_id"], [all_tracks.index(t) for t in tracks])

        self.k = k
        self.offsets, self.sizes = d["offsets"], d["sizes"]
        self.targets = d["targets"]
        self.augment = augment
        self._blob = None                  # opened per worker, never pickled
        self._rng = None                   # idem, one stream per worker

        env, ep, tid = d["env"], d["episode"], d["track_id"]
        curvature = valid_curvatures(self.targets)
        # a stack is usable if it stays within the same car, the same episode
        # and the same file, and if its target is not an outlier
        i = np.flatnonzero(keep)[: -(k - 1) or None]
        j = i + k - 1
        ok = ((env[i] == env[j]) & (ep[i] == ep[j]) & (tid[i] == tid[j])
              & keep[j] & curvature[j])
        self.index = i[ok]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, n):
        i = self.index[n]
        camera = None
        if self.augment:
            if self._rng is None:
                self._rng = np.random.default_rng()
            camera = augment.sample_camera(self._rng)   # one state for the stack
        x = torch.cat([self._frame(i + j, camera) for j in range(self.k)], dim=0)
        y = torch.from_numpy(self.targets[i + self.k - 1].copy())
        return x, y

    def _frame(self, row, camera=None):
        if self._blob is None:
            self._blob = np.memmap(CACHE / "images.bin", dtype=np.uint8, mode="r")
        o, size = self.offsets[row], self.sizes[row]
        img = Image.open(io.BytesIO(self._blob[o:o + size].tobytes()))
        a = np.asarray(img, dtype=np.float32) / 255.0   # (H, W, 3)
        if camera is not None:
            a = augment.apply(a, camera, self._rng)
        return torch.from_numpy(a).permute(2, 0, 1)     # (3, H, W)
