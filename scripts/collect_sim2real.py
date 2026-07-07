"""Collect a large, heterogeneous sim2real perception dataset.

Loops tracks in subprocesses (Genesis builds one scene per process) and runs
`collect_rollout_dataset` on each with the full DR stack: per-episode
world-color remap, per-step image aug (brightness/contrast/saturation/hue/
blur/cutout/noise), physics DR, random spawn + driving direction, and a
noisy privileged expert that wanders off the centerline (and off the track).

    python scripts/collect_sim2real.py --out /mnt/models/dr_perception/sim2real_rollouts
    python scripts/collect_sim2real.py --steps 4096 --num-envs 64 --tracks Monaco,Austin

Each track lands in its own subdirectory of parquet shards (rows sorted
env-major/time-contiguous, PNG frames + aligned feature vectors + episode
ids for frame stacking); a dataset_card.json summarizes the whole set.
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DEFAULT_TRACKS = [
    "reinvent_base", "reInvent2019_track", "2022_reinvent_champ",
    "Oval_track", "Bowtie_track", "AWS_track", "Canada_Training",
    "New_York_Track", "Spain_track", "Vegas_track", "Monaco", "Austin",
]

_CHILD = """
import experiments  # noqa: F401
from deepracer_genesis.experiment import (CameraEnvironment,
    DomainRandomizationCamera, DomainRandomizationPhysics,
    DomainRandomizationTrackAppearance)
from deepracer_genesis.experiment.data_collection import collect_rollout_dataset

collect_rollout_dataset(
    CameraEnvironment(resolution=(160, 120), num_envs={num_envs},
                      tracks=("{track}",), random_direction=True)
    >> DomainRandomizationTrackAppearance(strength=0.7)
    >> DomainRandomizationCamera(brightness=(0.6, 1.4), contrast=(0.7, 1.3),
                                 saturation=(0.5, 1.5), hue=0.08, blur=0.5,
                                 cutout=0.1, noise=0.02, camera_jitter=True)
    >> DomainRandomizationPhysics(),
    out={out!r}, steps={steps}, seed={seed}, noise={noise})
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/models/dr_perception/sim2real_rollouts")
    ap.add_argument("--tracks", default=",".join(DEFAULT_TRACKS))
    ap.add_argument("--steps", type=int, default=2048,
                    help="control steps per track (frames = steps * num_envs)")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--noise", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tracks = args.tracks.split(",")
    os.makedirs(args.out, exist_ok=True)
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

    card = {"tracks": {}, "config": vars(args)}
    t_start = time.time()
    for i, track in enumerate(tracks):
        out = os.path.join(args.out, track)
        code = _CHILD.format(track=track, num_envs=args.num_envs, out=out,
                             steps=args.steps, seed=args.seed + i,
                             noise=args.noise)
        print(f"--- [{i + 1}/{len(tracks)}] {track} ---", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, "-u", "-c", code], cwd=ROOT, env=env,
                           capture_output=True, text=True)
        tail = "\n".join(r.stdout.strip().splitlines()[-2:])
        print(tail or r.stderr[-800:], flush=True)
        if r.returncode != 0:
            print(f"    FAILED (exit {r.returncode})\n{r.stderr[-1500:]}", flush=True)
            card["tracks"][track] = {"status": "failed"}
            continue
        meta = json.load(open(os.path.join(out, "meta.json")))
        card["tracks"][track] = {"status": "ok", "frames": meta["frames"],
                                 "wall_s": round(time.time() - t0, 1)}

    card["total_frames"] = sum(t.get("frames", 0) for t in card["tracks"].values())
    card["wall_clock_s"] = round(time.time() - t_start, 1)
    with open(os.path.join(args.out, "dataset_card.json"), "w") as f:
        json.dump(card, f, indent=2)
    print(f"\nDONE: {card['total_frames']} frames across "
          f"{sum(1 for t in card['tracks'].values() if t['status'] == 'ok')} tracks "
          f"in {card['wall_clock_s']}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
