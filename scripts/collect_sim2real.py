"""Collect a large, heterogeneous sim2real perception dataset.

Loops tracks in ONE process (Genesis scenes rebuild fine in-process) and
runs `collect_rollout_dataset` on each with the full DR stack — per-episode
world-color remap, per-step image aug, physics DR, random spawn + driving
direction — driven by a `NoisyExpert` agent that wanders off the centerline
(and off the track). See deepracer_genesis/experiment/agents.py to plug in
your own driving behavior.

    python scripts/collect_sim2real.py --out /mnt/models/dr_perception/sim2real_rollouts
    python scripts/collect_sim2real.py --split train        # ML track split
    python scripts/collect_sim2real.py --tracks Monaco,Austin --steps 4096

Each track lands in its own subdirectory of parquet shards (rows sorted
env-major/time-contiguous, PNG frames + aligned feature vectors + episode
ids for frame stacking); a dataset_card.json summarizes the whole set.
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/models/dr_perception/sim2real_rollouts")
    ap.add_argument("--tracks", default=None,
                    help="comma list; default = the ML track split selected by --split")
    ap.add_argument("--split", default="all", choices=["train", "test", "holdout", "all"],
                    help="which side of the TrackDataset split to collect")
    ap.add_argument("--steps", type=int, default=2048,
                    help="control steps per track (frames = steps * num_envs)")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--noise", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from deepracer_genesis.experiment import (
        CameraEnvironment,
        DomainRandomizationCamera,
        DomainRandomizationPhysics,
        DomainRandomizationTrackAppearance,
    )
    from deepracer_genesis.experiment.agents import NoisyExpert
    from deepracer_genesis.experiment.data_collection import collect_rollout_dataset
    from deepracer_genesis.tools.track_split import TrackDataset

    if args.tracks:
        tracks = args.tracks.split(",")
    else:
        ds = TrackDataset()
        tracks = {"train": ds.train, "test": ds.test,
                  "holdout": ds.holdout, "all": ds.train + ds.test}[args.split]

    os.makedirs(args.out, exist_ok=True)
    card = {"tracks": {}, "config": vars(args)}
    t_start = time.time()
    for i, track in enumerate(tracks):
        print(f"--- [{i + 1}/{len(tracks)}] {track} ---", flush=True)
        t0 = time.time()
        try:
            collect_rollout_dataset(
                CameraEnvironment(resolution=(160, 120), num_envs=args.num_envs,
                                  tracks=(track,), random_direction=True)
                >> DomainRandomizationTrackAppearance(strength=0.7)
                >> DomainRandomizationCamera(brightness=(0.6, 1.4), contrast=(0.7, 1.3),
                                             saturation=(0.5, 1.5), hue=0.08, blur=0.5,
                                             cutout=0.1, noise=0.02, camera_jitter=True)
                >> DomainRandomizationPhysics(),
                out=os.path.join(args.out, track), steps=args.steps,
                seed=args.seed + i, agent=NoisyExpert(noise=args.noise))
        except Exception as e:  # keep collecting the remaining tracks
            print(f"    FAILED: {e}", flush=True)
            card["tracks"][track] = {"status": "failed", "error": str(e)}
            continue
        meta = json.load(open(os.path.join(args.out, track, "meta.json")))
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
