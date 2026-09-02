"""Evaluate the CNN-fed policy on many tracks, and report each track's geometry.

Answers "where does it fail, and why": the summary is sorted by off-track rate
and carries the length, width, waypoint spacing and curvature spread of every
track, including ones the policy never trained on.

Genesis will not rebuild a camera scene inside one process, so each track runs
in its own subprocess.

    python -m perception.evaluation.sweep_tracks <path/model.pt> [track count]
"""

import json
import math
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from perception.dataset import DATASET_TRACKS, REPO_ROOT

NUM_ENVS = 16
MODULE = "perception.evaluation.sweep_tracks"


def evaluate(ckpt, track):
    from deepracer_genesis.experiment import run
    from deepracer_genesis.experiment.evaluator import (build_single_track_sim,
                                                        evaluate_on_tracks)
    from deepracer_genesis.experiment.visualize import _rsl_actor

    from perception.cnn_features import CNNPerceptionFeatures
    from perception.train_policy_with_cnn import CNNPerceptionPolicy

    spec = run(CNNPerceptionPolicy, build_only=True,
               feature_set=CNNPerceptionFeatures, tracks=(track,), num_envs=NUM_ENVS)
    sim = build_single_track_sim(spec, track, NUM_ENVS)
    actor = _rsl_actor(spec, ckpt, sim)
    return evaluate_on_tracks(actor, (track,), sim_factory=lambda t, s=sim: s)[track]


def track_geometry(track):
    """Length, width, waypoint spacing and curvature spread of one track."""
    from deepracer_genesis.envs.track import ASSETS_DIR, TRACKS

    w = np.load(f"{ASSETS_DIR}/{TRACKS[track][1]}").astype(np.float64)
    c = w[:, :2]
    if np.allclose(c[0], c[-1], atol=1e-6):
        w, c = w[:-1], c[:-1]
    seg = np.linalg.norm(np.roll(c, -1, 0) - c, axis=1)
    yaw = np.arctan2(*(np.roll(c, -1, 0) - c).T[::-1])
    dyaw = (np.roll(yaw, -1) - yaw + math.pi) % (2 * math.pi) - math.pi
    k = dyaw / np.maximum(seg, 1e-6) / 2.5
    return {"length_m": seg.sum(),
            "width_m": float(np.linalg.norm(w[:, 4:6] - w[:, 2:4], axis=1).mean()),
            "waypoint_step_m": float(seg[seg > 1e-3].mean()),
            "k_std": float(k[np.abs(k) < 1].std())}


def main():
    ckpt = sys.argv[1]
    if "--one" in sys.argv:
        track = sys.argv[sys.argv.index("--one") + 1]
        print("RESULT " + json.dumps(evaluate(ckpt, track)))
        return

    from perception.train_policy_with_noise import TRAIN_TRACKS as SEEN

    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    all_tracks = [t[:-3] for t in DATASET_TRACKS]
    tracks = [all_tracks[i] for i in np.linspace(0, len(all_tracks) - 1, n).astype(int)]

    rows = []
    for i, track in enumerate(tracks, 1):
        r = subprocess.run([sys.executable, "-m", MODULE, ckpt, "--one", track],
                           capture_output=True, text=True, cwd=str(REPO_ROOT))
        line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
        if line is None:
            print(f"  {i:2}/{n}  {track:28} FAILED", flush=True)
            continue
        m = json.loads(line[7:])
        m.update(track_geometry(track), track=track, seen=track in SEEN)
        rows.append(m)
        print(f"  {i:2}/{n}  {track:28} offtrack {m['offtrack_rate']:.2f}", flush=True)

    rows.sort(key=lambda m: m["offtrack_rate"])
    print(f"\n{'track':28} {'seen':>5} {'length':>7} {'width':>6} {'wp step':>8} "
          f"{'k std':>6} {'offtrack':>9} {'progress':>9} {'laps':>6}")
    for m in rows:
        print(f"{m['track']:28} {'yes' if m['seen'] else '-':>5} {m['length_m']:6.0f}m "
              f"{m['width_m']:6.2f} {m['waypoint_step_m']:8.3f} {m['k_std']:6.3f} "
              f"{m['offtrack_rate']:9.2f} {m['mean_progress_m']:8.1f}m {m['mean_laps']:6.2f}")

    out = REPO_ROOT / "runs" / "sweep_tracks.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    unseen = [m for m in rows if not m["seen"]]
    print(f"\n{len(rows)} tracks, {len(unseen)} of them never seen by the policy")
    print(f"mean offtrack rate: {np.mean([m['offtrack_rate'] for m in rows]):.2f}  "
          f"(unseen tracks only: {np.mean([m['offtrack_rate'] for m in unseen]):.2f})")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
