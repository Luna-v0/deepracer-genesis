"""Paths of the 16 cars, exact perception against the CNN, side by side.

Where the summary table gives one number per track, this shows where on the lap
the two sources diverge, and marks every point a car left the track.

    python -m perception.visualization.plot_policy_paths <path/model.pt> <track>
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deepracer_genesis.envs.features import PerceptionFeatures
from deepracer_genesis.envs.track import ASSETS_DIR, TRACKS
from deepracer_genesis.experiment import run
from deepracer_genesis.experiment.evaluator import build_single_track_sim
from deepracer_genesis.experiment.visualize import _rsl_actor

from perception.cnn_features import CNNPerceptionFeatures
from perception.dataset import REPO_ROOT
from perception.train_policy_with_cnn import CNNPerceptionPolicy

SOURCES = (("exact perception", PerceptionFeatures),
           ("through the CNN", CNNPerceptionFeatures))
NUM_ENVS, STEPS = 16, 900


def drive(spec, track, ckpt):
    """Returns (paths, offtracks): positions per step, and where cars left."""
    sim = build_single_track_sim(spec, track, NUM_ENVS)
    actor = _rsl_actor(spec, ckpt, sim)
    sim.reset_idx(torch.arange(NUM_ENVS, device=sim.device))
    xy, offtracks = [], []
    for _ in range(STEPS):
        td = actor(sim.get_observations().clone())
        sim.step(td["action"])
        pos = sim.base_pos[:, :2].cpu().numpy()
        xy.append(pos.copy())
        info = sim.step_info
        outside = (info["offtrack"] | info["flipped"]).cpu().numpy()
        if outside.any():
            offtracks.extend(pos[outside])
    return np.stack(xy), np.array(offtracks).reshape(-1, 2)


def main():
    ckpt, track = sys.argv[1], sys.argv[2]
    w = np.load(f"{ASSETS_DIR}/{TRACKS[track][1]}").astype(np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, (name, feature_set) in zip(axes, SOURCES):
        spec = run(CNNPerceptionPolicy, build_only=True,
                   feature_set=feature_set, tracks=(track,))
        xy, offtracks = drive(spec, track, ckpt)
        for edge in (w[:, 2:4], w[:, 4:6]):
            ax.plot(*np.vstack([edge, edge[:1]]).T, color="0.35", lw=1.2)
        for i in range(NUM_ENVS):
            path = xy[:, i].copy()
            # respawning after an off-track makes a jump: break the line there
            jump = np.linalg.norm(np.diff(path, axis=0), axis=1) > 0.5
            path[1:][jump] = np.nan
            ax.plot(path[:, 0], path[:, 1], lw=0.9, alpha=0.75)
        if len(offtracks):
            ax.plot(*offtracks.T, "x", color="crimson", ms=6, mew=1.6,
                    label=f"{len(offtracks)} off-tracks")
            ax.legend(loc="upper right", fontsize=9)
        ax.set_title(f"{track} - {name}", fontsize=11)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.tight_layout()
    out = REPO_ROOT / "runs" / "figures" / f"policy_paths_{track}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print("open:", out)


if __name__ == "__main__":
    main()
