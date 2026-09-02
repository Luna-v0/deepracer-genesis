"""Top-down video showing ALL the cars at once.

The batched camera draws each environment in isolation, so one rendered frame
holds a single car. We recompose them: the background is the median of the N
frames (the empty track, since the cars are all in different places), then each
car is pasted back wherever its own frame departs from that background.

    python -m perception.visualization.fleet_video <path/model.pt> <track> [sim|cnn]
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import imageio.v2 as imageio
import numpy as np
import torch

from deepracer_genesis.envs.features import PerceptionFeatures
from deepracer_genesis.experiment import run
from deepracer_genesis.experiment.builder import Builder
from deepracer_genesis.experiment.visualize import _rsl_actor

from perception.cnn_features import CNNPerceptionFeatures
from perception.dataset import REPO_ROOT
from perception.train_policy_with_cnn import CNNPerceptionPolicy

NUM_ENVS, STEPS, RESOLUTION = 16, 900, (900, 675)
THRESHOLD = 28      # departure from the background, in levels, that reads as a car


def compose(frames):
    """Paste the cars from the N frames onto one empty-track background."""
    background = np.median(frames, axis=0).astype(np.uint8)
    out = background.copy()
    delta = np.abs(frames.astype(np.int16) - background).sum(-1)     # (N, H, W)
    for i in range(len(frames)):
        m = delta[i] > THRESHOLD
        out[m] = frames[i][m]
    return out


def main():
    ckpt, track = sys.argv[1], sys.argv[2]
    source = sys.argv[3] if len(sys.argv) > 3 else "cnn"
    feature_set = CNNPerceptionFeatures if source == "cnn" else PerceptionFeatures

    spec = run(CNNPerceptionPolicy, build_only=True, feature_set=feature_set,
               tracks=(track,), num_envs=NUM_ENVS)
    sim = Builder(spec).sim(extra_cfg={"vision": {"spectator": True,
                                                  "spectator_res": RESOLUTION}})
    actor = _rsl_actor(spec, ckpt, sim)
    sim.reset_idx(torch.arange(NUM_ENVS, device=sim.device))
    sim._post_physics(torch.arange(NUM_ENVS, device=sim.device))

    folder = REPO_ROOT / "runs" / "videos" / source
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"fleet_{track}.mp4"

    offtracks = 0
    with imageio.get_writer(out, fps=50) as video, torch.no_grad():
        for n in range(STEPS):
            td = actor(sim.get_observations().clone())
            sim.step(td["action"])
            info = sim.step_info
            offtracks += int((info["offtrack"] | info["flipped"]).sum())
            raw = np.asarray(sim.renderer.spec_cam.render(rgb=True)[0])
            video.append_data(compose(np.ascontiguousarray(raw)))
            if n % 150 == 0:
                print(f"  {100*n//STEPS:3d} %", flush=True)

    print(f"{track} / {source}: {offtracks} off-tracks")
    print("open:", out)


if __name__ == "__main__":
    main()
