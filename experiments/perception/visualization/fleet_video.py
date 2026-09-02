"""Top-down video showing ALL the cars at once.

    python -m experiments.perception.visualization.fleet_video <ckpt> <track> [sim|cnn]
"""

import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import imageio.v2 as imageio
import torch

from deepracer_genesis.envs.features import PerceptionFeatures
from deepracer_genesis.experiment import run
from deepracer_genesis.experiment.builder import Builder
from deepracer_genesis.experiment.visualize import _rsl_actor

from deepracer_genesis.perception.features import CNNPerceptionFeatures

from experiments.perception.train_policy_with_cnn import CNNPerceptionPolicy

REPO_ROOT = Path(__file__).resolve().parents[3]

NUM_ENVS, STEPS, RESOLUTION = 16, 900, (900, 675)


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
            # the spectator pass recomposes the per-env frames into one fleet view
            video.append_data(sim.render_spectator())
            if n % 150 == 0:
                print(f"  {100*n//STEPS:3d} %", flush=True)

    print(f"{track} / {source}: {offtracks} off-tracks")
    print("open:", out)


if __name__ == "__main__":
    main()
