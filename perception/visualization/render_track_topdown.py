"""Rendered image of a whole track seen from above.

    python -m perception.visualization.render_track_topdown <track>
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
from PIL import Image

import genesis as gs
from deepracer_genesis.configs.cfgs import get_env_cfg
from deepracer_genesis.envs.base_env import DeepRacerEnv

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    track = sys.argv[1].replace("_v2", "").replace("_dr", "")
    gs.init(backend=gs.cpu, logging_level="warning")

    cfg = get_env_cfg(vision=True, track=track, backend="cpu")
    cfg["vision"]["vision_renderer"] = "rasterizer"   # mac: no CUDA, so no Madrona
    cfg["vision"]["spectator"] = True
    cfg["vision"]["spectator_res"] = (1280, 960)

    env = DeepRacerEnv(num_envs=8, env_cfg=cfg)
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    env._post_physics(torch.arange(env.num_envs, device=env.device))

    out = REPO_ROOT / "runs" / "figures" / f"topdown_{track}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(env.render_spectator(), dtype=np.uint8)).save(out)
    print("open:", out)


if __name__ == "__main__":
    main()
