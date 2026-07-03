"""Record a Nyx-rendered demo video: trained policy driving, onboard +
top-down views (path traced, original colors).

  python scripts/nyx_demo.py --checkpoint logs/smoke_state/model_59.pt
"""

import argparse
import os
import sys

import torch

import genesis as gs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="logs/smoke_state/model_59.pt",
                        help="state-policy checkpoint (drives while Nyx renders)")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--res", default="640x480")
    parser.add_argument("--spp", type=int, default=24)
    parser.add_argument("--mode", default="FastPathTracer",
                        choices=["Forward", "FastPathTracer", "RefPathTracer"])
    parser.add_argument("--out", default="logs/nyx_demo")
    args = parser.parse_args()

    gs.init(backend=gs.cuda, logging_level="warning")

    from rsl_rl.runners import OnPolicyRunner

    from deepracer_genesis.configs.cfgs import get_env_cfg, get_train_cfg
    from deepracer_genesis.envs import DeepRacerEnv

    w, h = args.res.lower().split("x")
    env_cfg = get_env_cfg(vision=True, topdown=True)
    env_cfg["vision_renderer"] = "nyx"
    env_cfg["camera_res"] = (int(w), int(h))
    env_cfg["nyx_spp"] = args.spp
    env_cfg["nyx_mode"] = args.mode
    env_cfg["random_start"] = True
    env = DeepRacerEnv(num_envs=args.num_envs, env_cfg=env_cfg)

    # the state policy drives (it reads the "state" obs group, which the env
    # always provides); Nyx renders the camera groups purely for the video
    runner = OnPolicyRunner(env, get_train_cfg(vision=False), None, device=str(gs.device))
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=str(gs.device))

    os.makedirs(args.out, exist_ok=True)
    onboard, topdown = [], []
    obs = env.get_observations()
    with torch.no_grad():
        for t in range(args.steps):
            obs, _, _, _ = env.step(policy(obs))
            onboard.append((env.image_buf[0].permute(1, 2, 0) * 255).byte().cpu().numpy())
            topdown.append(env.render_topdown()[0].cpu().numpy())
            if t % 100 == 0:
                print(f"{t}/{args.steps} frames", flush=True)

    import imageio.v2 as imageio
    imageio.mimsave(f"{args.out}/nyx_onboard.mp4", onboard, fps=50)
    imageio.mimsave(f"{args.out}/nyx_topdown.mp4", topdown, fps=50)
    print(f"saved {args.out}/nyx_onboard.mp4 and nyx_topdown.mp4 "
          f"({args.steps} frames @ {args.res}, {args.mode} spp={args.spp})")


if __name__ == "__main__":
    main()
