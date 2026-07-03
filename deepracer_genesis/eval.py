"""Roll out a trained checkpoint headless and record onboard + top-down videos.

  python -m deepracer_genesis.eval --checkpoint logs/deepracer/model_300.pt
"""

import argparse
import os
import pickle

import torch

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--out", default="logs/eval")
    parser.add_argument("--res", default=None,
                        help="render resolution WxH for the recorded videos (e.g. 1280x960); "
                             "the policy still sees its training resolution")
    args = parser.parse_args()

    gs.init(backend=gs.cuda, logging_level="warning")

    from rsl_rl.runners import OnPolicyRunner

    from deepracer_genesis.configs.cfgs import get_env_cfg, get_train_cfg
    from deepracer_genesis.envs import DeepRacerEnv

    cfgs_path = os.path.join(os.path.dirname(args.checkpoint), "cfgs.pkl")
    if os.path.exists(cfgs_path):
        with open(cfgs_path, "rb") as f:
            saved = pickle.load(f)
        env_cfg, train_cfg = saved["env_cfg"], saved["train_cfg"]
    else:
        env_cfg, train_cfg = get_env_cfg(vision=True), get_train_cfg(vision=True)

    env_cfg["topdown_camera"] = env_cfg["vision"]
    env_cfg["random_start"] = False
    if args.res:
        w, h = args.res.lower().split("x")
        # render high-res for the recording; the policy keeps its training res
        env_cfg["policy_res"] = env_cfg["camera_res"]
        env_cfg["camera_res"] = (int(w), int(h))
    env = DeepRacerEnv(num_envs=args.num_envs, env_cfg=env_cfg)

    runner = OnPolicyRunner(env, train_cfg, None, device=str(gs.device))
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=str(gs.device))

    os.makedirs(args.out, exist_ok=True)
    onboard_frames, topdown_frames = [], []
    total_rew = torch.zeros(env.num_envs, device=env.device)
    laps = 0

    obs = env.get_observations()
    with torch.no_grad():
        for t in range(args.steps):
            actions = policy(obs)
            obs, rew, dones, _ = env.step(actions)
            total_rew += rew
            if env.vision:
                onboard_frames.append(
                    (env.image_buf[0].permute(1, 2, 0) * 255).byte().cpu().numpy())
                topdown_frames.append(env.render_topdown()[0].cpu().numpy())

    print(f"mean reward over {args.steps} steps: {total_rew.mean().item():.2f}")
    print(f"final progress (m): {[round(p, 2) for p in env.progress_m.tolist()]}")

    if onboard_frames:
        import imageio.v2 as imageio
        imageio.mimsave(f"{args.out}/onboard.mp4", onboard_frames, fps=50)
        imageio.mimsave(f"{args.out}/topdown.mp4", topdown_frames, fps=50)
        print(f"videos saved to {args.out}/")


if __name__ == "__main__":
    main()
