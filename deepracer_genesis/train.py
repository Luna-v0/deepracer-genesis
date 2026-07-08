"""Train a DeepRacer policy with rsl-rl PPO on Genesis.

  python -m deepracer_genesis.train -B 1024 --max_iterations 300
  python -m deepracer_genesis.train -B 256 --vision --randomize --max_iterations 500
"""

import argparse
import os

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", "--num_envs", type=int, default=1024)
    parser.add_argument("--max_iterations", type=int, default=300)
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--nyx", action="store_true",
                        help="render vision obs with the Nyx renderer (true colors, "
                             "photorealistic; slower than Madrona)")
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--track", default="reinvent_base")
    parser.add_argument("--exp_name", default="deepracer")
    parser.add_argument("--resume", default=None, help="checkpoint path to resume from")
    args = parser.parse_args()

    gs.init(backend=gs.cuda, logging_level="warning")

    from deepracer_genesis.algorithms.rsl_rl import build_runner
    from deepracer_genesis.configs.cfgs import get_env_cfg
    from deepracer_genesis.envs import DeepRacerEnv

    if args.nyx:
        args.vision = True
    env_cfg = get_env_cfg(vision=args.vision, track=args.track, randomize=args.randomize)
    if args.nyx:
        env_cfg["vision_renderer"] = "nyx"
    log_dir = os.path.join("logs", args.exp_name)
    env = DeepRacerEnv(num_envs=args.num_envs, env_cfg=env_cfg)
    runner = build_runner(env, vision=args.vision, log_dir=log_dir,
                          device=str(gs.device), num_envs=args.num_envs)
    if args.resume:
        runner.load(args.resume)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
