"""Train a DeepRacer policy with rsl-rl PPO on Genesis.

  python -m deepracer_genesis.train -B 1024 --max_iterations 300
  python -m deepracer_genesis.train -B 256 --vision --randomize --max_iterations 500
"""

import argparse
import copy
import os
import pickle

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", "--num_envs", type=int, default=1024)
    parser.add_argument("--max_iterations", type=int, default=300)
    parser.add_argument("--vision", action="store_true")
    parser.add_argument("--cpu", action="store_true",
                        help="run simulation + training on CPU (visual-only obs; "
                             "per-env rasterizer cameras instead of Madrona). Slow.")
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--track", default="reinvent_base")
    parser.add_argument("--exp_name", default="deepracer")
    parser.add_argument("--resume", default=None, help="checkpoint path to resume from")
    args = parser.parse_args()

    if args.cpu:
        args.vision = True  # the CPU branch trains from pixels only
    gs.init(backend=gs.cpu if args.cpu else gs.cuda, logging_level="warning")

    from rsl_rl.runners import OnPolicyRunner

    from deepracer_genesis.configs.cfgs import get_env_cfg, get_train_cfg
    from deepracer_genesis.envs import DeepRacerEnv

    env_cfg = get_env_cfg(vision=args.vision, track=args.track, randomize=args.randomize)
    train_cfg = get_train_cfg(vision=args.vision, visual_only=args.cpu)

    log_dir = os.path.join("logs", args.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "cfgs.pkl"), "wb") as f:
        pickle.dump({"env_cfg": env_cfg, "train_cfg": copy.deepcopy(train_cfg),
                     "num_envs": args.num_envs}, f)

    env = DeepRacerEnv(num_envs=args.num_envs, env_cfg=env_cfg)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=str(gs.device))
    if args.resume:
        runner.load(args.resume)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
