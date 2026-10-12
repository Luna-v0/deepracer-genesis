"""rsl-rl feature-PPO training throughput at one num_envs (GPU-scaling probe).

Prints one PERF line: aggregate steps/s, per-env steps/s, and peak VRAM.
"""

import argparse
import time

import genesis as gs

from deepracer_genesis._gs import ensure_init
from deepracer_genesis.configs.cfgs import get_env_cfg, get_train_cfg
from deepracer_genesis.envs import DeepRacerEnv


def measure(num_envs: int, iters: int, warmup: int) -> tuple[float, float]:
    """Time the rsl-rl training loop for num_envs and return (agg_sps, peak_vram_gb).

    Args:
        num_envs: Parallel environments.
        iters: Timed learning iterations.
        warmup: Untimed warmup iterations (JIT + allocator warm).

    Returns:
        Aggregate steps/s and peak VRAM in GB.
    """
    import torch
    from rsl_rl.runners import OnPolicyRunner
    cfg = get_env_cfg(vision=False, track="reinvent_base", randomize=True, backend="gpu")
    env = DeepRacerEnv(num_envs=num_envs, env_cfg=cfg)
    train_cfg = get_train_cfg(vision=False)
    horizon = train_cfg["num_steps_per_env"]
    runner = OnPolicyRunner(env, train_cfg, None, device=str(gs.device))
    runner.learn(num_learning_iterations=warmup, init_at_random_ep_len=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    runner.learn(num_learning_iterations=iters, init_at_random_ep_len=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return num_envs * iters * horizon / dt, torch.cuda.max_memory_allocated() / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, required=True)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    args = ap.parse_args()
    ensure_init("gpu")
    sps, vram = measure(args.envs, args.iters, args.warmup)
    print(f"PERF num_envs={args.envs} agg_sps={sps:.0f} "
          f"per_env={sps / args.envs:.1f} peak_vram_gb={vram:.2f}")


if __name__ == "__main__":
    main()
