# DeepRacer-Genesis throughput results

Hardware: NVIDIA GeForce RTX 4060 Ti, Linux-7.0.10-arch1-1-x86_64-with-glibc2.43
Versions: genesis-world 1.2.1, gs-madrona 0.0.7.post2, rsl-rl-lib 5.4.1, torch 2.12.1
Method: median of N timed runs after warmup (JIT + first render excluded); aggregate steps/s = env steps summed across parallel agents per wall-clock second.

| Configuration | Obs type | n_agents (parallel envs) | Max steps/s per agent | Aggregate steps/s (per-agent x n_agents) | GPU VRAM (GB) | Notes |
|---|---|---|---|---|---|---|
| Physics only | state | 64 | 311.5 | 19937 | 0.0 | no rendering, random actions |
| Physics only | state | 256 | 242.6 | 62107 | 0.0 | no rendering, random actions |
| Physics only | state | 1024 | 215.7 | 220909 | 0.0 | no rendering, random actions |
| Physics only | state | 2048 | 194.5 | 398335 | 0.01 | no rendering, random actions |
| Physics only | state | 4096 | 164.3 | 672963 | 0.02 | no rendering, random actions |
| **Physics only** | state | **8192** | **112.3** | **920185** | 0.03 | no rendering, random actions |
| Vision 160x120 | RGB camera | 32 | 187.3 | 5995 | 0.03 | BatchRenderer, random actions |
| Vision 160x120 | RGB camera | 64 | 172.1 | 11015 | 0.04 | BatchRenderer, random actions |
| Vision 160x120 | RGB camera | 128 | 142.7 | 18262 | 0.08 | BatchRenderer, random actions |
| Vision 160x120 | RGB camera | 256 | 107.2 | 27450 | 0.15 | BatchRenderer, random actions |
| Vision 160x120 | RGB camera | 512 | 77.1 | 39458 | 0.28 | BatchRenderer, random actions |
| Vision + world-color DR | RGB camera | 32 | 187.4 | 5998 | 0.04 | per-episode color remap |
| Vision + world-color DR | RGB camera | 64 | 149.3 | 9553 | 0.07 | per-episode color remap |
| Vision + world-color DR | RGB camera | 128 | 102.9 | 13168 | 0.14 | per-episode color remap |
| Vision + world-color DR | RGB camera | 256 | 62.5 | 15999 | 0.26 | per-episode color remap |
| Vision + world-color DR | RGB camera | 512 | 37.8 | 19338 | 0.52 | per-episode color remap |
| Vision + DR | RGB camera | 256 | 48.3 | 12367 | 0.15 | physics + camera-mount randomization |
| Vision, full DR | RGB camera | 256 | 36.0 | 9215 | 0.26 | physics+jitter+world color |
| Vision + PPO update | RGB camera | 256 | 13.4 | 3429 | 2.94 | full training loop (render + CNN fwd/bwd) |
| Gazebo baseline (reference) | RGB camera | 1 | ~15-60 (RTF 1-2x) | ~15-60 | n/a | single-world, CPU, ROS-mediated |

Peak row (bold) is the headline number: max steps/s per agent x number of agents running in parallel.
