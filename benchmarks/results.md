# DeepRacer-Genesis throughput results

Hardware: NVIDIA GeForce RTX 4060 Ti, Linux-7.0.10-arch1-1-x86_64-with-glibc2.43
Versions: genesis-world 1.2.0, gs-madrona 0.0.7.post2, rsl-rl-lib 5.4.1, torch 2.12.1
Method: median of N timed runs after warmup (JIT + first render excluded); aggregate steps/s = env steps summed across parallel agents per wall-clock second.

| Configuration | Obs type | n_agents (parallel envs) | Max steps/s per agent | Aggregate steps/s (per-agent x n_agents) | GPU VRAM (GB) | Notes |
|---|---|---|---|---|---|---|
| Physics only | state | 1 | 575.8 | 576 | 0.0 | no rendering, random actions |
| Physics only | state | 64 | 336.0 | 21506 | 0.0 | no rendering, random actions |
| Physics only | state | 256 | 247.7 | 63410 | 0.0 | no rendering, random actions |
| Physics only | state | 1024 | 222.7 | 228004 | 0.0 | no rendering, random actions |
| **Physics only** | state | **4096** | **168.5** | **690175** | 0.02 | no rendering, random actions |
| Vision 160x120 | RGB camera | 1 | 281.0 | 281 | 0.01 | BatchRenderer, random actions |
| Vision 160x120 | RGB camera | 64 | 171.3 | 10963 | 0.06 | BatchRenderer, random actions |
| Vision 160x120 | RGB camera | 256 | 107.6 | 27548 | 0.21 | BatchRenderer, random actions |
| Vision 160x120 | RGB camera | 512 | 78.2 | 40049 | 0.4 | BatchRenderer, random actions |
| Vision 160x120 | RGB camera | 1024 | 51.6 | 52814 | 0.8 | BatchRenderer, random actions |
| Vision + DR | RGB camera | 256 | 99.5 | 25473 | 0.21 | physics + camera-mount randomization |
| Vision + DR + heterogeneous | RGB camera | 256 | 92.2 | 23593 | 0.21 | 3 different track meshes across envs + DR |
| Vision + PPO update | RGB camera | 256 | 13.7 | 3495 | 2.94 | full training loop (render + CNN fwd/bwd) |
| Gazebo baseline (reference) | RGB camera | 1 | ~15-60 (RTF 1-2x) | ~15-60 | n/a | single-world, CPU, ROS-mediated |

Peak row (bold) is the headline number: max steps/s per agent x number of agents running in parallel.
