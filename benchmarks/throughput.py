"""Throughput benchmark (plan section j).

Single-config mode (fresh process per config for clean JIT/VRAM):
  python benchmarks/throughput.py --n_envs 256 --mode vision

Sweep mode: spawns subprocesses, aggregates, writes benchmarks/results.csv
and benchmarks/results.md with the final table (max steps/s per agent x
n_agents, peak row bolded):
  python benchmarks/throughput.py --sweep
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

HETERO_TRACKS = ["reinvent_base", "reInvent2019_track", "2022_reinvent_champ"]

MODES = {
    "physics": dict(vision=False, policy=False),
    "vision": dict(vision=True, policy=False),
    "vision_ppo": dict(vision=True, policy=True),
    "vision_dr": dict(vision=True, policy=False, randomize=True),
    "vision_hetero": dict(vision=True, policy=False, randomize=True, tracks=HETERO_TRACKS),
    "vision_raster": dict(vision=True, policy=False, raster=True),
}


def run_single(n_envs, mode, steps, warmup, repeats):
    import torch
    import genesis as gs

    gs.init(backend=gs.cuda, logging_level="warning", performance_mode=True)

    sys.path.insert(0, ROOT)
    from deepracer_genesis.configs.cfgs import get_env_cfg, get_train_cfg
    from deepracer_genesis.envs import DeepRacerEnv

    m = MODES[mode]
    track = m.get("tracks", "reinvent_base")
    env_cfg = get_env_cfg(vision=m["vision"], randomize=m.get("randomize", False), track=track)
    if m.get("raster"):
        env_cfg["vision_renderer"] = "raster"
    env = DeepRacerEnv(num_envs=n_envs, env_cfg=env_cfg)

    runner = None
    if m["policy"]:
        from rsl_rl.runners import OnPolicyRunner
        runner = OnPolicyRunner(env, get_train_cfg(vision=m["vision"]), None, device=str(gs.device))

    def random_actions():
        return torch.rand(n_envs, 2, device=env.device) * 2 - 1

    results = []
    if runner is None:
        for _ in range(warmup):
            env.step(random_actions())
        torch.cuda.synchronize()
        for _ in range(repeats):
            t0 = time.perf_counter()
            for _ in range(steps):
                env.step(random_actions())
            torch.cuda.synchronize()
            results.append(n_envs * steps / (time.perf_counter() - t0))
    else:
        # full training loop: rollout + PPO update (render + CNN fwd/bwd)
        runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)  # warmup + JIT
        steps_per_iter = runner.cfg["num_steps_per_env"] * n_envs
        for _ in range(repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            iters = max(1, steps // (runner.cfg["num_steps_per_env"]))
            runner.learn(num_learning_iterations=iters, init_at_random_ep_len=True)
            torch.cuda.synchronize()
            results.append(iters * steps_per_iter / (time.perf_counter() - t0))

    results.sort()
    median = results[len(results) // 2]
    vram_gb = torch.cuda.max_memory_allocated() / 1e9
    print("RESULT_JSON:" + json.dumps({
        "mode": mode, "n_envs": n_envs,
        "agg_sps": median, "per_agent_sps": median / n_envs,
        "vram_gb": round(vram_gb, 2), "runs": [round(r) for r in results],
    }))


def sweep(args):
    import torch

    configs = []
    for n in [1, 64, 256, 1024, 4096]:
        configs.append(("physics", n))
    for n in [1, 64, 256, 512, 1024]:
        configs.append(("vision", n))
    configs.append(("vision_dr", 256))
    configs.append(("vision_hetero", 256))
    configs.append(("vision_ppo", 256))

    rows = []
    for mode, n_envs in configs:
        cmd = [sys.executable, os.path.abspath(__file__), "--n_envs", str(n_envs), "--mode", mode,
               "--steps", str(args.steps), "--warmup", str(args.warmup), "--repeats", str(args.repeats)]
        print(f"--- {mode} n_envs={n_envs} ---", flush=True)
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=ROOT)
            line = [l for l in out.stdout.splitlines() if l.startswith("RESULT_JSON:")]
            if not line:
                print(f"  FAILED\n{out.stdout[-1500:]}\n{out.stderr[-1500:]}")
                continue
            r = json.loads(line[0][len("RESULT_JSON:"):])
            rows.append(r)
            print(f"  {r['agg_sps']:.0f} steps/s aggregate, {r['per_agent_sps']:.1f}/agent, {r['vram_gb']} GB")
        except subprocess.TimeoutExpired:
            print("  TIMEOUT")

    write_results(rows)


def write_results(rows):
    import csv
    import platform

    import torch

    csv_path = os.path.join(HERE, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "n_envs", "per_agent_sps", "agg_sps", "vram_gb", "runs"])
        w.writeheader()
        w.writerows(rows)

    import importlib.metadata as md
    gpu = torch.cuda.get_device_name(0)
    versions = {p: md.version(p) for p in ["genesis-world", "gs-madrona", "rsl-rl-lib", "torch"]}

    peak = max(rows, key=lambda r: r["agg_sps"]) if rows else None
    labels = {
        "physics": ("Physics only", "state", "no rendering, random actions"),
        "vision": ("Vision 160x120", "RGB camera", "BatchRenderer, random actions"),
        "vision_ppo": ("Vision + PPO update", "RGB camera", "full training loop (render + CNN fwd/bwd)"),
        "vision_dr": ("Vision + DR", "RGB camera", "physics + camera-mount randomization"),
        "vision_hetero": ("Vision + DR + heterogeneous", "RGB camera",
                          "3 different track meshes across envs + DR"),
    }
    lines = [
        "# DeepRacer-Genesis throughput results",
        "",
        f"Hardware: {gpu}, {platform.platform()}",
        f"Versions: " + ", ".join(f"{k} {v}" for k, v in versions.items()),
        "Method: median of N timed runs after warmup (JIT + first render excluded); "
        "aggregate steps/s = env steps summed across parallel agents per wall-clock second.",
        "",
        "| Configuration | Obs type | n_agents (parallel envs) | Max steps/s per agent | "
        "Aggregate steps/s (per-agent x n_agents) | GPU VRAM (GB) | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        name, obs, note = labels[r["mode"]]
        bold = "**" if peak and r is peak else ""
        lines.append(
            f"| {bold}{name}{bold} | {obs} | {bold}{r['n_envs']}{bold} | "
            f"{bold}{r['per_agent_sps']:.1f}{bold} | {bold}{r['agg_sps']:.0f}{bold} | "
            f"{r['vram_gb']} | {note} |")
    lines += [
        "| Gazebo baseline (reference) | RGB camera | 1 | ~15-60 (RTF 1-2x) | ~15-60 | n/a | "
        "single-world, CPU, ROS-mediated |",
        "",
        "Peak row (bold) is the headline number: max steps/s per agent x number of agents "
        "running in parallel.",
    ]
    md_path = os.path.join(HERE, "results.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {csv_path} and {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--n_envs", type=int, default=256)
    parser.add_argument("--mode", choices=list(MODES), default="physics")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.sweep:
        sweep(args)
    else:
        run_single(args.n_envs, args.mode, args.steps, args.warmup, args.repeats)
