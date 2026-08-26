"""GPU verification of per-env rasterizer camera-mount DR (``randomize_mount``).

With spawn randomization off and all physics DR neutralized, every env holds
the SAME pose, so the per-env rasterizer cameras can differ only by their mount
transform (pitch + position jitter, applied once per run from
``_init_buffers``). A nonzero cross-env frame difference therefore proves
``RasterizerObsRenderer.randomize_mount`` rewrites each camera's attach offset
per env; a control run with the jitter off should give ~identical frames.

Run on a GPU machine::

    python scripts/verify_rasterizer_mount.py            # mount DR ON  -> frames differ
    python scripts/verify_rasterizer_mount.py --off      # control      -> frames ~identical
"""

from __future__ import annotations

import argparse

import torch

from deepracer_genesis._gs import ensure_init
from deepracer_genesis.configs.cfgs import get_env_cfg
from deepracer_genesis.envs import DeepRacerEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", action="store_true", help="control: mount jitter disabled")
    ap.add_argument("--num_envs", type=int, default=4)
    args = ap.parse_args()

    n = args.num_envs
    env_cfg = get_env_cfg(vision=True, track="reinvent_base", randomize=True)
    env_cfg["vision"]["vision_renderer"] = "rasterizer"
    env_cfg["spawn"]["random_start"] = False       # identical pose across envs
    env_cfg["spawn"]["random_direction"] = False
    env_cfg["spawn"]["spawn_lateral_noise"] = 0.0  # spawn noise applies even with
    env_cfg["spawn"]["spawn_yaw_noise"] = 0.0      # random_start off — zero it too
    # randomize=True is needed so _init_buffers calls randomize_mount, but it
    # also arms physics DR — neutralize every other knob so ONLY the mount varies
    env_cfg["rand"].update({
        "friction_range": (1.0, 1.0),
        "mass_shift_kg": 0.0,
        "com_shift_m": 0.0,
        "steer_kp_scale": (1.0, 1.0),
        "wheel_kv_scale": (1.0, 1.0),
        "armature_range": (0.0, 0.0),
        "track_width_scale": (1.0, 1.0),
        # wide jitter so the per-env viewpoint difference is unmistakable
        "camera_pitch_jitter_deg": 0.0 if args.off else 5.0,
        "camera_pos_jitter_m": 0.0 if args.off else 0.02,
    })
    ensure_init(env_cfg["sim"]["backend"])
    env = DeepRacerEnv(num_envs=n, env_cfg=env_cfg)   # mounts jittered once here

    env.reset_idx(torch.arange(n, device=env.device))  # same spawn for all envs
    env._post_physics()                                # render through the mounts

    img = env.image_buf                            # (N, 3, H, W) in [0, 1]
    means = [[round(c) for c in (img[i].mean(dim=(1, 2)) * 255).tolist()] for i in range(n)]
    pair = [(img[i] - img[j]).abs().mean().item()
            for i in range(n) for j in range(i + 1, n)]
    print(f"mount DR: {'OFF (control)' if args.off else 'ON'}")
    print(f"per-env mean RGB: {means}")
    print(f"cross-env pair diffs: {[round(p, 4) for p in pair]}")

    if args.off:
        ok = max(pair) < 1e-3
        verdict = "frames ~identical (no mount jitter, as expected)"
    else:
        ok = max(pair) > 0.01
        verdict = "per-env viewpoints differ (mount jitter active)"
    print(f"[{'PASS' if ok else 'FAIL'}] {verdict}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
