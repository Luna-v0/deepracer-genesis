"""Camera-feed validation harness (plan section i).

Training is fully headless; this script proves the vision pipeline works by
saving paired images per env — the onboard camera feed and a top-down view
above the track — plus short videos, and running automated sanity checks:

  1. non-degenerate frames (pixel std > threshold; catches broken EGL/Madrona)
  2. frames change between steps (catches a frozen camera mount)
  3. frames differ across envs (catches the renderer returning env 0 for all)
  4. cross-view consistency: the car's sim-state position projected through
     the top-down camera matches where the car appears in the top-down image

  python -m deepracer_genesis.validation.camera_check [--num_envs 4] [--steps 120]
  python -m deepracer_genesis.validation.camera_check --checkpoint logs/deepracer/model_300.pt
"""

import argparse
import math
import os

import numpy as np
import torch
from PIL import Image

import genesis as gs


def project_topdown(env, pos_xy):
    """Project world (x, y) to top-down camera pixel coords (u, v).

    Each env's top-down camera looks straight down from (cx, cy, h) with
    up=+y, so u grows with world +x, v grows with world -y (origin top-left).
    """
    w, h = env.cfg["camera_res"]
    c = env.top_cam_center            # (N, 2)
    height = env.top_cam_height       # (N,)
    fov = math.radians(60)
    f = (h / 2) / math.tan(fov / 2)   # focal length in pixels (fov is vertical)
    u = (pos_xy[:, 0] - c[:, 0]) / height * f + w / 2
    v = -(pos_xy[:, 1] - c[:, 1]) / height * f + h / 2
    return torch.stack([u, v], dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--snapshot_every", type=int, default=40)
    parser.add_argument("--out", default="logs/validation")
    parser.add_argument("--checkpoint", default=None, help="run a trained policy instead of scripted actions")
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--tracks", default="reinvent_base",
                        help="comma-separated track names; >1 builds a heterogeneous scene")
    parser.add_argument("--res", default="1280x960",
                        help="spectator (high-res bird's-eye) resolution WxH; the onboard "
                             "and per-env topdown cameras stay at the DeepRacer-native 160x120")
    args = parser.parse_args()

    gs.init(backend=gs.cuda, logging_level="warning")

    from deepracer_genesis.configs.cfgs import get_env_cfg, get_train_cfg
    from deepracer_genesis.envs import DeepRacerEnv

    os.makedirs(args.out, exist_ok=True)
    tracks = args.tracks.split(",")
    env_cfg = get_env_cfg(vision=True, randomize=args.randomize, topdown=True,
                          track=tracks if len(tracks) > 1 else tracks[0])
    env_cfg["random_start"] = True
    if len(tracks) == 1:
        # all tracks overlap in world coords, so the all-envs spectator view
        # is only meaningful for a homogeneous scene
        env_cfg["spectator"] = True
        w, h = args.res.lower().split("x")
        env_cfg["spectator_res"] = (int(w), int(h))
    env = DeepRacerEnv(num_envs=args.num_envs, env_cfg=env_cfg)

    policy = None
    if args.checkpoint:
        from rsl_rl.runners import OnPolicyRunner
        runner = OnPolicyRunner(env, get_train_cfg(vision=True), None, device=str(gs.device))
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=str(gs.device))

    N = env.num_envs
    prev_frame = None
    frame_diffs, results = [], {}
    onboard_frames, topdown_frames = [], []

    obs = env.get_observations()
    for t in range(args.steps):
        if policy is not None:
            with torch.no_grad():
                actions = policy(obs)
        else:
            # scripted: accelerate, then weave left/right
            steer = 0.6 * math.sin(2 * math.pi * t / 60)
            actions = torch.tensor([[steer, -0.2]], device=env.device).repeat(N, 1)
        obs, _, _, _ = env.step(actions)

        onboard = (env.image_buf.permute(0, 2, 3, 1) * 255).byte().cpu().numpy()
        topdown = env.render_topdown().cpu().numpy()
        onboard_frames.append(onboard[0])
        topdown_frames.append(topdown[0])

        cur = env.image_buf.clone()
        if prev_frame is not None:
            frame_diffs.append((cur - prev_frame).abs().mean().item())
        prev_frame = cur

        if t % args.snapshot_every == 0:
            for i in range(N):
                Image.fromarray(onboard[i]).save(f"{args.out}/env{i}_step{t:04d}_onboard.png")
                Image.fromarray(topdown[i]).save(f"{args.out}/env{i}_step{t:04d}_topdown.png")
            if env.spec_cam is not None:
                Image.fromarray(env.render_spectator()).save(
                    f"{args.out}/spectator_step{t:04d}.png")

    # ---- automated checks ----
    img = env.image_buf  # (N, 3, H, W) in [0, 1]
    per_env_std = img.flatten(1).std(dim=1)
    results["nondegenerate (std>0.02 all envs)"] = bool((per_env_std > 0.02).all())

    results["frames change between steps (mean diff>1e-4)"] = (
        len(frame_diffs) > 0 and float(np.mean(frame_diffs)) > 1e-4)

    pair_diffs = [
        (img[i] - img[j]).abs().mean().item()
        for i in range(N) for j in range(i + 1, N)
    ]
    results["frames differ across envs (max pair diff>0.005)"] = max(pair_diffs) > 0.005

    # cross-view: with the cars held static on-track, project the sim car
    # position into the top-down image and compare against the pixel centroid
    # of (car present) - (car parked far away). Static bracketing avoids
    # transient mismatches from cars resetting on the final step.
    saved_qpos = env.car.get_qpos().clone()
    park = torch.zeros(N, 13, device=env.device)
    park[:, 0] = -50.0
    park[:, 3] = 1.0
    env.car.set_qpos(park)
    env.scene.step()
    top_empty = env.render_topdown().float().cpu()
    env.car.set_qpos(saved_qpos)
    env.scene.step()
    top_with = env.render_topdown().float().cpu()
    car_px = project_topdown(env, env.car.get_pos()[:, :2]).cpu().numpy()
    err_px = []
    for i in range(N):
        diff = (top_with[i] - top_empty[i]).abs().sum(dim=-1)  # (H, W)
        if diff.max() < 30.0:
            err_px.append(float("inf"))
            continue
        ys, xs = torch.nonzero(diff > 30.0, as_tuple=True)
        centroid = np.array([xs.float().mean().item(), ys.float().mean().item()])
        err_px.append(float(np.linalg.norm(centroid - car_px[i])))
    # inf = car not visible from above; legitimate under bridges/start gates
    # (e.g. reInvent2019), so require accuracy on visible cars + majority visible
    visible = [e for e in err_px if e != float("inf")]
    results["cross-view position consistency (visible cars <8px, >=50% visible)"] = (
        len(visible) >= N / 2 and all(e < 8.0 for e in visible))

    # ---- videos ----
    import imageio.v2 as imageio
    imageio.mimsave(f"{args.out}/onboard_env0.mp4", onboard_frames, fps=25)
    imageio.mimsave(f"{args.out}/topdown_env0.mp4", topdown_frames, fps=25)

    print(f"\n=== camera validation ({'policy' if policy else 'scripted'} run) ===")
    print(f"per-env pixel std: {[round(s, 3) for s in per_env_std.tolist()]}")
    print(f"mean frame-to-frame diff: {np.mean(frame_diffs):.5f}")
    print(f"max cross-env pair diff: {max(pair_diffs):.5f}")
    print(f"cross-view centroid error px: {[round(e, 1) for e in err_px]}")
    all_pass = True
    for name, ok in results.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        all_pass &= ok
    print(f"snapshots + videos saved to {args.out}/")
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
