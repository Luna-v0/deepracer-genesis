"""Teleport-sweep camera dataset: place cars on a pose grid, render, record.

No policy or dynamics needed; one call = one track.
"""

from __future__ import annotations

import itertools
import json
import os
from collections.abc import Sequence

import numpy as np
import torch


def collect_camera_dataset(
    track: str = "reinvent_base",
    out: str = "datasets/camera",
    lateral_fracs: Sequence[float] = (-0.6, -0.3, 0.0, 0.3, 0.6),
    yaw_offsets: Sequence[float] = (-0.3, 0.0, 0.3),
    waypoint_stride: int = 1,
    resolution: tuple[int, int] = (160, 120),
    image_aug: dict | None = None,
    num_envs: int = 64,
    shard_size: int = 512,
    render: str = "madrona",
) -> str:
    """Sweep `track` on a pose grid and write camera frames + state to `out/`.

    Args:
        track: Track name; one call = one track (Genesis builds one scene
            per process).
        out: Output dataset directory.
        lateral_fracs: Fractions of the local half-width (|1.0| = road edge).
        yaw_offsets: Radians relative to the track tangent.
        waypoint_stride: Sample every k-th waypoint.
        resolution: Camera resolution (width, height).
        image_aug: Same dict as DomainRandomizationCamera produces (e.g.
            from a built spec's obs_dr.image_aug); applies the training-time
            augmentation to the stored frames.
        num_envs: Cars placed (and frames rendered) per batch.
        shard_size: Frames per .npz shard.
        render: Batch renderer, "madrona" or "nyx".

    Returns:
        The dataset directory `out`.
    """
    from ..configs.cfgs import get_env_cfg
    from ..envs import DeepRacerEnv
    from ..experiment.builder import _ensure_genesis

    _ensure_genesis()
    cfg = get_env_cfg(vision=True, track=track)
    cfg["vision"]["camera_res"] = tuple(resolution)
    if render == "nyx":
        cfg["vision"]["vision_renderer"] = "nyx"
    sim = DeepRacerEnv(num_envs=num_envs, env_cfg=cfg)
    trk = sim.track.tracks[0]
    aug = dict(image_aug) if image_aug else None

    # the full pose grid, chunked into batches of num_envs
    grid = list(itertools.product(
        range(0, trk.n_wps, waypoint_stride), lateral_fracs, yaw_offsets))
    os.makedirs(out, exist_ok=True)

    ids_all = torch.arange(num_envs, device=sim.device)
    images, states, poses = [], [], []
    shard, written = 0, 0

    def flush() -> None:
        nonlocal shard, images, states, poses, written
        if not images:
            return
        np.savez_compressed(
            os.path.join(out, f"shard_{shard:04d}.npz"),
            image=np.concatenate(images),
            state=np.concatenate(states),
            pose=np.concatenate(poses))
        written += sum(len(i) for i in images)
        shard += 1
        images, states, poses = [], [], []

    with torch.no_grad():
        for start in range(0, len(grid), num_envs):
            chunk = grid[start:start + num_envs]
            n = len(chunk)
            wp = torch.tensor([c[0] for c in chunk], device=sim.device)
            lat = torch.tensor([c[1] for c in chunk], device=sim.device,
                               dtype=torch.float32)
            dyaw = torch.tensor([c[2] for c in chunk], device=sim.device,
                                dtype=torch.float32)

            center = trk.center[wp]
            normal = trk.normal[wp]
            yaw = trk.track_yaw[wp] + dyaw
            pos_xy = center + normal * (lat * trk.half_width[wp]).unsqueeze(1)

            qpos = torch.zeros(num_envs, 13, device=sim.device)
            qpos[:, 3] = 1.0
            qpos[:n, 0:2] = pos_xy
            qpos[:n, 2] = sim.cfg["spawn"]["spawn_height"]
            qpos[:n, 3] = torch.cos(yaw / 2)
            qpos[:n, 6] = torch.sin(yaw / 2)
            sim.car.set_qpos(qpos)
            sim._post_physics(ids_all)           # refresh state + camera

            img = sim.image_buf[:n]
            if aug is not None:
                from ..randomization.image_aug import apply_image_aug
                img = apply_image_aug(img, aug)
            images.append((img.permute(0, 2, 3, 1) * 255)
                          .byte().cpu().numpy())
            states.append(sim.state_buf[:n].cpu().numpy())
            poses.append(torch.stack(
                [pos_xy[:, 0], pos_xy[:, 1], yaw,
                 sim.progress_m[:n]], dim=1).cpu().numpy())

            if sum(len(i) for i in images) >= shard_size:
                flush()
    flush()

    meta = {"track": track, "resolution": list(resolution),
            "lateral_fracs": list(lateral_fracs),
            "yaw_offsets": list(yaw_offsets),
            "waypoint_stride": waypoint_stride,
            "image_aug": image_aug or {}, "frames": written,
            "shards": shard, "render": render}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[collect] {written} frames -> {out} ({shard} shard(s))")
    return out


