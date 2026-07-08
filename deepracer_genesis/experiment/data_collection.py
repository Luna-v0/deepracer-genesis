"""Camera dataset collection: sweep the track and record what the car sees.

Teleport-based, no policy or dynamics needed: the batched sim places N cars
per step on a (waypoint x lateral offset x yaw offset) grid, renders, and
stores frames + the privileged state — uniform coverage of the visual space,
orders of magnitude faster than driving. Useful for representation
pretraining, offline encoders, or auditing what the camera actually sees.

    from deepracer_genesis.experiment.data_collection import collect_camera_dataset
    collect_camera_dataset(track="reinvent_base", out="datasets/reinvent",
                           lateral_fracs=(-0.6, 0.0, 0.6), yaw_offsets=(-0.3, 0, 0.3))

Output: shard_XXXX.npz files with `image` (B,H,W,3) uint8, `state` (B,28)
float32, `pose` (B,4) float32 [x, y, yaw, progress_m], plus meta.json.
One call = one track (Genesis builds one scene per process).
"""

from __future__ import annotations

import itertools
import json
import os
from typing import Optional, Sequence

import numpy as np
import torch


def collect_camera_dataset(
    track: str = "reinvent_base",
    out: str = "datasets/camera",
    lateral_fracs: Sequence[float] = (-0.6, -0.3, 0.0, 0.3, 0.6),
    yaw_offsets: Sequence[float] = (-0.3, 0.0, 0.3),
    waypoint_stride: int = 1,
    resolution: tuple[int, int] = (160, 120),
    image_aug: Optional[dict] = None,
    num_envs: int = 64,
    shard_size: int = 512,
    render: str = "madrona",
) -> str:
    """Sweep `track` and write camera frames + state to `out/`.

    lateral_fracs are fractions of the local half-width (|1.0| = road edge);
    yaw_offsets are radians relative to the track tangent; `image_aug` (same
    dict as DomainRandomizationCamera produces, e.g. from a built spec's
    obs_dr.image_aug) applies the training-time augmentation to the stored
    frames. Returns the dataset directory.
    """
    from ..configs.cfgs import get_env_cfg
    from ..envs import DeepRacerEnv
    from .builder import _ensure_genesis
    from .transforms import ImageAug

    _ensure_genesis()
    cfg = get_env_cfg(vision=True, track=track)
    cfg["camera_res"] = tuple(resolution)
    if render == "nyx":
        cfg["vision_renderer"] = "nyx"
    sim = DeepRacerEnv(num_envs=num_envs, env_cfg=cfg)
    trk = sim.track.tracks[0]
    aug = ImageAug(image_aug) if image_aug else None

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
            qpos[:n, 2] = sim.cfg["spawn_height"]
            qpos[:n, 3] = torch.cos(yaw / 2)
            qpos[:n, 6] = torch.sin(yaw / 2)
            sim.car.set_qpos(qpos)
            sim._post_physics(ids_all)           # refresh state + camera

            img = sim.image_buf[:n]
            if aug is not None:
                img = aug._apply_transform(img)
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


# ======================================================================
# Rollout collection: temporally-coherent frames from a noisy expert
# ======================================================================
def collect_rollout_dataset(
    target,
    *,
    out: str = "datasets/rollouts",
    steps: int = 2048,
    num_envs: Optional[int] = None,
    agent=None,
    shard_steps: int = 256,
    seed: int = 0,
    compress: bool = True,
) -> str:
    """Drive a noisy PRIVILEGED expert under the pipeline's DR and record
    temporally-contiguous (frame, feature-vector) sequences.

    `target` is any experiment handle — most usefully a `>>` chain whose env
    and DR stages define what gets collected (the policy stage is optional
    for collection and ignored):

        from deepracer_genesis.experiment import (CameraEnvironment,
            DomainRandomizationCamera, DomainRandomizationPhysics,
            DomainRandomizationTrackAppearance)
        from deepracer_genesis.experiment.data_collection import collect_rollout_dataset

        collect_rollout_dataset(
            CameraEnvironment(resolution=(160, 120), num_envs=16)
            >> DomainRandomizationTrackAppearance(strength=0.6)
            >> DomainRandomizationCamera(brightness=(0.7, 1.3), hue=0.05)
            >> DomainRandomizationPhysics(),
            out="datasets/reinvent_rollouts", steps=4096)

    `agent` is any PrivilegedAgent (see experiment/agents.py); the default
    NoisyExpert steers from privileged track state with Ornstein-Uhlenbeck
    noise on top — temporally-correlated wandering, so trajectories drift off
    the centerline and DO go off-track sometimes (those episodes end and
    respawn, exactly the data a frame-stacking CNN needs to see). Subclass
    PrivilegedAgent for custom behavior. Collection is deterministic in
    `seed` for a given agent.

    Output: parquet shards, rows sorted (env, t) — env-major and
    time-contiguous, so a k-frame stack is k consecutive rows of one env:

      rollout_XXXX.parquet columns:
        env int16, t int32, episode int32   temporal bookkeeping
        done bool                           True at the LAST step of an episode
        image binary                        PNG, HxWx3 — the policy camera
                                            view AFTER world-color + image-aug
        state list<float32>[28]             aligned privileged feature vector
        action list<float32>[2]             expert action actually applied
        pose  list<float32>[4]              [x, y, yaw, progress_m]
      meta.json (shapes, dt, DR config, seed).

    A k-stack window (env e, ending at t) is valid iff all k rows share the
    same `episode` value — no window ever crosses a respawn.
    """
    import io
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    from .agents import NoisyExpert
    from .builder import Builder
    from .run import build
    from .spec import SpecError
    from .stages import Pipeline, Stage, VectorPolicy
    from .transforms import ImageAug

    agent = agent or NoisyExpert()

    if isinstance(target, Stage):
        target = Pipeline((target,))
    if isinstance(target, Pipeline):
        try:
            spec = target.build()
        except SpecError:
            spec = (target >> VectorPolicy()).build()   # policy unused; validation only
    else:
        spec = build(target)
    if spec.env.modality != "camera":
        raise SpecError("rollout collection records the camera; use a camera env stage")
    if num_envs:
        from .ablation import override
        spec = override(spec, "env.num_envs", num_envs)

    torch.manual_seed(seed)
    b = Builder(spec)
    sim = b.sim()
    n = sim.num_envs
    dev = sim.device
    aug = ImageAug(spec.obs_dr.image_aug) if spec.obs_dr.image_aug else None

    os.makedirs(out, exist_ok=True)
    buf: dict[str, list] = {k: [] for k in ("image", "state", "action", "pose", "done")}
    shard_idx, t_base, frames_out = 0, 0, 0
    episode = np.zeros(n, dtype=np.int64)
    pool = ThreadPoolExecutor(max_workers=8)

    def _png(frame: np.ndarray) -> bytes:
        bio = io.BytesIO()
        Image.fromarray(frame).save(bio, "PNG",
                                    compress_level=6 if compress else 1)
        return bio.getvalue()

    def flush():
        nonlocal shard_idx, t_base, frames_out, episode
        if not buf["image"]:
            return
        T = len(buf["image"])
        img = np.stack(buf["image"], axis=1)          # (N, T, H, W, 3)
        st = np.stack(buf["state"], axis=1)
        ac = np.stack(buf["action"], axis=1)
        po = np.stack(buf["pose"], axis=1)
        dn = np.stack(buf["done"], axis=1)            # (N, T)
        # per-row episode ids: increment AFTER each done row
        ep = episode[:, None] + np.concatenate(
            [np.zeros((n, 1), dtype=np.int64), np.cumsum(dn[:, :-1], axis=1)], axis=1)
        episode = ep[:, -1] + dn[:, -1]
        pngs = list(pool.map(_png, img.reshape(-1, *img.shape[2:])))  # env-major
        table = pa.table({
            "env": pa.array(np.repeat(np.arange(n, dtype=np.int16), T)),
            "t": pa.array(np.tile(t_base + np.arange(T, dtype=np.int32), n)),
            "episode": pa.array(ep.reshape(-1).astype(np.int32)),
            "done": pa.array(dn.reshape(-1)),
            "image": pa.array(pngs, type=pa.binary()),
            "state": pa.array(list(st.reshape(n * T, -1))),
            "action": pa.array(list(ac.reshape(n * T, -1))),
            "pose": pa.array(list(po.reshape(n * T, -1))),
        })
        pq.write_table(table, os.path.join(out, f"rollout_{shard_idx:04d}.parquet"),
                       compression="zstd")
        frames_out += n * T
        t_base += T
        shard_idx += 1
        for v in buf.values():
            v.clear()

    sim.reset_idx(torch.arange(n, device=dev))
    sim._post_physics(torch.arange(n, device=dev))
    with torch.no_grad():
        for _ in range(steps):
            act = agent.act(sim)

            img = sim.obs_image_buf                                # post world-color
            if aug is not None:
                img = aug._apply_transform(img)
            state = sim.state_buf.clone()
            pose = torch.stack([sim.base_pos[:, 0], sim.base_pos[:, 1],
                                sim.yaw, sim.progress_m], dim=1)

            _, _, dones, _ = sim.step(act)
            reset_ids = dones.nonzero(as_tuple=False).flatten()
            if len(reset_ids):
                agent.reset(reset_ids)

            buf["image"].append((img.permute(0, 2, 3, 1) * 255).byte().cpu().numpy())
            buf["state"].append(state.cpu().numpy())
            buf["action"].append(act.cpu().numpy())
            buf["pose"].append(pose.cpu().numpy())
            buf["done"].append(dones.bool().cpu().numpy())
            if len(buf["image"]) >= shard_steps:
                flush()
    flush()

    pool.shutdown()
    meta = {"num_envs": n, "steps": steps, "shard_steps": shard_steps,
            "frames": frames_out,
            "resolution": list(spec.env.resolution), "control_dt": sim.dt,
            "agent": type(agent).__name__, "seed": seed,
            "layout": "rows sorted (env, t); k-stack valid iff same episode",
            "appearance": dict(spec.obs_dr.appearance),
            "image_aug": dict(spec.obs_dr.image_aug),
            "physics_dr": dict(spec.obs_dr.physics), "shards": shard_idx,
            "tracks": list(spec.env.tracks)}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[collect] {frames_out} frames in {shard_idx} parquet shard(s) -> {out}",
          flush=True)
    return out
