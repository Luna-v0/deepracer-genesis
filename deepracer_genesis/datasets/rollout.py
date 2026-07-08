"""Rollout dataset: a scripted agent drives, frames + features are recorded.

See collect_rollout_dataset below; agents live in deepracer_genesis.agents.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch


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
    """Record temporally-contiguous (frame, feature-vector) rollout sequences.

    A noisy PRIVILEGED expert drives under the pipeline's DR while frames and
    aligned features are recorded. Typical use — a `>>` chain whose env and
    DR stages define what gets collected:

        from deepracer_genesis.experiment import (CameraEnvironment,
            DomainRandomizationCamera, DomainRandomizationPhysics,
            DomainRandomizationTrackAppearance)
        from deepracer_genesis.datasets import collect_rollout_dataset

        collect_rollout_dataset(
            CameraEnvironment(resolution=(160, 120), num_envs=16)
            >> DomainRandomizationTrackAppearance(strength=0.6)
            >> DomainRandomizationCamera(brightness=(0.7, 1.3), hue=0.05)
            >> DomainRandomizationPhysics(),
            out="datasets/reinvent_rollouts", steps=4096)

    Output layout: parquet shards, rows sorted (env, t) — env-major and
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

    Args:
        target: Any experiment handle — most usefully a `>>` chain (Stage or
            Pipeline) as above; the policy stage is optional for collection
            and ignored. Must build a camera env.
        out: Output dataset directory.
        steps: Control steps to record (per env; total frames = steps x
            num_envs).
        num_envs: Override the pipeline's parallel-env count.
        agent: Any PrivilegedAgent (see deepracer_genesis.agents); the
            default NoisyExpert steers from privileged track state with
            Ornstein-Uhlenbeck noise on top — temporally-correlated
            wandering, so trajectories drift off the centerline and DO go
            off-track sometimes (those episodes end and respawn, exactly the
            data a frame-stacking CNN needs to see). Subclass
            PrivilegedAgent for custom behavior.
        shard_steps: Steps buffered per parquet shard.
        seed: Collection is deterministic in `seed` for a given agent.
        compress: PNG compress_level 6 when True, 1 (faster, larger) when
            False.

    Returns:
        The dataset directory `out`.

    Raises:
        SpecError: If the target's env is not a camera env (rollout
            collection records the camera).
    """
    import io
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    from ..agents import NoisyExpert
    from ..experiment.builder import Builder
    from ..experiment.run import build
    from ..experiment.spec import SpecError
    from ..experiment.stages import Pipeline, Stage, VectorPolicy
    from ..experiment.transforms import ImageAug

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
        from ..experiment.ablation import override
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
    fs = sim.feature_set
    meta = {"num_envs": n, "steps": steps, "shard_steps": shard_steps,
            "frames": frames_out,
            "resolution": list(spec.env.resolution), "control_dt": sim.dt,
            "agent": type(agent).__name__, "seed": seed,
            "feature_set": spec.env.feature_set,
            "state_layout": type(fs).layout_for(lookahead_k=spec.env.lookahead_k,
                                                params=spec.env.feature_params),
            # rows state[:, slice] are the channels a deployed CNN must
            # predict from pixels — the supervision targets
            "cnn_target_slice": list(fs.cnn_target_slice) if fs.cnn_target_slice else None,
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
