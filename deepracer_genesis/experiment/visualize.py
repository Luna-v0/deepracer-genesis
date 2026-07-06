"""Visual verification tools: policy rollout videos and a DR preview.

    from deepracer_genesis.experiment.visualize import rollout_video
    rollout_video("feature_baseline")                       # its own track
    rollout_video("feature_baseline", track="reInvent2019_track")

Videos are the spectator view (bird's-eye, every parallel car in one frame,
true colors via the rasterizer) plus, for camera policies, the onboard feed
of env 0. Rollouts are deterministic (mean action) and metrics are printed,
so "is the car actually driving well on track X?" gets both an answer and
footage.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Optional

import torch

from .ablation import override
from .run import build
from .spec import ActionDRSpec, ExperimentSpec, ObsDRSpec

_SPECTATOR = {"spectator": True, "spectator_res": (1280, 960)}


def _load_actor(builder, ckpt_path: str):
    """Builder-shaped actor with checkpointed weights."""
    actor = builder.actor()
    payload = torch.load(ckpt_path, map_location=builder.sim().device,
                         weights_only=False)
    actor.load_state_dict(payload["actor"])
    return actor


def _controller(sim) -> torch.Tensor:
    """Privileged P-controller on the centerline (scripted driving for
    previews — no trained policy needed)."""
    lat = sim.lateral / sim.half_width.clamp(min=0.1)
    steer = (-(1.1 * lat + 0.9 * torch.sin(sim.heading_err))).clamp(-1, 1)
    speed = torch.full_like(steer, -0.3)          # ~1.5 m/s
    return torch.stack([steer, speed], dim=1)


def rollout_video(target, *, root: str = "runs", ckpt: Optional[str] = None,
                  track: Optional[str] = None, steps: int = 500,
                  num_envs: Optional[int] = None, out: Optional[str] = None,
                  **overrides) -> str:
    """Record a deterministic rollout of a trained experiment.

    `target` is any experiment handle (registered name / function / class /
    spec). The checkpoint resolves from the experiment's own run directory
    unless `ckpt` is given; `track` evaluates the SAME policy on a different
    track (policies are track-agnostic — observations are track-relative).
    Returns the spectator video path; prints eval metrics.
    """
    import imageio.v2 as imageio

    from .builder import Builder
    from .evaluator import aggregate_episodes

    spec: ExperimentSpec = build(target, **overrides)
    run_dir = spec.run_dir(root)
    ckpt = ckpt or os.path.join(run_dir, "best.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"no checkpoint at {ckpt} — train the experiment first "
            f"(run({target!r}))")

    # evaluate under NOMINAL conditions: no image aug, no action noise/delay,
    # no physics randomization — the footage shows the policy, not the DR
    eval_spec = replace(spec, obs_dr=ObsDRSpec(), action_dr=ActionDRSpec())
    if track:
        eval_spec = override(eval_spec, "env.tracks", (track,))
    if num_envs:
        eval_spec = override(eval_spec, "env.num_envs", num_envs)
    eval_spec.validate()

    b = Builder(eval_spec)
    sim = b.sim(extra_cfg=dict(_SPECTATOR))
    actor = _load_actor(b, ckpt)
    obs_transform = None
    if eval_spec.encoder.kind == "frozen_cnn":
        encoder, _ = b.encoder_module()
        key = eval_spec.encoder.out_key
        obs_transform = lambda td: td.set(key, encoder(td["camera"]))  # noqa: E731

    out_dir = out or os.path.join(run_dir, "videos")
    os.makedirs(out_dir, exist_ok=True)
    suffix = track or eval_spec.env.tracks[0]

    from torchrl.envs.utils import ExplorationType, set_exploration_type

    n = sim.num_envs
    sim.reset_idx(torch.arange(n, device=sim.device))
    sim._post_physics(torch.arange(n, device=sim.device))

    spectator_frames, onboard_frames = [], []
    streams = {k: [] for k in ("reward", "done", "progress_delta", "offtrack")}
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for _ in range(steps):
            td = sim.get_observations().clone()
            if obs_transform is not None:
                td = obs_transform(td)
            td = actor(td)
            _, rew, dones, _ = sim.step(td["action"])
            info = sim.step_info
            streams["reward"].append(rew.clone())
            streams["done"].append(dones.clone())
            streams["progress_delta"].append(info["progress_delta"])
            streams["offtrack"].append(info["offtrack"] | info["flipped"])
            spectator_frames.append(sim.render_spectator())
            if sim.vision:
                onboard_frames.append(
                    (sim.image_buf[0].permute(1, 2, 0) * 255).byte().cpu().numpy())

    metrics = aggregate_episodes(control_dt=sim.dt,
                                 track_length=sim.track.total_len_env,
                                 **{k: torch.stack(v) for k, v in streams.items()})
    print(f"[visualize] {suffix}: "
          + ", ".join(f"{k}={v:.3g}" for k, v in metrics.items()
                      if isinstance(v, (int, float))))

    spectator_path = os.path.join(out_dir, f"spectator_{suffix}.mp4")
    imageio.mimsave(spectator_path, spectator_frames, fps=50)
    if onboard_frames:
        imageio.mimsave(os.path.join(out_dir, f"onboard_{suffix}.mp4"),
                        onboard_frames, fps=50)
    return spectator_path


def dr_preview_video(target="cam_baseline", *, steps: int = 300,
                     num_envs: int = 8, out: str = "runs/dr_preview",
                     **overrides) -> str:
    """Show the domain randomization in action, no trained policy needed.

    A privileged P-controller drives; the output pairs the RAW onboard camera
    with the AUGMENTED frame the policy would see (side by side), plus the
    spectator view where per-episode random respawns are visible. Physics DR
    (per-env friction/mass/COM/gains) resamples at every reset underneath.
    """
    import imageio.v2 as imageio
    import numpy as np

    from .builder import Builder
    from .transforms import ImageAug

    spec = build(target, **overrides)
    if spec.env.modality != "camera" or not spec.obs_dr.image_aug:
        raise ValueError("dr_preview_video needs a camera experiment with "
                         "DomainRandomizationCamera (e.g. cam_baseline)")
    spec = override(spec, "env.num_envs", num_envs)

    b = Builder(spec)
    sim = b.sim(extra_cfg=dict(_SPECTATOR))
    aug = ImageAug(spec.obs_dr.image_aug)

    os.makedirs(out, exist_ok=True)
    paired, spectator = [], []
    with torch.no_grad():
        for _ in range(steps):
            sim.step(_controller(sim))
            raw = sim.image_buf[0]                       # (3, H, W) in [0,1]
            seen = aug._apply_transform(sim.image_buf)[0]
            frame = torch.cat([raw, seen], dim=2)        # side by side
            paired.append((frame.permute(1, 2, 0) * 255).byte().cpu().numpy())
            spectator.append(sim.render_spectator())

    onboard_path = os.path.join(out, "onboard_raw_vs_augmented.mp4")
    imageio.mimsave(onboard_path, paired, fps=50)
    imageio.mimsave(os.path.join(out, "spectator_random_spawns.mp4"),
                    np.stack(spectator), fps=50)
    print(f"[visualize] DR preview written to {out}/")
    return onboard_path
