"""Visual verification tools: deterministic policy rollout videos and a DR preview.

Videos pair the bird's-eye spectator view with the onboard feed of env 0.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Optional

import torch

from .overrides import override
from ..agents import CenterlineFollower
from .run import build
from .spec import ActionDRSpec, ExperimentSpec, ObsDRSpec

_CONTROLLER = CenterlineFollower()
_SPECTATOR = {"spectator": True, "spectator_res": (1280, 960)}


def _rsl_actor(spec, ckpt_path: str, sim):
    """Load the trained rsl-rl policy as an evaluator-shaped ``actor(td)``.

    Args:
        spec: The (nominal-conditions) experiment spec.
        ckpt_path: Path to the rsl-rl checkpoint (model.pt).
        sim: The built sim the policy will drive.

    Returns:
        A callable that sets ``td["action"]`` from the inference policy.
    """
    from rsl_rl.runners import OnPolicyRunner

    from .rsl_backend import _RslActor, spec_to_train_cfg
    runner = OnPolicyRunner(sim, spec_to_train_cfg(spec), None, device=str(sim.device))
    runner.load(ckpt_path)
    return _RslActor(runner.get_inference_policy(device=str(sim.device)))


def rollout_video(target, *, root: str = "runs", ckpt: Optional[str] = None,
                  track: Optional[str] = None, steps: int = 500,
                  num_envs: Optional[int] = None, out: Optional[str] = None,
                  spectator_res: tuple = (1280, 960),
                  **overrides) -> str:
    """Record a deterministic rollout of a trained experiment under nominal conditions.

    Args:
        target: Any experiment handle (registered name / function / class /
            spec).
        root: Runs directory the run dir resolves under.
        ckpt: Checkpoint path; defaults to best.pt in the experiment's own
            run directory.
        track: Evaluate the SAME policy on a different track (policies are
            track-agnostic — observations are track-relative).
        steps: Number of control steps to record.
        num_envs: Override the number of parallel cars.
        out: Output directory; defaults to <run_dir>/videos.
        spectator_res: Spectator camera resolution (width, height).
        **overrides: Keyword overrides forwarded to build(target).

    Returns:
        Path of the spectator MP4 (bird's-eye, every parallel car in one
        frame); camera policies additionally get an onboard MP4 of env 0
        next to it.

    Raises:
        FileNotFoundError: If no checkpoint exists — train the experiment
            first.
    """
    import imageio.v2 as imageio

    from .builder import Builder
    from .evaluator import aggregate_episodes

    spec: ExperimentSpec = build(target, **overrides)
    run_dir = spec.run_dir(root)
    ckpt = ckpt or os.path.join(run_dir, "model.pt")
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
    sim = b.sim(extra_cfg={"vision": {"spectator": True,
                                      "spectator_res": tuple(spectator_res)}})
    actor = _rsl_actor(eval_spec, ckpt, sim)

    out_dir = out or os.path.join(run_dir, "videos")
    os.makedirs(out_dir, exist_ok=True)
    suffix = track or eval_spec.env.tracks[0]

    n = sim.num_envs
    sim.reset_idx(torch.arange(n, device=sim.device))
    sim._post_physics(torch.arange(n, device=sim.device))

    spectator_frames, onboard_frames = [], []
    streams = {k: [] for k in ("reward", "done", "progress_delta", "offtrack")}
    with torch.no_grad():
        for _ in range(steps):
            td = sim.get_observations().clone()
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

    Args:
        target: Experiment handle; must be a camera experiment with
            DomainRandomizationCamera (e.g. cam_baseline).
        steps: Number of control steps to record.
        num_envs: Number of parallel cars.
        out: Output directory for the two MP4s.
        **overrides: Keyword overrides forwarded to build(target).

    Returns:
        Path of the raw-vs-augmented onboard MP4.

    Raises:
        ValueError: If the target is not a camera experiment with image
            augmentation.
    """
    import imageio.v2 as imageio
    import numpy as np

    from .builder import Builder
    from ..randomization.image_aug import apply_image_aug

    spec = build(target, **overrides)
    if spec.env.modality != "camera" or not spec.obs_dr.image_aug:
        raise ValueError("dr_preview_video needs a camera experiment with "
                         "DomainRandomizationCamera (e.g. cam_baseline)")
    spec = override(spec, "env.num_envs", num_envs)

    b = Builder(spec)
    sim = b.sim(extra_cfg={"vision": dict(_SPECTATOR)})
    aug = dict(spec.obs_dr.image_aug)

    os.makedirs(out, exist_ok=True)
    paired, spectator = [], []
    with torch.no_grad():
        for _ in range(steps):
            sim.step(_CONTROLLER.act(sim))
            raw = sim.image_buf[0]                       # (3, H, W) in [0,1]
            seen = apply_image_aug(sim.image_buf, aug)[0]
            frame = torch.cat([raw, seen], dim=2)        # side by side
            paired.append((frame.permute(1, 2, 0) * 255).byte().cpu().numpy())
            spectator.append(sim.render_spectator())

    onboard_path = os.path.join(out, "onboard_raw_vs_augmented.mp4")
    imageio.mimsave(onboard_path, paired, fps=50)
    imageio.mimsave(os.path.join(out, "spectator_random_spawns.mp4"),
                    np.stack(spectator), fps=50)
    print(f"[visualize] DR preview written to {out}/")
    return onboard_path


def view_zoo(target, *, root: str = "runs", num_envs: Optional[int] = None,
             steps: int = 40, upscale: int = 1, save: Optional[str] = None,
             **overrides) -> list:
    """Capture each car's onboard camera view across the experiment's tracks.

    Builds the experiment's sim, drives a short seeded warmup with the
    centerline follower so every car sits somewhere sensible on its track,
    and grabs the newest RGB frame of each env's camera observation — at
    the exact resolution the policy (and the physical car) sees. With a
    zoo/multi-track spec this is the fastest way to eyeball what the cars
    see on every variant. (For the bird's-eye view of the tiled zoo itself,
    use ``tools.zoo.view_zoo`` — this is the onboard counterpart.)

    Args:
        target: Any experiment handle accepted by ``run.build`` (an
            Experiment class, a pipeline, a spec, ...). Must be a camera
            modality.
        root: Runs directory (only used to resolve the spec, not written).
        num_envs: Override the number of cars (default: one per track when
            the spec has several tracks, else the spec's own count capped
            at 8).
        steps: Seeded warmup control steps before the snapshot.
        upscale: Integer nearest-neighbour upscale of the returned images
            (1 = the camera's native resolution).
        save: Optional path for a track-labelled contact-sheet PNG of all
            views.
        **overrides: Keyword overrides forwarded to ``build(target)``.

    Returns:
        One ``PIL.Image`` per env, in env order (track order for the
        one-car-per-track default). PIL images render inline in Jupyter.

    Raises:
        SpecError: If the target is not a camera-modality experiment.
    """
    import numpy as np
    from PIL import Image

    from .builder import Builder
    from .spec import SpecError

    spec = build(target, **overrides)
    if spec.env.modality != "camera":
        raise SpecError("view_zoo needs a camera-modality experiment "
                        f"(got {spec.env.modality!r})")
    if num_envs is None:
        num_envs = (len(spec.env.tracks) if len(spec.env.tracks) > 1
                    else min(spec.env.num_envs, 8))
    spec = replace(spec, env=replace(spec.env, num_envs=num_envs))

    sim = Builder(spec).sim()
    with torch.inference_mode():
        for _ in range(steps):
            sim.step(_CONTROLLER.act(sim))
        cam = sim.get_observations()["camera"].float().cpu().numpy()

    rgb = np.clip(cam[:, -3:].transpose(0, 2, 3, 1) * 255, 0, 255).astype("uint8")
    images = [Image.fromarray(frame) for frame in rgb]
    if upscale > 1:
        images = [im.resize((im.width * upscale, im.height * upscale),
                            Image.NEAREST) for im in images]
    if save:
        _save_contact_sheet(images, _env_track_names(sim), save)
    return images


def _env_track_names(sim) -> list:
    """Per-env track names, or empty strings when the sim has no variants.

    Args:
        sim: The built DeepRacerEnv.

    Returns:
        One name per env.
    """
    try:
        idx = sim.track.variant_idx.tolist()
        return [sim.track.names[i] for i in idx]
    except AttributeError:
        return [""] * sim.num_envs


def _save_contact_sheet(images: list, labels: list, path: str) -> None:
    """Write a labelled grid PNG of the onboard views.

    Args:
        images: The per-env PIL images.
        labels: One caption per image (track names).
        path: Destination PNG path (parents are created).
    """
    import math

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = min(4, len(images))
    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.9 * rows),
                             squeeze=False)
    for ax in axes.flat[len(images):]:
        ax.axis("off")
    for image, label, ax in zip(images, labels, axes.flat):
        ax.imshow(image)
        ax.set_title(label, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] onboard contact sheet written to {path}")
