"""Pure-torch replay of the observation pipeline, one explicit stage at a time.

The env's post-render pipeline is pure torch end to end, applied in this exact
order (``envs/renderers.py::_CameraRenderer.render`` then
``envs/vision_env.py``): raw render -> world-colour GEMM -> pixel noise ->
area-downscale to ``policy_res`` -> ``apply_image_aug`` -> frame latency ->
frame stack. A DR-on env exposes no truly raw buffer (world colour and pixel
noise land inside ``render()``), so the editor builds its live scene
DR-STRIPPED and replays every stage offline from the raw grab — each stage
becomes an explicit tensor, exact by construction (same functions, same
order), and the whole image tier runs with no Genesis at all.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from ...randomization.image_aug import apply_image_aug
from ...randomization.latency import FrameLatency
from ...randomization.visual import add_pixel_noise, sample_world_color
from .rng import seeded

# stage names in application order; "raw" is the DR-stripped render
STAGES: tuple[str, ...] = ("raw", "world_color", "pixel_noise", "policy_res",
                           "image_aug", "latency", "stack")


def apply_world_color(img: torch.Tensor, strength: float, *,
                      color: Optional[tuple] = None) -> torch.Tensor:
    """Apply the per-env world-colour remap exactly as the renderer does.

    Mirrors ``_CameraRenderer.render``: the GEMM runs channels-last at the
    frame's own resolution, then clamps. Sampling draws from the (caller-
    seeded) global RNG like ``resample_appearance``.

    Args:
        img: ``(N, 3, H, W)`` float frames in [0, 1].
        strength: ``world_color`` strength; <= 0 returns ``img`` unchanged.
        color: Optional pre-sampled ``(color_mat, color_bias)`` pair to reuse
            (e.g. to show the SAME episode palette twice); sampled per env
            when None.

    Returns:
        The remapped ``(N, 3, H, W)`` frames.
    """
    if strength <= 0:
        return img
    n, c, h, w = img.shape
    if color is None:
        color = sample_world_color(n, strength, img.device)
    color_mat, color_bias = color
    flat = img.permute(0, 2, 3, 1).reshape(n, h * w, c)
    flat = (flat @ color_mat.transpose(1, 2) + color_bias).clamp_(0.0, 1.0)
    return flat.reshape(n, h, w, c).permute(0, 3, 1, 2).contiguous()


def replay_stages(raw: torch.Tensor, dr: dict, *,
                  policy_res: Optional[tuple] = None, frame_stack: int = 1,
                  seed: Optional[int] = None,
                  color: Optional[tuple] = None) -> dict[str, torch.Tensor]:
    """Replay every pipeline stage on a raw batch; return all intermediates.

    For a single-instant grab the latency stage is an identity by definition
    (the k-steps-ago frame of a static pose IS the current frame) and the
    stack stage shows the fresh-episode contract (prime by repeating, oldest
    first) — both exactly what the env produces in those situations.

    Args:
        raw: ``(N, 3, H, W)`` float raw frames in [0, 1] (a DR-stripped
            render, or bank frames).
        dr: DR parameters in spec shape: optional keys ``world_color`` (float
            strength), ``pixel_noise`` (float), ``image_aug`` (the aug dict,
            whose ``latency_steps``/``frame_drop`` ride along like at
            runtime).
        policy_res: ``(W, H)`` policy resolution, or None to keep the render
            resolution (mirrors ``vision.policy_res``).
        frame_stack: Channel-stack depth (env default 4; 1 = no stack).
        seed: Seed for every random draw (world colour, aug); None leaves the
            global RNG alone (non-reproducible).
        color: Optional pre-sampled world-colour ``(mat, bias)`` to reuse.

    Returns:
        ``{stage: tensor}`` for every :data:`STAGES` entry — all
        ``(N, 3, h, w)`` except ``stack`` which is ``(N, 3*k, h, w)``.
    """
    def _run() -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {"raw": raw}
        img = apply_world_color(raw, float(dr.get("world_color", 0.0) or 0.0),
                                color=color)
        out["world_color"] = img
        img = add_pixel_noise(img, float(dr.get("pixel_noise", 0.0) or 0.0))
        out["pixel_noise"] = img
        if policy_res is not None and tuple(policy_res) != (img.shape[3], img.shape[2]):
            pw, ph = policy_res
            img = F.interpolate(img, size=(ph, pw), mode="area")
        out["policy_res"] = img
        aug = dict(dr.get("image_aug", {}) or {})
        img = apply_image_aug(img, aug) if aug else img
        out["image_aug"] = img
        out["latency"] = img            # static instant: delayed frame == frame
        k = max(1, int(frame_stack))
        out["stack"] = img.repeat(1, k, 1, 1) if k > 1 else img
        return out

    if seed is None:
        return _run()
    with seeded(seed, raw.device):
        return _run()


def replay_temporal(frames: Sequence[torch.Tensor], dr: dict, *,
                    seed: Optional[int] = None) -> list[torch.Tensor]:
    """Run the stateful latency/frame-drop stage over a frame sequence.

    Args:
        frames: Time-ordered ``(N, 3, H, W)`` float frames (already at the
            stage BEFORE latency, i.e. post image-aug).
        dr: DR parameters; reads ``image_aug.latency_steps`` and
            ``image_aug.frame_drop`` exactly like ``vision_env`` does.
        seed: Seed for the frame-drop coin flips; None = global RNG.

    Returns:
        The frames the policy would see at each step, same length as input.
    """
    aug = dict(dr.get("image_aug", {}) or {})
    lat = int(aug.get("latency_steps", 0) or 0)
    drop = float(aug.get("frame_drop", 0.0) or 0.0)
    if lat <= 0 and drop <= 0:
        return list(frames)
    fl = FrameLatency(frames[0].shape[0], lat, drop, frames[0].device)

    def _run() -> list[torch.Tensor]:
        return [fl.advance(f) for f in frames]

    if seed is None:
        return _run()
    with seeded(seed, frames[0].device):
        return _run()


def dr_from_spec(spec) -> dict:
    """Extract the editor's DR-parameter dict from an ExperimentSpec.

    Args:
        spec: A built ``ExperimentSpec``.

    Returns:
        ``{"image_aug", "world_color", "pixel_noise", "env_map",
        "camera_jitter", "physics", "action_dr"}`` in the shapes the replay
        and the live session consume (empty/zero entries included).
    """
    ad = spec.action_dr
    return {
        "image_aug": dict(spec.obs_dr.image_aug),
        "world_color": float(spec.obs_dr.appearance.get("world_color", 0.0)),
        "pixel_noise": float(spec.obs_dr.pixel_noise),
        "env_map": dict(spec.obs_dr.env_map),
        "camera_jitter": dict(spec.obs_dr.camera_jitter),
        "physics": dict(spec.obs_dr.physics),
        "action_dr": {"steer_noise": ad.steer_noise,
                      "speed_noise": ad.speed_noise,
                      "delay_steps": ad.delay_steps},
    }
