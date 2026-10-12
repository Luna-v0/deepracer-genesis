"""Visual domain-randomization *definitions* (Part L).

The single home for the appearance/observation DR math — world-color YIQ remap,
per-episode camera-mount jitter, and pixel noise — while the renderer remains
the *application* site (it imports and calls these). Keeping the sampling here
means "what visual knobs exist" lives in one place, not scattered across the
renderer subclasses.
"""

from __future__ import annotations

import math

import torch

# RGB <-> YIQ (NTSC luma/chroma): hue rotates in the IQ plane, sat/val scale on
# the diagonal. Shared by the world-color remap here and the ImageAug transform.
RGB2YIQ = torch.tensor([[0.299, 0.587, 0.114],
                        [0.596, -0.274, -0.322],
                        [0.211, -0.523, 0.312]])
YIQ2RGB = torch.tensor([[1.0, 0.956, 0.621],
                        [1.0, -0.272, -0.647],
                        [1.0, -1.106, 1.703]])


def uniform(lo, hi, shape, device) -> torch.Tensor:
    """Draw a uniform sample on ``[lo, hi)`` of ``shape`` on ``device``."""
    return lo + (hi - lo) * torch.rand(shape, device=device)


def sample_world_color(n: int, strength: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw a per-env, episode-static world-color remap.

    Composes a hue rotation, saturation/value scaling, a small random channel
    mix, and a bias — all scaled by ``strength`` — in YIQ space, mapped back to
    RGB.

    Args:
        n: Number of environments to sample for.
        strength: World-color DR strength ``s`` (0 disables; typical 0.6).
        device: Torch device for the returned tensors.

    Returns:
        ``(color_mat, color_bias)`` of shapes ``(n, 3, 3)`` and ``(n, 1, 3)``,
        applied as ``img @ color_mat.T + color_bias``.
    """
    s = strength
    dev = device
    theta = (torch.rand(n, device=dev) * 2 - 1) * math.pi * s
    cos, sin = torch.cos(theta), torch.sin(theta)
    rot = torch.zeros(n, 3, 3, device=dev)
    rot[:, 0, 0] = 1.0
    rot[:, 1, 1] = cos; rot[:, 1, 2] = -sin
    rot[:, 2, 1] = sin; rot[:, 2, 2] = cos
    sat = 1.0 + (torch.rand(n, device=dev) * 2 - 1) * 0.6 * s
    val = 1.0 + (torch.rand(n, device=dev) * 2 - 1) * 0.35 * s
    scale = torch.zeros(n, 3, 3, device=dev)
    scale[:, 0, 0] = val
    scale[:, 1, 1] = sat; scale[:, 2, 2] = sat
    m = YIQ2RGB.to(dev) @ scale @ rot @ RGB2YIQ.to(dev)
    mix = torch.randn(n, 3, 3, device=dev) * 0.08 * s
    color_mat = m + mix
    color_bias = (torch.rand(n, 1, 3, device=dev) * 2 - 1) * 0.12 * s
    return color_mat, color_bias


def sample_env_map(n: int, tint_range: tuple[float, float] = (0.35, 0.75),
                   mult_range: tuple[float, float] = (0.5, 2.0), *,
                   device, generator: torch.Generator | None = None
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample per-env Nyx environment-map (sky) DR: RGB tint + exposure (Part P.1).

    Draws a per-env uniform-radiance sky — a colour ``tint`` and an exposure
    ``multiplier`` — the values a texture-less ``EnvironmentMapAsset`` needs. The
    returned tensors are device-agnostic and Nyx-free (the NyxRenderer is the
    application site that converts them to ``nps.float3``); this keeps the range
    definition testable without a GPU/Nyx path-tracer.

    Args:
        n: Number of environments to sample for.
        tint_range: ``(lo, hi)`` for each RGB tint channel.
        mult_range: ``(lo, hi)`` for the exposure multiplier.
        device: Torch device for the returned tensors.
        generator: Optional ``torch.Generator`` for reproducible draws.

    Returns:
        ``(tint, multiplier)`` of shapes ``(n, 3)`` and ``(n,)``.

    Note:
        Nyx bakes env maps at ``scene.build()``, so these are per-ENV-FIXED
        (per run), NOT per-episode — sample ONCE at build with ``n`` draws; do
        not wire this into the per-episode reset path.
    """
    tlo, thi = tint_range
    mlo, mhi = mult_range
    tint = tlo + (thi - tlo) * torch.rand(n, 3, device=device, generator=generator)
    mult = mlo + (mhi - mlo) * torch.rand(n, device=device, generator=generator)
    return tint, mult


def add_pixel_noise(img: torch.Tensor, scale: float) -> torch.Tensor:
    """Add gaussian pixel noise (scale > 0) and clamp back to ``[0, 1]``."""
    if scale <= 0:
        return img
    return (img + torch.randn_like(img) * scale).clamp(0, 1)


def sample_mount_transforms(base_T: torch.Tensor, jitter_deg: float,
                            jitter_pos: float, n: int, device) -> torch.Tensor:
    """Jitter the camera-mount transform per env.

    Applies a uniform pitch rotation (``+/- jitter_deg``) and position offset
    (``+/- jitter_pos``) to the base mount transform.

    Args:
        base_T: The ``(4, 4)`` base mount transform to perturb.
        jitter_deg: Max pitch jitter in degrees.
        jitter_pos: Max position jitter in metres (applied per axis).
        n: Number of environments to sample for.
        device: Torch device for the returned tensor.

    Returns:
        An ``(n, 4, 4)`` batch of jittered mount transforms.
    """
    base = torch.as_tensor(base_T, dtype=torch.float32, device=device)
    p = torch.deg2rad(uniform(-jitter_deg, jitter_deg, (n,), device))
    rx = torch.zeros(n, 4, 4, device=device)
    rx[:, 0, 0] = 1.0
    rx[:, 3, 3] = 1.0
    rx[:, 1, 1] = torch.cos(p)
    rx[:, 1, 2] = -torch.sin(p)
    rx[:, 2, 1] = torch.sin(p)
    rx[:, 2, 2] = torch.cos(p)
    T = base.expand(n, 4, 4).clone() @ rx
    T[:, :3, 3] += uniform(-jitter_pos, jitter_pos, (n, 3), device)
    return T
