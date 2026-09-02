"""Photometric jitter standing in for a real camera's sensor and optics.

Exposure, response and white balance drift slowly, so they are drawn once per
frame stack; sensor noise is independent and is drawn per frame.
"""

from __future__ import annotations

import numpy as np

GAIN = (0.75, 1.30)        # auto-exposure, darker and brighter than nominal
GAMMA = (0.80, 1.25)       # the sensor's response curve
CONTRAST = (0.85, 1.15)
WHITE_BALANCE = 0.07       # per-channel gain around neutral, +/-
NOISE = (0.0, 0.03)        # per-frame sensor noise, in units of full scale

CameraState = tuple[float, float, float, np.ndarray, float]


def sample_camera(rng: np.random.Generator) -> CameraState:
    """Draw one camera state, to be shared by every frame of a stack.

    Args:
        rng: Random stream to draw from.

    Returns:
        A ``(gain, gamma, contrast, white_balance, noise)`` tuple.
    """
    return (float(rng.uniform(*GAIN)),
            float(rng.uniform(*GAMMA)),
            float(rng.uniform(*CONTRAST)),
            rng.uniform(1 - WHITE_BALANCE, 1 + WHITE_BALANCE, 3).astype(np.float32),
            float(rng.uniform(*NOISE)))


def apply(frame: np.ndarray, camera: CameraState,
          rng: np.random.Generator) -> np.ndarray:
    """Apply a camera state to one frame, leaving the caller's array untouched.

    Args:
        frame: An ``(H, W, 3)`` float32 frame in ``[0, 1]``.
        camera: The stack's camera state from :func:`sample_camera`.
        rng: Random stream for the per-frame sensor noise.

    Returns:
        A new ``(H, W, 3)`` float32 frame in ``[0, 1]``.
    """
    gain, gamma, contrast, white_balance, noise = camera
    out = frame * (gain * white_balance)
    np.clip(out, 0.0, 1.0, out=out)
    out **= gamma
    out = (out - 0.5) * contrast + 0.5
    if noise:
        out += rng.normal(0.0, noise, out.shape).astype(np.float32)
    np.clip(out, 0.0, 1.0, out=out)
    return out
