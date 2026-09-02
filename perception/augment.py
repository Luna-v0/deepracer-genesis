"""Photometric jitter standing in for a real camera's sensor and optics.

The dataset is rendered by a clean synthetic camera: fixed exposure, no
white-balance drift, no sensor noise. A DeepRacer's camera has all three, and
they are the axes on which a CNN trained on clean renders fails on the car.

The camera's own state -- exposure, response curve, white balance -- drifts
slowly next to a 4-frame stack (80 ms at 50 Hz), so it is drawn once per stack
and shared by every frame in it. Sensor noise is independent between frames, so
it is drawn per frame. Drawing both the same way would let the network average
the noise away across the stack and learn nothing from it.
"""

import numpy as np

GAIN = (0.75, 1.30)        # auto-exposure, darker and brighter than nominal
GAMMA = (0.80, 1.25)       # the sensor's response curve
CONTRAST = (0.85, 1.15)
WHITE_BALANCE = 0.07       # per-channel gain around neutral, +/-
NOISE = (0.0, 0.03)        # per-frame sensor noise, in units of full scale


def sample_camera(rng):
    """Draw one camera state, to be shared by every frame of a stack."""
    return (rng.uniform(*GAIN),
            rng.uniform(*GAMMA),
            rng.uniform(*CONTRAST),
            rng.uniform(1 - WHITE_BALANCE, 1 + WHITE_BALANCE, 3).astype(np.float32),
            rng.uniform(*NOISE))


def apply(frame, camera, rng):
    """Apply a camera state to one (H, W, 3) float32 frame within [0, 1]."""
    gain, gamma, contrast, white_balance, noise = camera
    frame = frame * (gain * white_balance)
    np.clip(frame, 0.0, 1.0, out=frame)
    frame **= gamma
    frame = (frame - 0.5) * contrast + 0.5
    if noise:
        frame += rng.normal(0.0, noise, frame.shape).astype(np.float32)
    np.clip(frame, 0.0, 1.0, out=frame)
    return frame
