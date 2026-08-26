"""Deterministic sampling contexts for the editor ("show me sample #7 again").

Editor policy (decided 2026-08-25): the shared DR samplers keep drawing from
the global torch RNG; the editor wraps every sampling call in a forked, seeded
RNG bubble instead of threading ``generator=`` through training code. The fork
covers BOTH the CPU generator and the frame's CUDA device — one image-aug draw
(the shared blur sigma) is CPU-side even on CUDA runs, so forking only the
device generator would leak nondeterminism.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


@contextmanager
def seeded(seed: int, device=None):
    """Fork the global RNG state and seed it for one deterministic block.

    Args:
        seed: Seed applied inside the fork (``torch.manual_seed`` seeds the
            CPU generator and every CUDA generator).
        device: Device whose CUDA generator must be forked alongside the CPU
            one; a CPU device (or None) forks the CPU generator only.

    Yields:
        None. All torch sampling inside the block is deterministic in
        ``seed`` and leaves the caller's RNG state untouched.
    """
    device = torch.device(device) if device is not None else torch.device("cpu")
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield


def stamp(seed: int, sample: int = 0) -> str:
    """Filename-safe provenance tag for an artifact's RNG coordinates.

    Args:
        seed: The seed the artifact was generated under.
        sample: Sample index within the seed (e.g. grid column).

    Returns:
        A short tag like ``"s0-i7"`` embedded in artifact filenames so any
        frame can be regenerated exactly.
    """
    return f"s{seed}-i{sample}"
