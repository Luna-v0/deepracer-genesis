"""Process-wide RNG seeding, applied by ``run()`` so ``spec.seed`` actually holds."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed python, numpy, and torch (CPU + all CUDA devices) in one call.

    Args:
        seed: The seed applied to every RNG.
        deterministic: Also request deterministic torch algorithms
            (``warn_only=True``: kernels without a deterministic variant warn
            instead of raising) — for nondeterminism investigations (P5).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
