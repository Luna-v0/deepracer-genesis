"""Expose the state-vector (feature) DeepRacer environment with no camera."""

from __future__ import annotations

from .base_env import DeepRacerEnv


class VectorDeepRacerEnv(DeepRacerEnv):
    """Feature-only DeepRacer environment that emits the ``state`` group alone.

    A thin no-vision specialization of :class:`~.base_env.DeepRacerEnv`.
    """

