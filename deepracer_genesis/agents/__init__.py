"""Scripted (non-learning) agents that drive the environment."""

from .scripted import CenterlineFollower, NoisyExpert, PrivilegedAgent

__all__ = ["CenterlineFollower", "NoisyExpert", "PrivilegedAgent"]
