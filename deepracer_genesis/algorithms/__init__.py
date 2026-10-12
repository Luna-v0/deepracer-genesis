"""Algorithms: the built-in PPO runs via rsl-rl; custom classes plug in here.

A custom algorithm passed to ``Algo(cls=...)`` routes through the SAME rsl-rl
``OnPolicyRunner`` PPO uses — it just has to speak the :class:`RslAlgorithm`
interface (see :mod:`deepracer_genesis.algorithms.protocol`).
"""

from .protocol import ALGO_INTERFACE, RslAlgorithm, missing_algorithm_methods

__all__ = ["ALGO_INTERFACE", "RslAlgorithm", "missing_algorithm_methods"]
