"""Training algorithms: the Algorithm protocol, its registry, and the
shipped implementations (PPO, PPO-Lagrangian). See protocol.py for the guide
to writing your own; rsl_rl.py holds the legacy rsl-rl-lib integration."""

from .lagrangian import PIDLagrangian, PPOLagrangian
from .ppo import PPO
from .protocol import ALGORITHMS, Algorithm, make_algorithm, register_algorithm

__all__ = ["ALGORITHMS", "Algorithm", "register_algorithm", "make_algorithm",
           "PPO", "PPOLagrangian", "PIDLagrangian"]
