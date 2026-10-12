"""Public entry point and common base for the DeepRacer env.

``DeepRacerEnv`` dispatches by ``env_cfg["vision"]`` to the vision or vector env.
"""

from .base_env import DeepRacerEnv
from .vector_env import VectorDeepRacerEnv
from .vision_env import VisionDeepRacerEnv

__all__ = ["DeepRacerEnv", "VectorDeepRacerEnv", "VisionDeepRacerEnv"]
