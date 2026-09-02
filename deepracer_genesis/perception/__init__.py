"""Camera-based perception: a frozen CNN standing in for privileged state.

The CNN predicts the channels of ``PerceptionFeatures`` a real DeepRacer has no
sensor for; the rest stay computed onboard.
"""

from deepracer_genesis.perception.features import (
    CHANNEL_NAMES,
    SIGMA,
    CNNPerceptionFeatures,
    NoisyPerceptionFeatures,
)
from deepracer_genesis.perception.model import PerceptionCNN

__all__ = [
    "CHANNEL_NAMES",
    "SIGMA",
    "CNNPerceptionFeatures",
    "NoisyPerceptionFeatures",
    "PerceptionCNN",
]
