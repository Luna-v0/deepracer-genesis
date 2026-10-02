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
from deepracer_genesis.perception.model import (
    DEFAULT_ARCH,
    PerceptionCNN,
    load_checkpoint,
    save_checkpoint,
)

__all__ = [
    "CHANNEL_NAMES",
    "SIGMA",
    "CNNPerceptionFeatures",
    "NoisyPerceptionFeatures",
    "PerceptionCNN",
    "DEFAULT_ARCH",
    "save_checkpoint",
    "load_checkpoint",
]
