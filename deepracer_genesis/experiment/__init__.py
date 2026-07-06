"""Config-as-code experiment framework (plan section 0-2).

Authoring is Python: experiments are functions or `Experiment` classes that
compose stages with `>>` into a frozen `ExperimentSpec`. This package's
declaration layer imports no torch/genesis; the heavy imports live in
`builder`/`trainer`, loaded lazily by `run()`.
"""

from .spec import (
    ActionDRSpec,
    AlgorithmSpec,
    EncoderSpec,
    EnvSpec,
    ExperimentSpec,
    ObsDRSpec,
    PolicySpec,
    SpecError,
)
from .stages import (
    Algo,
    PPO,
    AsymmetricCameraPolicy,
    AsymmetricVectorPolicy,
    CameraEnvironment,
    DomainRandomizationActions,
    DomainRandomizationCamera,
    DomainRandomizationTrackAppearance,
    DomainRandomizationPhysics,
    FeatureEnvironment,
    FrozenCNNToFeatureVector,
    Pipeline,
    PPOLagrangian,
    SafeRLCameraEnvironment,
    SafeRLFeatureEnvironment,
    Stage,
    VectorPolicy,
)
from .registry import REGISTRY, Experiment, experiment
from .run import build, run

__all__ = [
    "ActionDRSpec", "AlgorithmSpec", "EncoderSpec", "EnvSpec", "ExperimentSpec",
    "ObsDRSpec", "PolicySpec", "SpecError",
    "Stage", "Pipeline",
    "CameraEnvironment", "FeatureEnvironment",
    "SafeRLCameraEnvironment", "SafeRLFeatureEnvironment",
    "DomainRandomizationCamera", "DomainRandomizationPhysics",
    "DomainRandomizationTrackAppearance",
    "DomainRandomizationActions",
    "FrozenCNNToFeatureVector",
    "AsymmetricCameraPolicy", "VectorPolicy", "AsymmetricVectorPolicy",
    "PPO", "PPOLagrangian", "Algo",
    "REGISTRY", "experiment", "Experiment",
    "build", "run",
]
