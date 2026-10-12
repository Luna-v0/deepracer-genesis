"""Config-as-code experiment framework composing stages into frozen specs.

The declaration layer imports no torch/genesis; heavy imports load lazily via `run()`.
"""

from .authoring import Experiment
from .run import build, run
from .spec import (
    ActionDRSpec,
    AlgorithmSpec,
    EncoderSpec,
    EnvSpec,
    EvalConfig,
    ExperimentSpec,
    ObsDRSpec,
    PolicySpec,
    SpecError,
)
from .stages import (
    PPO,
    Algo,
    AsymmetricCameraPolicy,
    AsymmetricVectorPolicy,
    CameraEnvironment,
    DomainRandomizationActions,
    DomainRandomizationCamera,
    DomainRandomizationPhysics,
    DomainRandomizationTrackAppearance,
    Evaluation,
    FeatureEnvironment,
    FrozenCNNToFeatureVector,
    Pipeline,
    PPOLagrangian,
    RewardShaping,
    SafeRLCameraEnvironment,
    SafeRLFeatureEnvironment,
    Stage,
    VectorPolicy,
)
from .visualize import rollout_video, view_zoo

__all__ = [
    "PPO",
    "ActionDRSpec",
    "Algo",
    "AlgorithmSpec",
    "AsymmetricCameraPolicy",
    "AsymmetricVectorPolicy",
    "CameraEnvironment",
    "DomainRandomizationActions",
    "DomainRandomizationCamera",
    "DomainRandomizationPhysics",
    "DomainRandomizationTrackAppearance",
    "EncoderSpec",
    "EnvSpec",
    "EvalConfig",
    "Evaluation",
    "Experiment",
    "ExperimentSpec",
    "FeatureEnvironment",
    "FrozenCNNToFeatureVector",
    "ObsDRSpec",
    "PPOLagrangian",
    "Pipeline",
    "PolicySpec",
    "RewardShaping",
    "SafeRLCameraEnvironment",
    "SafeRLFeatureEnvironment",
    "SpecError",
    "Stage",
    "VectorPolicy",
    "build",
    "rollout_video",
    "run",
    "view_zoo",
]
