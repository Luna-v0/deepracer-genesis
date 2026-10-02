"""Config-as-code experiment framework composing stages into frozen specs.

The declaration layer imports no torch/genesis; heavy imports load lazily via `run()`.
"""

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
    Algo,
    PPO,
    AsymmetricCameraPolicy,
    AsymmetricVectorPolicy,
    CameraEnvironment,
    DomainRandomizationActions,
    DomainRandomizationCamera,
    DomainRandomizationTrackAppearance,
    DomainRandomizationPhysics,
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
from .authoring import Experiment
from .run import build, run
from .visualize import rollout_video, view_zoo

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
    "PPO", "PPOLagrangian", "Algo", "RewardShaping",
    "Evaluation", "EvalConfig",
    "Experiment",
    "build", "run",
    "rollout_video", "view_zoo",
]
