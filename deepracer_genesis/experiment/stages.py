"""The `>>` builder DSL (plan section 1).

`>>` is `__rshift__`: each Stage folds itself into the spec via
`apply(spec) -> spec`; composition builds a Pipeline; `Pipeline.build()`
left-folds the stages over an empty spec, infers the algorithm, validates.
Build-time only — nothing here runs per-step.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

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

# ----------------------------------------------------------------------
# defaults shared by stages
DEFAULT_CNN = {
    "channels": (16, 32, 64),
    "kernels": (8, 4, 3),
    "strides": (4, 2, 1),
    "activation": "relu",
}
DEFAULT_MLP = {"hidden": (256, 128, 64), "activation": "elu"}
DEFAULT_PPO = {
    "clip": 0.2,
    "epochs": 5,
    "minibatches": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "lr": 3.0e-4,
    "entropy_coef": 0.01,
    "max_grad_norm": 1.0,
    "horizon": 24,           # rollout steps per env per PPO iteration
}
DEFAULT_PID = (0.05, 0.0005, 0.1)


# ----------------------------------------------------------------------
class Stage:
    """One slice of the spec. Subclasses implement apply(spec) -> spec."""

    KIND: str = "stage"

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        raise NotImplementedError

    def __rshift__(self, other):
        return Pipeline([self]) >> other


class Pipeline:
    def __init__(self, stages):
        self.stages = list(stages)

    def __rshift__(self, other):
        extra = other.stages if isinstance(other, Pipeline) else [other]
        return Pipeline(self.stages + extra)

    def _check_structure(self):
        if not self.stages:
            raise SpecError("empty pipeline")
        if self.stages[0].KIND != "environment":
            raise SpecError("the first stage must be an Environment; got %s"
                            % type(self.stages[0]).__name__)
        counts: dict[str, int] = {}
        for st in self.stages:
            counts[st.KIND] = counts.get(st.KIND, 0) + 1
        if counts.get("policy", 0) != 1:
            raise SpecError("pipeline must include exactly one Policy stage; got %d"
                            % counts.get("policy", 0))
        for kind, limit in (("environment", 1), ("encoder", 1),
                            ("action_dr", 1), ("algorithm", 1),
                            ("obs_dr_camera", 1), ("obs_dr_physics", 1)):
            if counts.get(kind, 0) > limit:
                raise SpecError("at most %d %s stage(s) allowed; got %d"
                                % (limit, kind, counts[kind]))

    def build(self, **overrides) -> ExperimentSpec:
        self._check_structure()
        spec = ExperimentSpec()
        for st in self.stages:
            spec = st.apply(spec)
        if overrides:
            spec = replace(spec, **overrides)
        spec = _infer_algorithm(spec)
        spec.validate()
        return spec


# ----------------------------------------------------------------------
# Environment stages (source; must be first)
@dataclass(frozen=True)
class FeatureEnvironment(Stage):
    """State-vector env: waypoint-relative features, no rendering."""

    features: tuple[str, ...] = ()
    lookahead_k: int = 10
    tracks: tuple[str, ...] = ("reinvent_base",)   # >1 => heterogeneous per-env
    num_envs: int = 512
    random_start: bool = True
    random_direction: bool = False     # coin-flip CW/CCW per episode

    KIND = "environment"

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        return replace(spec, env=EnvSpec(
            modality="feature", render="none",
            features=tuple(self.features), lookahead_k=self.lookahead_k,
            tracks=tuple(self.tracks), num_envs=self.num_envs,
            random_start=self.random_start,
            random_direction=self.random_direction,
        ))


@dataclass(frozen=True)
class CameraEnvironment(Stage):
    """Front-RGB-camera env; `tracks` with >1 entry trains heterogeneously
    (each parallel env simulates + renders its own track; Madrona only)."""

    render: str = "madrona"
    resolution: tuple[int, int] = (160, 120)
    fov: float = 90.0
    lookahead_k: int = 10
    tracks: tuple[str, ...] = ("reinvent_base",)
    num_envs: int = 128
    random_start: bool = True
    random_direction: bool = False     # coin-flip CW/CCW per episode

    KIND = "environment"

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        return replace(spec, env=EnvSpec(
            modality="camera", render=self.render,
            resolution=tuple(self.resolution), fov=self.fov,
            lookahead_k=self.lookahead_k, tracks=tuple(self.tracks),
            num_envs=self.num_envs, random_start=self.random_start,
            random_direction=self.random_direction,
        ))


@dataclass(frozen=True)
class SafeRLFeatureEnvironment(FeatureEnvironment):
    """Feature env that also emits a cost signal (=> PPO-Lagrangian inferred)."""
    cost: str = "offtrack"
    budget: float = 25.0

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        spec = super().apply(spec)
        return replace(spec, env=replace(
            spec.env, emits_cost=True, cost_fn=self.cost, cost_budget=self.budget))


@dataclass(frozen=True)
class SafeRLCameraEnvironment(CameraEnvironment):
    """Camera env that also emits a cost signal (=> PPO-Lagrangian inferred)."""
    cost: str = "offtrack"
    budget: float = 25.0

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        spec = super().apply(spec)
        return replace(spec, env=replace(
            spec.env, emits_cost=True, cost_fn=self.cost, cost_budget=self.budget))


# ----------------------------------------------------------------------
# Observation DR stages
@dataclass(frozen=True)
class DomainRandomizationTrackAppearance(Stage):
    """Scene-level color/texture DR: bake `variants` visual versions of the
    track (per-variant RGB tints; road surface optionally swapped for the
    shipped brick/carpet/concrete/grass materials; per-variant field color)
    and give each parallel env one of them for the whole run — every batch
    then spans the appearance distribution. Madrona + single track only."""

    variants: int = 8
    tint: tuple[float, float] = (0.6, 1.4)          # RGB multiplier range
    line_tint: tuple[float, float] = (0.9, 1.1)     # lane lines: milder
    swap_road_materials: bool = True
    randomize_field_color: bool = True
    field_tint: tuple[float, float] = (0.5, 1.5)
    seed: int = 0

    KIND = "obs_dr_appearance"

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        return replace(spec, obs_dr=replace(spec.obs_dr, appearance={
            "variants": self.variants, "tint": tuple(self.tint),
            "line_tint": tuple(self.line_tint),
            "swap_road_materials": self.swap_road_materials,
            "randomize_field_color": self.randomize_field_color,
            "field_tint": tuple(self.field_tint), "seed": self.seed,
        }))


@dataclass(frozen=True)
class DomainRandomizationCamera(Stage):
    brightness: Optional[tuple[float, float]] = None
    contrast: Optional[tuple[float, float]] = None
    saturation: Optional[tuple[float, float]] = None
    hue: float = 0.0
    blur: float = 0.0
    cutout: float = 0.0            # probability of a cutout patch per frame
    noise: float = 0.0             # additive gaussian sigma
    camera_jitter: bool | dict = False

    KIND = "obs_dr_camera"

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        aug = {}
        if self.brightness: aug["brightness"] = tuple(self.brightness)
        if self.contrast:   aug["contrast"] = tuple(self.contrast)
        if self.saturation: aug["saturation"] = tuple(self.saturation)
        if self.hue:        aug["hue"] = self.hue
        if self.blur:       aug["blur"] = self.blur
        if self.cutout:     aug["cutout"] = self.cutout
        if self.noise:      aug["noise"] = self.noise
        if self.camera_jitter is True:
            jitter = {"pitch_deg": 2.0, "pos_m": 0.005}
        elif isinstance(self.camera_jitter, dict):
            jitter = dict(self.camera_jitter)
        else:
            jitter = {}
        return replace(spec, obs_dr=replace(spec.obs_dr,
                                            image_aug=aug, camera_jitter=jitter))


@dataclass(frozen=True)
class DomainRandomizationPhysics(Stage):
    friction: tuple[float, float] = (0.6, 1.4)
    mass: float = 0.2              # +- kg per link
    com: float = 0.01              # +- m per link
    gains: tuple[float, float] = (0.8, 1.2)
    armature: tuple[float, float] = (0.0, 0.01)

    KIND = "obs_dr_physics"

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        physics = {
            "friction_range": tuple(self.friction),
            "mass_shift_kg": self.mass,
            "com_shift_m": self.com,
            "steer_kp_scale": tuple(self.gains),
            "wheel_kv_scale": tuple(self.gains),
            "armature_range": tuple(self.armature),
        }
        return replace(spec, obs_dr=replace(spec.obs_dr, physics=physics))


# ----------------------------------------------------------------------
# Encoder stage
@dataclass(frozen=True)
class FrozenCNNToFeatureVector(Stage):
    checkpoint: str = ""
    output_dim: int = 256
    layer: Optional[str] = None
    out_key: str = "encoded"

    KIND = "encoder"

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        return replace(spec, encoder=EncoderSpec(
            kind="frozen_cnn", checkpoint=self.checkpoint or None,
            output_dim=self.output_dim, layer=self.layer, out_key=self.out_key,
        ))


# ----------------------------------------------------------------------
# Policy stages (exactly one)
@dataclass(frozen=True)
class AsymmetricCameraPolicy(Stage):
    actor_keys: tuple[str, ...] = ("camera",)
    critic_keys: tuple[str, ...] = ("camera", "state")
    cnn: Optional[dict] = None
    mlp: Optional[dict] = None

    KIND = "policy"

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        return replace(spec, policy=PolicySpec(
            actor_keys=tuple(self.actor_keys), critic_keys=tuple(self.critic_keys),
            cnn=dict(self.cnn or DEFAULT_CNN), mlp=dict(self.mlp or DEFAULT_MLP),
        ))


@dataclass(frozen=True)
class VectorPolicy(Stage):
    keys: tuple[str, ...] = ("state",)
    mlp: Optional[dict] = None

    KIND = "policy"

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        return replace(spec, policy=PolicySpec(
            actor_keys=tuple(self.keys), critic_keys=tuple(self.keys),
            cnn=None, mlp=dict(self.mlp or DEFAULT_MLP),
        ))


@dataclass(frozen=True)
class AsymmetricVectorPolicy(Stage):
    actor_keys: tuple[str, ...] = ("state",)
    critic_keys: tuple[str, ...] = ("state",)
    mlp: Optional[dict] = None

    KIND = "policy"

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        return replace(spec, policy=PolicySpec(
            actor_keys=tuple(self.actor_keys), critic_keys=tuple(self.critic_keys),
            cnn=None, mlp=dict(self.mlp or DEFAULT_MLP),
        ))


# ----------------------------------------------------------------------
# Action DR stage
@dataclass(frozen=True)
class DomainRandomizationActions(Stage):
    steer_noise: float = 0.0
    speed_noise: float = 0.0
    delay_steps: int = 0

    KIND = "action_dr"

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        return replace(spec, action_dr=ActionDRSpec(
            steer_noise=self.steer_noise, speed_noise=self.speed_noise,
            delay_steps=self.delay_steps,
        ))


# ----------------------------------------------------------------------
# Algorithm stages (optional terminal; usually inferred)
@dataclass(frozen=True)
class PPO(Stage):
    clip: float = 0.2
    epochs: int = 5
    minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3.0e-4
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    horizon: int = 24

    KIND = "algorithm"

    def _ppo_dict(self):
        return {
            "clip": self.clip, "epochs": self.epochs,
            "minibatches": self.minibatches, "gamma": self.gamma,
            "gae_lambda": self.gae_lambda, "lr": self.lr,
            "entropy_coef": self.entropy_coef,
            "max_grad_norm": self.max_grad_norm, "horizon": self.horizon,
        }

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        return replace(spec, algorithm=AlgorithmSpec(kind="ppo", ppo=self._ppo_dict()))


@dataclass(frozen=True)
class PPOLagrangian(PPO):
    budget: Optional[float] = None          # None => taken from the env stage
    pid: tuple[float, float, float] = DEFAULT_PID
    cost_gae_lambda: float = 0.95
    lambda_init: float = 0.0

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        return replace(spec, algorithm=AlgorithmSpec(
            kind="ppo_lagrangian", ppo=self._ppo_dict(),
            lagrangian={
                "budget": self.budget, "pid": tuple(self.pid),
                "cost_gae_lambda": self.cost_gae_lambda,
                "lambda_init": self.lambda_init,
            },
        ))


@dataclass(frozen=True)
class Algo(PPO):
    """Terminal stage selecting a CUSTOM registered algorithm by kind.

    The PPO hyperparameters double as generic on-policy knobs (horizon,
    minibatches, lr, ...); `params` carries anything algorithm-specific.
    Register the implementation with
    `@register_algorithm("my_kind")` in experiment/algorithms.py's registry —
    see the Algorithm protocol there for the full contract.
    """

    kind: str = "ppo"
    params: Optional[dict] = None

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        return replace(spec, algorithm=AlgorithmSpec(
            kind=self.kind, ppo=self._ppo_dict(), params=dict(self.params or {})))


# ----------------------------------------------------------------------
def _infer_algorithm(spec: ExperimentSpec) -> ExperimentSpec:
    """Cost-emitting env => PPO-Lagrangian; else PPO. Fill missing budgets."""
    env = spec.env
    if spec.algorithm is None:
        if env is not None and env.emits_cost:
            algo = AlgorithmSpec(kind="ppo_lagrangian", ppo=dict(DEFAULT_PPO),
                                 lagrangian={
                                     "budget": env.cost_budget,
                                     "pid": DEFAULT_PID,
                                     "cost_gae_lambda": 0.95,
                                     "lambda_init": 0.0,
                                 })
        else:
            algo = AlgorithmSpec(kind="ppo", ppo=dict(DEFAULT_PPO))
        return replace(spec, algorithm=algo)
    algo = spec.algorithm
    if (algo.kind == "ppo_lagrangian" and algo.lagrangian.get("budget") is None
            and env is not None and env.cost_budget is not None):
        lag = dict(algo.lagrangian, budget=env.cost_budget)
        return replace(spec, algorithm=replace(algo, lagrangian=lag))
    return spec
