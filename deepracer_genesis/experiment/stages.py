"""The `>>` builder DSL: Stages fold into a spec, `Pipeline.build()` validates.
Build-time only — nothing here runs per-step.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..envs.features import FeatureSet
    from ..envs.rewards import RewardFn
    from ..randomization.spaces import Space

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


def _dr_cfg_value(x):
    """Normalize a DR field to its ``cfg['rand']`` shape.

    Accepts a shared :class:`~deepracer_genesis.randomization.spaces.Space`
    (the same type HPO uses — Part H) and emits its ``to_cfg()`` value (a
    ``(lo, hi)`` tuple for ranges, a scalar magnitude for ``SymRange``), or a
    raw tuple/scalar unchanged. Keeps the exact shape ``domain_rand`` expects.

    Args:
        x: A ``Space`` object, a ``(lo, hi)`` tuple, or a scalar.

    Returns:
        The cfg-shaped value (tuple or scalar).
    """
    from ..randomization.spaces import Space
    if isinstance(x, Space):
        return x.to_cfg()
    return tuple(x) if isinstance(x, (list, tuple)) else x


# ----------------------------------------------------------------------
class Stage:
    """One slice of the spec: subclasses implement `apply(spec) -> spec` and set
    KIND; `>>` composes stages into a Pipeline.

    Attributes:
        KIND: Category tag used by the pipeline to validate stage ordering/counts.
    """

    KIND: str = "stage"

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        raise NotImplementedError

    def __rshift__(self, other):
        return Pipeline([self]) >> other


class Pipeline:
    """An ordered chain of Stages, composed with `>>`; `build()` folds and validates.
    Build-time only — nothing here runs per-step.

    Attributes:
        stages: Ordered list of Stages to fold into the spec.
    """

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
                            ("obs_dr_camera", 1), ("obs_dr_physics", 1),
                            ("eval", 1)):
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
    """State-vector env: no rendering; the vector is picked by feature_set.

    Attributes:
        feature_set: FeatureSet subclass to assemble the state, or None for default.
        feature_params: Extra parameters for the chosen feature set.
        obs_routing: Per-signal actor/critic routing (Part K.3), or None for a
            single ``state`` vector. When set, ``{"base": (block names...),
            "actor": sel, "critic": sel}`` emits ``obs_actor``/``obs_critic``.
        lookahead_k: Number of upcoming waypoints exposed to the agent.
        tracks: Track names to train on; more than one trains heterogeneously.
        num_envs: Parallel simulation instances.
        random_start: Whether episodes begin at a random track position.
        random_direction: Whether each episode randomizes CW/CCW travel.
        KIND: Stage category tag (environment).
    """

    feature_set: "type[FeatureSet] | None" = None   # None -> ClassicFeatures
    feature_params: Optional[dict] = None
    obs_routing: Optional[dict] = None              # Part K.3 (feature mode only)
    lookahead_k: int = 10
    tracks: tuple[str, ...] = ("reinvent_base",)   # >1 => heterogeneous per-env
    num_envs: int = 512
    random_start: bool = True
    random_direction: bool = False     # coin-flip CW/CCW per episode
    backend: str = "gpu"               # "gpu" | "cpu" (Part M)
    view: str = "none"                 # "none" | "gui" | "spectator" | "topdown"
    realtime_factor: float = 1.0       # viewer pacing (view="gui"); <=0 = uncapped

    max_speed: Optional[float] = None   # plafond d'action en m/s

    KIND = "environment"

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        return replace(spec, env=EnvSpec(
            modality="feature", render="none",
            feature_set=self.feature_set,
            feature_params=dict(self.feature_params or {}),
            obs_routing=dict(self.obs_routing) if self.obs_routing is not None else None,
            lookahead_k=self.lookahead_k,
            tracks=tuple(self.tracks), num_envs=self.num_envs,
            random_start=self.random_start,
            random_direction=self.random_direction,
            max_speed=self.max_speed,
            backend=self.backend, view=self.view,
            realtime_factor=self.realtime_factor,
        ))


@dataclass(frozen=True)
class CameraEnvironment(Stage):
    """Front-RGB-camera env; >1 `tracks` trains heterogeneously (Madrona only).

    Attributes:
        render: Rendering backend used to produce the camera image.
        resolution: Camera image width and height in pixels.
        fov: Horizontal field of view in degrees.
        lookahead_k: Number of upcoming waypoints exposed to the agent.
        feature_set: FeatureSet subclass for the auxiliary state, or None for default.
        feature_params: Extra parameters for the chosen feature set.
        tracks: Track names to train on; more than one trains heterogeneously.
        num_envs: Parallel simulation instances.
        random_start: Whether episodes begin at a random track position.
        random_direction: Whether each episode randomizes CW/CCW travel.
        KIND: Stage category tag (environment).
    """

    render: str = "madrona"
    resolution: tuple[int, int] = (160, 120)
    fov: float = 90.0
    frame_stack: int = 4               # frames stacked along channels (contract default)
    lookahead_k: int = 10
    feature_set: "type[FeatureSet] | None" = None
    feature_params: Optional[dict] = None
    tracks: tuple[str, ...] = ("reinvent_base",)
    num_envs: int = 128
    random_start: bool = True
    random_direction: bool = False     # coin-flip CW/CCW per episode
    backend: str = "gpu"               # "gpu" | "cpu" (Part M)
    view: str = "none"                 # "none" | "gui" | "spectator" | "topdown"
    realtime_factor: float = 1.0       # viewer pacing (view="gui"); <=0 = uncapped

    max_speed: Optional[float] = None   # plafond d'action en m/s

    KIND = "environment"

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        return replace(spec, env=EnvSpec(
            modality="camera", render=self.render,
            resolution=tuple(self.resolution), fov=self.fov,
            frame_stack=self.frame_stack,
            lookahead_k=self.lookahead_k,
            feature_set=self.feature_set,
            feature_params=dict(self.feature_params or {}),
            tracks=tuple(self.tracks),
            num_envs=self.num_envs, random_start=self.random_start,
            random_direction=self.random_direction,
            max_speed=self.max_speed,
            backend=self.backend, view=self.view,
            realtime_factor=self.realtime_factor,
        ))


@dataclass(frozen=True)
class SafeRLFeatureEnvironment(FeatureEnvironment):
    """Feature env that also emits a cost signal (=> PPO-Lagrangian inferred).

    Attributes:
        cost: Named cost function to accumulate per step.
        budget: Per-episode cost budget for the safety constraint.
    """
    cost: str = "offtrack"
    budget: float = 25.0

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        spec = super().apply(spec)
        return replace(spec, env=replace(
            spec.env, emits_cost=True, cost_fn=self.cost, cost_budget=self.budget))


@dataclass(frozen=True)
class SafeRLCameraEnvironment(CameraEnvironment):
    """Camera env that also emits a cost signal (=> PPO-Lagrangian inferred).

    Attributes:
        cost: Named cost function to accumulate per step.
        budget: Per-episode cost budget for the safety constraint.
    """
    cost: str = "offtrack"
    budget: float = 25.0

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        spec = super().apply(spec)
        return replace(spec, env=replace(
            spec.env, emits_cost=True, cost_fn=self.cost, cost_budget=self.budget))


def discrete_grid(steer_bins: int = 5, speed_bins: int = 2,
                  max_speed_frac: float = 1.0) -> tuple:
    """Build the classic DeepRacer (steer x speed) action grid in [-1, 1] units.

    Args:
        steer_bins: Number of steering values, evenly spaced over [-1, 1].
        speed_bins: Number of speed values; throttle -1 maps to min speed,
            so the bins spread over the upper speed range.
        max_speed_frac: Fraction of full speed reached by the fastest bin.

    Returns:
        Tuple of (steer, speed) pairs, rounded to 3 decimals.
    """
    import numpy as np
    steers = np.linspace(-1.0, 1.0, steer_bins)
    # throttle -1 maps to min speed; spread the bins over the upper range
    speeds = np.linspace(-0.4, -1.0 + 2.0 * max_speed_frac, speed_bins)
    return tuple((round(float(st), 3), round(float(sp), 3))
                 for sp in speeds for st in steers)


# ----------------------------------------------------------------------
# Reward stage
@dataclass(frozen=True)
class RewardShaping(Stage):
    """Set the reward callable (``None`` keeps built-in ``deepracer``) and/or
    override entries of the default reward_scales dict.

    Attributes:
        fn: Custom reward callable, or None to keep the built-in reward.
        scales: Overrides merged into the default reward-scales dict.
        KIND: Stage category tag (reward).
    """

    fn: Optional["RewardFn"] = None
    scales: Optional[dict] = None

    KIND = "reward"

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        return replace(spec, env=replace(
            spec.env, reward=self.fn, reward_scales=dict(self.scales or {})))


# ----------------------------------------------------------------------
# Observation DR stages
@dataclass(frozen=True)
class DomainRandomizationTrackAppearance(Stage):
    """World-appearance DR: each env draws its own per-episode color remap of the
    rendered observation; `strength` in [0, 1] scales all ranges.

    Attributes:
        strength: Magnitude in [0, 1] scaling all color-remap ranges.
        env_map_tint: Per-env Nyx sky-tint range ``(lo, hi)``, or None (Part P.1;
            baked at build -> per-env-fixed, per run).
        env_map_multiplier: Per-env Nyx sky exposure range ``(lo, hi)``, or None.
        KIND: Stage category tag (obs_dr_appearance).
    """

    strength: float = 0.6
    env_map_tint: Optional[tuple[float, float]] = None
    env_map_multiplier: Optional[tuple[float, float]] = None

    KIND = "obs_dr_appearance"

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        env_map = {}
        if self.env_map_tint:
            env_map["tint"] = tuple(self.env_map_tint)
        if self.env_map_multiplier:
            env_map["multiplier"] = tuple(self.env_map_multiplier)
        return replace(spec, obs_dr=replace(spec.obs_dr, appearance={
            "world_color": float(self.strength),
        }, env_map=env_map))


@dataclass(frozen=True)
class DomainRandomizationCamera(Stage):
    """Image-augmentation DR applied to the rendered camera observation.

    Attributes:
        brightness: Random brightness scale range, or None to skip.
        contrast: Random contrast scale range, or None to skip.
        saturation: Random saturation scale range, or None to skip.
        hue: Maximum random hue shift.
        blur: Blur augmentation strength.
        cutout: Probability of a cutout patch per frame.
        noise: Additive Gaussian noise sigma.
        gamma: Random gamma/exposure curve range, or None to skip (models the
            render's lack of auto-exposure).
        white_balance: Per-channel gain magnitude (colour cast; also insures
            against the documented Madrona R<->G swap).
        vignette: Max radial corner-darkening strength in [0, 1].
        distortion: Max |radial| barrel/pincushion coefficient (wide-angle lens).
        crop: Max fraction croppable per frame, resized back (FOV/PP jitter).
        shot_noise: Brightness-dependent (sqrt-intensity) sensor noise scale.
        latency_steps: Camera pipeline delay in control steps (stateful; the
            policy sees the frame from this many steps ago).
        frame_drop: Per-step probability of repeating the previous frame (a
            dropped/stale sensor read; stateful).
        pixel_noise: Gaussian per-pixel noise scale added in the render path.
        camera_jitter: Enable/override per-episode camera pose jitter.
        KIND: Stage category tag (obs_dr_camera).
    """

    brightness: Optional[tuple[float, float]] = None
    contrast: Optional[tuple[float, float]] = None
    saturation: Optional[tuple[float, float]] = None
    hue: float = 0.0
    blur: float = 0.0
    cutout: float = 0.0            # probability of a cutout patch per frame
    noise: float = 0.0             # additive gaussian sigma
    gamma: Optional[tuple[float, float]] = None
    white_balance: float = 0.0     # per-channel gain magnitude
    vignette: float = 0.0          # max corner-darkening strength
    distortion: float = 0.0        # max |radial| barrel coefficient
    crop: float = 0.0              # max crop fraction, resized back (FOV jitter)
    shot_noise: float = 0.0        # sqrt-intensity sensor noise scale
    latency_steps: int = 0         # camera pipeline delay in steps (stateful)
    frame_drop: float = 0.0        # prob of repeating the previous frame (stateful)
    pixel_noise: float = 0.0       # gaussian render-path noise (renderer applies)
    camera_jitter: bool | dict = False

    KIND = "obs_dr_camera"

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        aug = {}
        if self.brightness:    aug["brightness"] = tuple(self.brightness)
        if self.contrast:      aug["contrast"] = tuple(self.contrast)
        if self.saturation:    aug["saturation"] = tuple(self.saturation)
        if self.hue:           aug["hue"] = self.hue
        if self.blur:          aug["blur"] = self.blur
        if self.cutout:        aug["cutout"] = self.cutout
        if self.noise:         aug["noise"] = self.noise
        if self.gamma:         aug["gamma"] = tuple(self.gamma)
        if self.white_balance: aug["white_balance"] = self.white_balance
        if self.vignette:      aug["vignette"] = self.vignette
        if self.distortion:    aug["distortion"] = self.distortion
        if self.crop:          aug["crop"] = self.crop
        if self.shot_noise:    aug["shot_noise"] = self.shot_noise
        if self.latency_steps: aug["latency_steps"] = self.latency_steps
        if self.frame_drop:    aug["frame_drop"] = self.frame_drop
        if self.camera_jitter is True:
            jitter = {"pitch_deg": 2.0, "pos_m": 0.005}
        elif isinstance(self.camera_jitter, dict):
            jitter = dict(self.camera_jitter)
        else:
            jitter = {}
        return replace(spec, obs_dr=replace(spec.obs_dr, image_aug=aug,
                                            camera_jitter=jitter,
                                            pixel_noise=self.pixel_noise))


@dataclass(frozen=True)
class DomainRandomizationPhysics(Stage):
    """Physics DR: per-env randomization of friction, mass, and actuator gains.

    Attributes:
        friction: Friction scale range.
        mass: Per-link mass shift magnitude in kg.
        com: Per-link center-of-mass shift magnitude in meters.
        gains: Steering/wheel gain scale range.
        armature: Joint armature value range.
        track_width: Per-episode scale on the rulebook track half-width
            (feature mode only; off = (1.0, 1.0)). The rendered mesh width is
            fixed, so a camera view would desync — the env applies this only in
            feature mode (ignored for camera envs).
        KIND: Stage category tag (obs_dr_physics).
    """

    # each field accepts a shared Space (FloatRange/SymRange — Part H) or a
    # raw tuple/scalar; normalized to the cfg shape at build via _dr_cfg_value.
    friction: "Space | tuple[float, float]" = (0.6, 1.4)
    mass: "Space | float" = 0.2              # +- kg per link
    com: "Space | float" = 0.01              # +- m per link
    gains: "Space | tuple[float, float]" = (0.8, 1.2)
    armature: "Space | tuple[float, float]" = (0.0, 0.01)
    track_width: "Space | tuple[float, float]" = (1.0, 1.0)  # off by default

    KIND = "obs_dr_physics"

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        physics = {
            "friction_range": _dr_cfg_value(self.friction),
            "mass_shift_kg": _dr_cfg_value(self.mass),
            "com_shift_m": _dr_cfg_value(self.com),
            "steer_kp_scale": _dr_cfg_value(self.gains),
            "wheel_kv_scale": _dr_cfg_value(self.gains),
            "armature_range": _dr_cfg_value(self.armature),
            "track_width_scale": _dr_cfg_value(self.track_width),
        }
        return replace(spec, obs_dr=replace(spec.obs_dr, physics=physics))


# ----------------------------------------------------------------------
# Encoder stage
@dataclass(frozen=True)
class FrozenCNNToFeatureVector(Stage):
    """Encoder stage: a frozen CNN checkpoint maps images to a feature vector.

    Attributes:
        checkpoint: Path to the frozen CNN weights, or empty for none.
        output_dim: Size of the produced feature vector.
        layer: Named layer to read features from, or None for the default.
        out_key: Observation key under which the encoded vector is stored.
        KIND: Stage category tag (encoder).
    """

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
    """Actor-critic policy where the critic sees extra keys beyond the camera.

    Attributes:
        actor_keys: Observation keys fed to the actor network.
        critic_keys: Observation keys fed to the critic network.
        cnn: CNN configuration override, or None for the default.
        mlp: MLP configuration override, or None for the default.
        actions: Discrete (steer, speed) pairs, or None for continuous control.
        KIND: Stage category tag (policy).
    """

    actor_keys: tuple[str, ...] = ("camera",)
    critic_keys: tuple[str, ...] = ("camera", "state")
    cnn: Optional[dict] = None
    mlp: Optional[dict] = None
    actions: Optional[tuple] = None       # (steer, speed) pairs => discrete
    distribution: Optional[dict] = None   # rsl-rl distribution_cfg overrides

    KIND = "policy"

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        return replace(spec, policy=PolicySpec(
            actor_keys=tuple(self.actor_keys), critic_keys=tuple(self.critic_keys),
            cnn=dict(self.cnn or DEFAULT_CNN), mlp=dict(self.mlp or DEFAULT_MLP),
            actions=tuple(map(tuple, self.actions)) if self.actions else None,
            distribution=dict(self.distribution) if self.distribution else None,
        ))


@dataclass(frozen=True)
class VectorPolicy(Stage):
    """Shared-encoder MLP policy over state vectors (actor and critic share keys).

    Attributes:
        keys: Observation keys fed to both actor and critic.
        mlp: MLP configuration override, or None for the default.
        actions: Discrete (steer, speed) pairs, or None for continuous control.
        KIND: Stage category tag (policy).
    """

    keys: tuple[str, ...] = ("state",)
    mlp: Optional[dict] = None
    actions: Optional[tuple] = None       # (steer, speed) pairs => discrete

    KIND = "policy"

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        return replace(spec, policy=PolicySpec(
            actor_keys=tuple(self.keys), critic_keys=tuple(self.keys),
            cnn=None, mlp=dict(self.mlp or DEFAULT_MLP),
            actions=tuple(map(tuple, self.actions)) if self.actions else None,
        ))


@dataclass(frozen=True)
class AsymmetricVectorPolicy(Stage):
    """MLP policy over state vectors with distinct actor and critic key sets.

    Attributes:
        actor_keys: Observation keys fed to the actor network.
        critic_keys: Observation keys fed to the critic network.
        mlp: MLP configuration override, or None for the default.
        actions: Discrete (steer, speed) pairs, or None for continuous control.
        KIND: Stage category tag (policy).
    """

    actor_keys: tuple[str, ...] = ("state",)
    critic_keys: tuple[str, ...] = ("state",)
    mlp: Optional[dict] = None
    actions: Optional[tuple] = None       # (steer, speed) pairs => discrete

    KIND = "policy"

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        return replace(spec, policy=PolicySpec(
            actor_keys=tuple(self.actor_keys), critic_keys=tuple(self.critic_keys),
            cnn=None, mlp=dict(self.mlp or DEFAULT_MLP),
            actions=tuple(map(tuple, self.actions)) if self.actions else None,
        ))


# ----------------------------------------------------------------------
# Action DR stage
@dataclass(frozen=True)
class DomainRandomizationActions(Stage):
    """Action DR: adds noise and latency to the agent's commanded actions.

    Attributes:
        steer_noise: Standard deviation of steering noise.
        speed_noise: Standard deviation of speed noise.
        delay_steps: Number of steps to delay applied actions.
        KIND: Stage category tag (action_dr).
    """

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
    """Algorithm stage configuring standard PPO hyperparameters.

    Attributes:
        clip: PPO surrogate clipping range.
        epochs: Optimization epochs per iteration.
        minibatches: Minibatches per epoch.
        gamma: Reward discount factor.
        gae_lambda: GAE smoothing coefficient.
        lr: Optimizer learning rate.
        entropy_coef: Entropy bonus coefficient.
        max_grad_norm: Gradient-norm clipping threshold.
        horizon: Rollout steps per env per iteration.
        schedule: "adaptive" retunes lr from the measured KL, "fixed" keeps it.
        desired_kl: KL target the adaptive schedule steers toward.
        KIND: Stage category tag (algorithm).
    """

    clip: float = 0.2
    epochs: int = 5
    minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3.0e-4
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    horizon: int = 24
    schedule: str = "adaptive"
    desired_kl: float = 0.01

    KIND = "algorithm"

    def _ppo_dict(self):
        return {
            "clip": self.clip, "epochs": self.epochs,
            "minibatches": self.minibatches, "gamma": self.gamma,
            "gae_lambda": self.gae_lambda, "lr": self.lr,
            "entropy_coef": self.entropy_coef,
            "max_grad_norm": self.max_grad_norm, "horizon": self.horizon,
            "schedule": self.schedule, "desired_kl": self.desired_kl,
        }

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        return replace(spec, algorithm=AlgorithmSpec(cls=None, ppo=self._ppo_dict()))


@dataclass(frozen=True)
class PPOLagrangian(PPO):
    """PPO with a Lagrangian safety constraint driven by a PID multiplier.

    Attributes:
        budget: Per-episode cost budget, or None to take from the env stage.
        pid: PID gains updating the Lagrange multiplier.
        cost_gae_lambda: GAE smoothing coefficient for the cost advantage.
        lambda_init: Initial Lagrange multiplier value.
    """

    budget: Optional[float] = None          # None => taken from the env stage
    pid: tuple[float, float, float] = DEFAULT_PID
    cost_gae_lambda: float = 0.95
    lambda_init: float = 0.0

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        # cls left None: cost/safe-RL is gated at run() pending an rsl-rl Lagrangian
        return replace(spec, algorithm=AlgorithmSpec(
            cls=None, ppo=self._ppo_dict(),
            lagrangian={
                "budget": self.budget, "pid": tuple(self.pid),
                "cost_gae_lambda": self.cost_gae_lambda,
                "lambda_init": self.lambda_init,
            },
        ))


@dataclass(frozen=True)
class Algo(PPO):
    """Terminal stage selecting a CUSTOM Algorithm class directly; PPO fields
    are generic on-policy knobs and `params` carries algorithm-specific args.

    Attributes:
        cls: Algorithm class to instantiate, or None for the built-in PPO.
        params: Algorithm-specific arguments passed through.
    """

    cls: "type | None" = None
    params: Optional[dict] = None

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        return replace(spec, algorithm=AlgorithmSpec(
            cls=self.cls, ppo=self._ppo_dict(), params=dict(self.params or {})))


# ----------------------------------------------------------------------
# Evaluation stage (optional; sets the first-class EvalConfig — Part N)
@dataclass(frozen=True)
class Evaluation(Stage):
    """Configure evaluation: out-of-loop real-track holdout eval + charts.

    Attributes:
        real_tracks: holdout tracks evaluated independently after training
            (empty = no holdout eval). Pass e.g. ``TrackDataset().holdout``.
        eval_num_envs: parallel envs per eval rollout.
        eval_episodes: episodes per eval (None derives from the rollout window).
        charts: render eval charts (matplotlib optional extra).
        gui: open the interactive viewer during the out-of-loop holdout eval so
            you can watch the policy drive each real track (needs a display;
            keep eval_num_envs small).
        KIND: Stage category tag (eval).
    """

    real_tracks: tuple[str, ...] = ()
    eval_num_envs: int = 64
    eval_episodes: Optional[int] = None
    charts: bool = True
    gui: bool = False

    KIND = "eval"

    def apply(self, spec: ExperimentSpec) -> ExperimentSpec:
        return replace(spec, eval=EvalConfig(
            real_tracks=tuple(self.real_tracks),
            eval_num_envs=self.eval_num_envs,
            eval_episodes=self.eval_episodes,
            charts=self.charts,
            gui=self.gui))


# ----------------------------------------------------------------------
def _infer_algorithm(spec: ExperimentSpec) -> ExperimentSpec:
    """Cost-emitting env => PPO-Lagrangian; else PPO. Fill missing budgets."""
    env = spec.env
    if spec.algorithm is None:
        if env is not None and env.emits_cost:
            # cls None: cost/safe-RL is gated at run() pending an rsl-rl Lagrangian
            algo = AlgorithmSpec(cls=None, ppo=dict(DEFAULT_PPO),
                                 lagrangian={
                                     "budget": env.cost_budget,
                                     "pid": DEFAULT_PID,
                                     "cost_gae_lambda": 0.95,
                                     "lambda_init": 0.0,
                                 })
        else:
            algo = AlgorithmSpec(cls=None, ppo=dict(DEFAULT_PPO))
        return replace(spec, algorithm=algo)
    algo = spec.algorithm
    if (algo.requires_cost and algo.lagrangian.get("budget") is None
            and env is not None and env.cost_budget is not None):
        lag = dict(algo.lagrangian, budget=env.cost_budget)
        return replace(spec, algorithm=replace(algo, lagrangian=lag))
    return spec
