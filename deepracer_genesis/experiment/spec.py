"""Frozen ExperimentSpec value the `>>` DSL builds, hashed by content into a
run id (identical configs share a run dir; runs retrain and overwrite)."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from ..envs.features import FeatureSet
    from ..envs.rewards import RewardFn


def _json_default(o):
    """Record a pluggable callable/class by its name so the spec dump stays
    JSON and the run-dir id stays stable."""
    return getattr(o, "__qualname__", None) or getattr(o, "__name__", None) or repr(o)


class SpecError(ValueError):
    """A structurally or semantically invalid experiment declaration."""


VALID_COST_FNS = ("offtrack", "offtrack_or_overspeed", "crash")


@dataclass(frozen=True)
class EnvSpec:
    """The simulator slice of the spec (filled by an Environment stage).

    Attributes:
        modality: whether the env yields camera images or feature vectors.
        render: which renderer backs a camera env (or none for feature envs).
        resolution: rendered image width/height.
        fov: camera field of view in degrees.
        lookahead_k: number of upcoming waypoints exposed to the policy.
        feature_set: FeatureSet subclass to assemble the state, or None for the default.
        feature_params: tuning knobs (horizons, history lengths) for the feature set.
        tracks: track names the env trains across.
        num_envs: parallel environment count.
        max_speed: action-cap speed in m/s, or None for the physics default.
        random_start: randomize spawn waypoint and lateral/yaw offset per episode.
        random_direction: flip driving direction (CW/CCW) per episode.
        reward: reward callable, or None for the built-in default.
        reward_scales: per-term scale overrides for the reward.
        emits_cost: whether the env produces a cost signal for SafeRL.
        cost_fn: which cost function to emit.
        cost_budget: per-episode cost budget.
        backend: Genesis compute backend ("gpu" or "cpu") — Part M.
        view: interactive/offscreen view renderer ("none"/"gui"/"spectator"/
            "topdown") — Part M.
    """

    modality: Literal["camera", "feature"]
    render: Literal["madrona", "nyx", "none"] = "none"
    backend: Literal["gpu", "cpu"] = "gpu"
    view: Literal["none", "gui", "spectator", "topdown"] = "none"
    realtime_factor: float = 1.0   # viewer pacing (view="gui"); <=0 = uncapped
    resolution: tuple[int, int] = (160, 120)
    fov: float = 90.0
    max_speed: float | None = None
    # camera frames stacked along channels (1 = single frame). Gives the
    # actor temporal context (ego-velocity is unobservable from one frame);
    # stack order oldest-first, fresh episodes prime by repeating the first
    # frame — both mirrored exactly by the car node (deployment contract).
    # Default 4 per the maintainer's ruling (matches AWS/dr-gym precedent).
    frame_stack: int = 4
    lookahead_k: int = 10
    # the FeatureSet subclass the env assembles (envs/features.py): None keeps
    # the default ClassicFeatures (waypoint lookahead); PerceptionFeatures gives
    # the sim2real CNN targets + error channels. feature_params tunes it.
    feature_set: "type[FeatureSet] | None" = None
    feature_params: dict = field(default_factory=dict)
    # per-signal actor/critic obs routing (Part K.3), feature mode only. None =
    # the single "state" vector (default). When set:
    #   {"base": (block names...), "actor": sel, "critic": sel}
    # where each sel is a subset of base or ("*",) (= all base blocks). The env
    # then emits distinct "obs_actor"/"obs_critic" vectors instead of "state" —
    # the vector-mode privileged-critic analogue (critic sees a superset).
    obs_routing: Optional[dict] = None
    tracks: tuple[str, ...] = ("reinvent_base",)
    num_envs: int = 512
    # spawn randomization: every episode starts at a random waypoint with
    # lateral/yaw noise; laps are measured by cumulative progress from the
    # spawn, so a "completed lap" ends back at that same random location
    random_start: bool = True
    # coin-flip the driving direction (CW vs CCW) each episode; heading /
    # progress / lookahead observations follow the chosen direction
    random_direction: bool = False
    # reward: a reward CALLABLE (envs/rewards.py: env -> {term: (N,) tensor}) +
    # scale overrides. None keeps the built-in `deepracer` default. The fn's
    # NAME is recorded in the run-dir id (not its body); runs always retrain.
    reward: "RewardFn | None" = None
    reward_scales: dict = field(default_factory=dict)
    emits_cost: bool = False
    cost_fn: Optional[str] = None
    cost_budget: Optional[float] = None


@dataclass(frozen=True)
class ObsDRSpec:
    """Observation-side domain randomization applied at reset/render.

    Attributes:
        image_aug: image augmentation settings for the rendered observation.
        camera_jitter: per-render camera pose/intrinsics jitter.
        physics: physics parameter overrides applied env-side at reset.
        appearance: per-env, per-episode color remap of the rendered scene.
        pixel_noise: gaussian per-pixel noise scale added in the render path
            (0 = off); the catalogued ``vision.pixel_noise`` visual knob.
    """

    image_aug: dict = field(default_factory=dict)
    camera_jitter: dict = field(default_factory=dict)
    physics: dict = field(default_factory=dict)   # applied env-side at reset
    # per-env, per-episode color remap of the rendered observation
    # ({"world_color": strength}); see DomainRandomizationTrackAppearance
    appearance: dict = field(default_factory=dict)
    pixel_noise: float = 0.0                       # gaussian render-path noise
    # Part P.1: per-env Nyx sky DR ({"tint": (lo,hi), "multiplier": (lo,hi)}),
    # baked at scene.build() -> per-env-fixed (per run), not per-episode.
    env_map: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EncoderSpec:
    """Optional feature-encoder stage inserted between env and policy.

    Attributes:
        kind: which encoder to apply (none or a frozen CNN).
        checkpoint: path to the pretrained encoder weights.
        output_dim: dimensionality of the encoded feature vector.
        layer: which network layer to tap for features.
        out_key: observation key under which encoded features are exposed.
    """

    kind: Literal["none", "frozen_cnn"] = "none"
    checkpoint: Optional[str] = None
    output_dim: Optional[int] = None
    layer: Optional[str] = None
    out_key: str = "encoded"


@dataclass(frozen=True)
class PolicySpec:
    """The policy/network slice of the spec (filled by a Policy stage).

    Attributes:
        actor_keys: observation keys the actor consumes.
        critic_keys: observation keys the critic consumes.
        cnn: CNN config, or None for a pure vector policy.
        mlp: MLP head configuration.
        actions: discrete (steer, speed) action list, or None for continuous.
    """

    actor_keys: tuple[str, ...]
    critic_keys: tuple[str, ...]
    cnn: Optional[dict] = None               # None => pure vector policy
    mlp: dict = field(default_factory=dict)
    # None => continuous TanhNormal over [steer, speed] in [-1, 1]^2 (default).
    # A tuple of (steer, speed) pairs => DISCRETE Categorical policy over that
    # action list — the original AWS DeepRacer action-space style.
    actions: Optional[tuple] = None
    # Overrides merged into the rsl-rl distribution_cfg (e.g. init_std,
    # std_range, learn_std). Needed because the unbounded learned std is
    # bistable on the camera task: entropy_coef=0.01 explodes it (18+),
    # <=0.001 collapses it (0.02) — a std_range ceiling gives entropy a
    # stable operating point. None => backend defaults.
    distribution: Optional[dict] = None


@dataclass(frozen=True)
class ActionDRSpec:
    """Action-side domain randomization for continuous policies.

    Attributes:
        steer_noise: magnitude of noise added to the steering action.
        speed_noise: magnitude of noise added to the speed action.
        delay_steps: number of steps to delay applied actions.
    """

    steer_noise: float = 0.0
    speed_noise: float = 0.0
    delay_steps: int = 0


@dataclass(frozen=True)
class AlgorithmSpec:
    """Training-algorithm slice; `cls` is a custom algorithm class or None.

    Attributes:
        cls: a custom algorithm class, or None for the built-in rsl-rl PPO.
        ppo: PPO hyperparameters.
        lagrangian: Lagrangian settings (budget, PID gains, ...).
        params: free-form parameters for custom algorithm classes.
    """

    cls: "type | None" = None   # None => built-in rsl-rl PPO
    ppo: dict = field(default_factory=dict)
    lagrangian: dict = field(default_factory=dict)  # budget, pid=(kp,ki,kd), ...
    params: dict = field(default_factory=dict)      # free-form for custom classes

    @property
    def resolved_cls(self):
        """The custom Algorithm class, or None for the default rsl-rl PPO."""
        return self.cls

    @property
    def requires_cost(self) -> bool:
        """Whether this is a cost/safe-RL algorithm (has a Lagrangian budget)."""
        return bool(self.lagrangian)


@dataclass(frozen=True)
class EvalConfig:
    """First-class evaluation config (Part N).

    Makes eval intent explicit instead of silently defaulting. The out-of-loop
    per-track holdout eval is opt-in: it runs after ``fit()`` only when
    ``real_tracks`` is non-empty.

    Attributes:
        real_tracks: out-of-loop holdout tracks evaluated INDEPENDENTLY (a fresh
            single-track sim per track); empty = no holdout eval.
        eval_num_envs: parallel envs for each eval rollout.
        eval_episodes: episodes per eval (None = derive from the rollout window).
        charts: whether to render eval charts (matplotlib, optional extra).
        gui: open the interactive viewer during the out-of-loop holdout eval so
            you can watch the policy drive each real track (needs a display; use
            a small ``eval_num_envs``). Orthogonal to the obs renderer — the
            window shows the cars while the policy still runs on its own obs.
    """

    real_tracks: tuple[str, ...] = ()
    eval_num_envs: int = 64
    eval_episodes: Optional[int] = None
    charts: bool = True
    gui: bool = False


@dataclass(frozen=True)
class ExperimentSpec:
    """Frozen, content-hashed experiment config assembled by the `>>` DSL.

    Attributes:
        env: the environment/simulator slice.
        obs_dr: observation-side domain randomization.
        encoder: optional feature-encoder stage.
        policy: the policy/network slice.
        action_dr: action-side domain randomization.
        algorithm: the training-algorithm slice.
        eval: evaluation config (periodic + out-of-loop holdout + charts).
        resume: checkpoint the policy starts from, or None for random weights.
        total_env_steps: total environment steps to train for.
        eval_every_steps: eval interval in env-steps (0 = final eval only).
        seed: random seed.
        ablation_group: bookkeeping tag grouping related runs.
        variant: bookkeeping tag naming this run within its group.
    """

    env: Optional[EnvSpec] = None
    obs_dr: ObsDRSpec = field(default_factory=ObsDRSpec)
    encoder: EncoderSpec = field(default_factory=EncoderSpec)
    policy: Optional[PolicySpec] = None
    action_dr: ActionDRSpec = field(default_factory=ActionDRSpec)
    algorithm: Optional[AlgorithmSpec] = None
    eval: EvalConfig = field(default_factory=EvalConfig)
    resume: Optional[str] = None      # checkpoint to start from (fine-tuning)
    total_env_steps: int = 5_000_000
    eval_every_steps: int = 0        # 0 = final eval only; N = also every N env-steps
    seed: int = 0
    ablation_group: Optional[str] = None
    variant: Optional[str] = None

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """One-way dump (tuples -> lists, callables -> their name) for the
        hashing + run records."""
        return json.loads(json.dumps(asdict(self), default=_json_default))

    def id(self) -> str:
        """Content-hash identity (sha1 of the config JSON, excluding the
        ablation_group/variant tags) so equal configs share a run dir."""
        # sha1, NOT built-in hash(): identity must be stable across processes.
        # ablation_group/variant are bookkeeping tags, not configuration —
        # the same training config keeps one id however it is tagged.
        payload = {k: v for k, v in self.to_dict().items()
                   if k not in ("ablation_group", "variant")}
        return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

    def run_dir(self, root: str = "runs") -> str:
        group = self.ablation_group or "default"
        variant = self.variant or "run"
        return f"{root}/{group}/{variant}-{self.seed}-{self.id()}"

    # ------------------------------------------------------------------
    def available_keys(self) -> tuple[str, ...]:
        """Observation keys the env (+ encoder) makes visible to policies."""
        # Part K.3: routed feature envs emit two vectors instead of "state".
        if self.env is not None and self.env.obs_routing is not None:
            keys = ["obs_actor", "obs_critic"]
        else:
            keys = ["state"]
            if self.env is not None and self.env.modality == "camera":
                keys.append("camera")
        if self.encoder.kind != "none":
            keys.append(self.encoder.out_key)
        return tuple(keys)

    def validate(self) -> "ExperimentSpec":
        """Check the spec is structurally and semantically coherent.

        Returns:
            The spec itself, so ``validate()`` can be chained.

        Raises:
            SpecError: If the env, policy, encoder, obs/action DR, or algorithm
                slice is missing or incoherent.
        """
        if self.env is None:
            raise SpecError("pipeline must start with an Environment stage")
        if self.policy is None:
            raise SpecError("pipeline must include exactly one Policy stage")
        self._validate_environment()
        self._validate_obs_routing()
        self._validate_key_routing()
        self._validate_encoder()
        self._validate_obs_dr()
        self._validate_action_dr()
        self._validate_algorithm()
        self._validate_learnability()
        return self

    def _validate_environment(self) -> None:
        """Check env modality/render coherence and the cost signal.

        Raises:
            SpecError: On a modality/render mismatch, nyx multi-track, or a
                bad cost_fn/cost_budget.
        """
        env = self.env
        from ..tracks import exists, names
        for t in env.tracks:
            if not exists(t):
                raise SpecError("unknown track %r; available: %s" % (t, names()))
        match env.modality:
            case "feature" if env.render != "none":
                raise SpecError("feature envs do not render; got render=%r" % env.render)
            case "camera" if env.render not in ("madrona", "nyx"):
                raise SpecError("camera envs need render='madrona'|'nyx'; got %r" % env.render)
        if env.render == "nyx" and len(env.tracks) > 1:
            raise SpecError(
                "heterogeneous tracks are Madrona-only (repo constraint); "
                "render='nyx' with tracks=%r" % (env.tracks,))
        # Part M Tier 2: Madrona/Nyx are GPU-only; camera obs on the CPU backend
        # renders per-env through the (unbatched, slow) RasterizerObsRenderer,
        # which the builder selects by setting vision_renderer='rasterizer'. It
        # is a debug / small-num_envs / no-GPU path, not a throughput path, and —
        # unlike Madrona spatial tiling (Part O) — supports a SINGLE track only.
        if env.modality == "camera" and env.backend == "cpu":
            # multi-track is fine now: spatial tiling is renderer-agnostic, so
            # each track sits on its own tile and a camera only frames its own.
            warnings.warn(
                "camera obs on backend='cpu' uses the per-env RasterizerObsRenderer "
                "(unbatched, far slower than Madrona) — a debug / small-num_envs / "
                "no-GPU path, not a training-throughput path", stacklevel=2)
        if env.emits_cost:
            if env.cost_fn not in VALID_COST_FNS:
                raise SpecError("cost_fn must be one of %s; got %r" % (VALID_COST_FNS, env.cost_fn))
            if env.cost_budget is None or env.cost_budget <= 0:
                raise SpecError("cost-emitting env needs a positive cost_budget")
        # Part M: the interactive viewer is a debug/watch path (needs a display,
        # small batch), not a throughput path — warn, don't hard-fail.
        if env.view == "gui" and env.num_envs > 64:
            warnings.warn(
                "view='gui' opens an interactive window and is a debug/watch path; "
                "num_envs=%d is large for it — consider a small batch" % env.num_envs,
                stacklevel=2)

    def _validate_obs_routing(self) -> None:
        """Check per-signal actor/critic obs routing (Part K.3), if declared.

        Raises:
            SpecError: If routing is set on a camera env, or its base/actor/
                critic block selections are empty or name unknown blocks.
        """
        routing = self.env.obs_routing
        if routing is None:
            return
        from ..envs.features import FEATURE_BLOCKS
        if self.env.modality != "feature":
            raise SpecError(
                "obs_routing is per-signal feature routing; it needs a feature "
                "env (camera obs stays routed per key)")
        base = tuple(routing.get("base", ()))
        if not base:
            raise SpecError("obs_routing needs a non-empty 'base' block list")
        unknown = [n for n in base if n not in FEATURE_BLOCKS]
        if unknown:
            raise SpecError("obs_routing base has unknown block(s) %s; available: %s"
                            % (unknown, sorted(FEATURE_BLOCKS)))
        resolved = {}
        for role in ("actor", "critic"):
            sel = tuple(routing.get(role, ("*",)))
            if not sel:
                raise SpecError("obs_routing '%s' selection may not be empty" % role)
            if sel == ("*",):
                resolved[role] = set(base)
                continue
            bad = [n for n in sel if n not in base]
            if bad:
                raise SpecError("obs_routing '%s' selects block(s) %s not in base %s"
                                % (role, bad, base))
            resolved[role] = set(sel)
        # privileged-critic invariant: the critic must see at least what the
        # actor sees (obs_critic supersets obs_actor block-wise).
        actor_only = resolved["actor"] - resolved["critic"]
        if actor_only:
            raise SpecError(
                "obs_routing critic must see >= the actor's blocks (privileged "
                "critic); actor-only blocks %s" % sorted(actor_only))

    def _validate_key_routing(self) -> None:
        """Check actor/critic obs keys, discrete actions, and camera routing.

        Raises:
            SpecError: On empty/undefined keys, an asymmetric critic missing
                actor keys, a malformed discrete action space, or a
                camera-key/policy mismatch.
        """
        policy = self.policy
        avail = set(self.available_keys())
        a_keys, c_keys = set(policy.actor_keys), set(policy.critic_keys)
        if not policy.actor_keys:
            raise SpecError("policy actor_keys may not be empty")
        if not a_keys <= avail:
            raise SpecError("actor_keys %s not produced by env/encoder (available: %s)"
                            % (sorted(a_keys - avail), sorted(avail)))
        if not c_keys <= avail:
            raise SpecError("critic_keys %s not produced by env/encoder (available: %s)"
                            % (sorted(c_keys - avail), sorted(avail)))
        # Part K.3: the routed pair (obs_actor, obs_critic) satisfies the
        # asymmetry invariant by construction — obs_critic is the value-vector
        # that supersets the actor's blocks (enforced in _validate_obs_routing) —
        # even though the two KEYS differ, so map obs_actor -> obs_critic here.
        check_a = a_keys
        if self.env is not None and self.env.obs_routing is not None and "obs_actor" in a_keys:
            check_a = (a_keys - {"obs_actor"}) | {"obs_critic"}
        if not check_a <= c_keys:
            raise SpecError("asymmetric policies require critic_keys ⊇ actor_keys; "
                            "actor has %s the critic lacks" % sorted(check_a - c_keys))
        if policy.actions is not None:
            if len(policy.actions) < 2:
                raise SpecError("a discrete action space needs >= 2 actions")
            for a in policy.actions:
                if len(a) != 2 or not all(-1.0 <= float(x) <= 1.0 for x in a):
                    raise SpecError(
                        f"discrete actions are (steer, speed) pairs in [-1, 1]; got {a}")
            if (self.action_dr.steer_noise or self.action_dr.speed_noise
                    or self.action_dr.delay_steps):
                raise SpecError("action DR (noise/delay) operates on continuous "
                                "actions; not compatible with a discrete policy")
        if policy.cnn is None and "camera" in (a_keys | c_keys):
            raise SpecError("a vector policy cannot consume the raw 'camera' key; "
                            "add an encoder stage or use a camera policy")
        if policy.cnn is not None and "camera" not in a_keys:
            raise SpecError("a camera policy's actor must read the 'camera' key")

    def _validate_encoder(self) -> None:
        """Check the frozen-CNN encoder's env/checkpoint/policy requirements.

        Raises:
            SpecError: If a frozen_cnn encoder lacks a camera env, a checkpoint,
                or a downstream vector policy.
        """
        match self.encoder.kind:
            case "frozen_cnn":
                if self.env.modality != "camera":
                    raise SpecError("FrozenCNNToFeatureVector requires an upstream camera env")
                if self.encoder.checkpoint is None:
                    raise SpecError("FrozenCNNToFeatureVector requires a checkpoint path")
                if self.policy.cnn is not None:
                    raise SpecError("FrozenCNNToFeatureVector requires a downstream vector "
                                    "policy (VectorPolicy/AsymmetricVectorPolicy)")

    def _validate_obs_dr(self) -> None:
        """Check observation-DR stages are paired with a camera env.

        Raises:
            SpecError: If appearance/image-aug/camera-jitter DR is used without a
                camera env.
        """
        env, obs_dr = self.env, self.obs_dr
        if obs_dr.appearance and env.modality != "camera":
            raise SpecError("appearance DR recolors the rendered observation; "
                            "it needs a camera env")
        if obs_dr.env_map and env.modality != "camera":
            raise SpecError("env_map DR randomizes the rendered sky; "
                            "it needs a camera env")
        if env.modality == "camera" and len(env.tracks) > 1 and env.render == "madrona":
            # Part O: sound via spatial tiling (each variant on its own world
            # tile), NOT the broken heterogeneous-morph path. It costs K× track
            # geometry per env (render + memory ~linear in K) — run the Part O
            # benchmark gate before scaling K.
            warnings.warn(
                "multi-track camera training uses Part O spatial tiling: all %d "
                "track meshes load into every env (render + memory scale ~K); "
                "benchmark aggregate steps/sec before scaling the track count"
                % len(env.tracks), stacklevel=2)
        if (obs_dr.image_aug or obs_dr.camera_jitter or obs_dr.pixel_noise) \
                and env.modality != "camera":
            raise SpecError("DomainRandomizationCamera requires a camera env")

    def _validate_action_dr(self) -> None:
        """Check action-DR magnitudes are non-negative.

        Raises:
            SpecError: If delay_steps or a noise magnitude is negative.
        """
        if self.action_dr.delay_steps < 0:
            raise SpecError("delay_steps must be >= 0")
        if self.action_dr.steer_noise < 0 or self.action_dr.speed_noise < 0:
            raise SpecError("action noise magnitudes must be >= 0")

    def _validate_algorithm(self) -> None:
        """Check the algorithm matches the env's cost signal and budget.

        Raises:
            SpecError: If the algorithm is missing, PPO-Lagrangian lacks a cost
                env/budget, or the env and algorithm budgets conflict.
        """
        env, algo = self.env, self.algorithm
        if algo is None:
            raise SpecError("algorithm missing: build() must run _infer_algorithm")
        if algo.cls is not None:
            # a custom Algo(cls=...) must speak the rsl-rl runner interface, or it
            # would fail cryptically deep inside OnPolicyRunner — check up front.
            from ..algorithms.protocol import missing_algorithm_methods
            missing = missing_algorithm_methods(algo.cls)
            if missing:
                raise SpecError(
                    "custom algorithm %s is missing rsl-rl interface method(s) %s; "
                    "implement algorithms.protocol.RslAlgorithm (subclassing "
                    "rsl_rl.algorithms.PPO and overriding compute_returns/update is "
                    "the easy path)"
                    % (getattr(algo.cls, "__name__", algo.cls), missing))
        if algo.requires_cost:
            if not env.emits_cost:
                raise SpecError(
                    "%s requires a SafeRL* env that emits a cost signal"
                    % "PPO-Lagrangian")
            if not algo.lagrangian.get("budget"):
                raise SpecError(
                    "%s needs a budget (explicit or from the env stage)"
                    % "PPO-Lagrangian")
            if (env.cost_budget is not None
                    and algo.lagrangian.get("budget") not in (None, env.cost_budget)):
                raise SpecError(
                    "conflicting budgets: env.cost_budget=%r vs algorithm.lagrangian"
                    "['budget']=%r — sweep 'env.cost_budget' (ablation.override keeps "
                    "them in sync)" % (env.cost_budget, algo.lagrangian.get("budget")))
        elif env.emits_cost:
            warnings.warn(
                "cost-emitting env trained with a reward-only algorithm (%s): the "
                "cost stream is collected but unconstrained (was this intentional?)"
                % "PPO-Lagrangian",
                stacklevel=2)

    # ------------------------------------------------------------------
    def _routed_signals(self, role: str) -> "set[str]":
        """Signals an ``obs_actor``/``obs_critic`` vector recovers (Part K.3).

        Resolves the routed block selection for ``role`` (``"actor"``/
        ``"critic"``); a ``("*",)`` selection means "all base blocks", mapped to
        ``{"*"}`` to match the coarse ``state`` behavior (so a privileged
        ``critic=["*"]`` stays warning-free), while an explicit block subset maps
        to exactly the signals those blocks carry.
        """
        from ..envs.features import blocks_signal_set
        routing = self.env.obs_routing or {}
        sel = tuple(routing.get(role, ("*",)))
        if sel == ("*",):
            return {"*"}
        base = tuple(routing.get("base", ()))
        chosen = tuple(n for n in base if n in set(sel))   # base order
        return blocks_signal_set(chosen)

    def _signals_for_keys(self, keys: "set[str]") -> "set[str]":
        """Map obs keys to the signal names a consumer can recover (K.5).

        A feature vector (``state``) carries every signal; a raw camera or a
        frozen-CNN encoding over it recovers only the ``pixel_observable``
        subset; the routed ``obs_actor``/``obs_critic`` vectors (Part K.3) carry
        exactly their selected blocks' signals. ``"*"`` (all signals) is passed
        through for a config that lists signals explicitly.

        Args:
            keys: the actor's or critic's obs keys (from the PolicySpec).

        Returns:
            The set of signal names that consumer can condition on, or ``{"*"}``
            when it sees the full feature vector (every signal).
        """
        if "*" in keys or "state" in keys:
            return {"*"}
        # Part K.3 routed vectors: the actor/critic each carry their own blocks.
        if "obs_actor" in keys or "obs_critic" in keys:
            out: set[str] = set()
            if "obs_actor" in keys:
                out |= self._routed_signals("actor")
            if "obs_critic" in keys:
                out |= self._routed_signals("critic")
            return {"*"} if "*" in out else out
        # pixel-only consumer: raw camera or a CNN encoding over it. It can only
        # recover the pixel_observable signals.
        from ..envs.signals import SIGNALS
        pixel_keys = {"camera", self.encoder.out_key}
        if keys & pixel_keys:
            return {n for n, s in SIGNALS.items() if s.pixel_observable}
        return set()

    def _validate_learnability(self) -> None:
        """Warn (never raise) if the reward/cost read signals the CRITIC can't see (K.5).

        Because the reward and cost declare their signal ``reads`` over the
        shared signal vocabulary (Part K.4), the build statically checks
        ``reward.reads ∪ cost.reads ⊆ critic-visible signals`` and warns when a
        term is genuinely **unlearnable** — no network (not even the critic) can
        recover its inputs, so the run is wasted. Purely additive: it emits at
        most one :class:`UserWarning`, changing no run behavior. An undeclared
        custom reward (no ``reads``) is skipped.

        The softer "the actor can't directly act on this signal" advisory that
        :func:`~deepracer_genesis.envs.signals.check_learnability` also returns is
        **deliberately not emitted here**: it fires on the intended asymmetric
        privileged-critic design (a critic that sees ``state``/progress while a
        pixel-only actor does not — the very pattern the repo is built on), so a
        per-build warning about it would flag correct experiments as suspect.
        That advisory stays available in ``check_learnability`` for opt-in use.

        Note:
            Coarse per-key routing (K.3 not required): ``state`` -> all signals,
            ``camera``/encoder -> the pixel-observable subset.
        """
        from ..envs.rewards import cost_reads, deepracer, reward_reads
        from ..envs.signals import check_learnability

        env, policy = self.env, self.policy
        reward_fn = env.reward or deepracer
        r_reads = reward_reads(reward_fn)
        c_reads = cost_reads(env.cost_fn) if env.emits_cost else frozenset()
        if not r_reads and not c_reads:
            return   # nothing declared to verify (undeclared custom reward)

        actor_signals = self._signals_for_keys(set(policy.actor_keys))
        critic_signals = self._signals_for_keys(set(policy.critic_keys))
        problems = check_learnability(
            reward_reads=set(r_reads), cost_reads=set(c_reads),
            actor_signals=actor_signals, critic_signals=critic_signals)
        # only the hard "unlearnable" problems (critic can't see the signal);
        # the soft privileged-critic advisory is intentional, not a warning.
        unlearnable = [p for p in problems if p.startswith("unlearnable")]
        if unlearnable:
            warnings.warn("learnability: " + "; ".join(unlearnable), stacklevel=2)
