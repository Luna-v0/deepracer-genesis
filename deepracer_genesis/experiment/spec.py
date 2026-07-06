"""ExperimentSpec: the frozen value the `>>` DSL builds (plan section 2).

Identity is content: `id()` hashes the spec, so identical configs collide
(intentional cache) and any field change produces a new run directory.
`to_dict()` is a ONE-WAY dump used for the hash and the run record; nothing
in the authoring path ever loads a spec back from a file.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass, field
from typing import Literal, Optional


class SpecError(ValueError):
    """A structurally or semantically invalid experiment declaration."""


VALID_COST_FNS = ("offtrack", "offtrack_or_overspeed", "crash")


@dataclass(frozen=True)
class EnvSpec:
    modality: Literal["camera", "feature"]
    render: Literal["madrona", "nyx", "none"] = "none"
    resolution: tuple[int, int] = (160, 120)
    fov: float = 90.0
    lookahead_k: int = 10
    features: tuple[str, ...] = ()          # extra feature-vector channels
    tracks: tuple[str, ...] = ("reinvent_base",)
    num_envs: int = 512
    emits_cost: bool = False
    cost_fn: Optional[str] = None
    cost_budget: Optional[float] = None


@dataclass(frozen=True)
class ObsDRSpec:
    image_aug: dict = field(default_factory=dict)
    camera_jitter: dict = field(default_factory=dict)
    physics: dict = field(default_factory=dict)   # applied env-side at reset


@dataclass(frozen=True)
class EncoderSpec:
    kind: Literal["none", "frozen_cnn"] = "none"
    checkpoint: Optional[str] = None
    output_dim: Optional[int] = None
    layer: Optional[str] = None
    out_key: str = "encoded"


@dataclass(frozen=True)
class PolicySpec:
    actor_keys: tuple[str, ...]
    critic_keys: tuple[str, ...]
    cnn: Optional[dict] = None               # None => pure vector policy
    mlp: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ActionDRSpec:
    steer_noise: float = 0.0
    speed_noise: float = 0.0
    delay_steps: int = 0


@dataclass(frozen=True)
class AlgorithmSpec:
    kind: Literal["ppo", "ppo_lagrangian"] = "ppo"
    ppo: dict = field(default_factory=dict)
    lagrangian: dict = field(default_factory=dict)  # budget, pid=(kp,ki,kd), ...


@dataclass(frozen=True)
class ExperimentSpec:
    env: Optional[EnvSpec] = None
    obs_dr: ObsDRSpec = field(default_factory=ObsDRSpec)
    encoder: EncoderSpec = field(default_factory=EncoderSpec)
    policy: Optional[PolicySpec] = None
    action_dr: ActionDRSpec = field(default_factory=ActionDRSpec)
    algorithm: Optional[AlgorithmSpec] = None
    total_env_steps: int = 5_000_000
    seed: int = 0
    ablation_group: Optional[str] = None
    variant: Optional[str] = None

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """One-way dump (tuples normalized to lists) for hashing + records."""
        return json.loads(json.dumps(asdict(self)))

    def id(self) -> str:
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
        keys = ["state"]
        if self.env is not None and self.env.modality == "camera":
            keys.append("camera")
        if self.encoder.kind != "none":
            keys.append(self.encoder.out_key)
        return tuple(keys)

    def validate(self) -> "ExperimentSpec":
        env, policy = self.env, self.policy
        if env is None:
            raise SpecError("pipeline must start with an Environment stage")
        if policy is None:
            raise SpecError("pipeline must include exactly one Policy stage")

        # --- environment coherence ---
        if env.modality == "feature" and env.render != "none":
            raise SpecError("feature envs do not render; got render=%r" % env.render)
        if env.modality == "camera" and env.render not in ("madrona", "nyx"):
            raise SpecError("camera envs need render='madrona'|'nyx'; got %r" % env.render)
        if env.render == "nyx" and len(env.tracks) > 1:
            raise SpecError(
                "heterogeneous tracks are Madrona-only (repo constraint); "
                "render='nyx' with tracks=%r" % (env.tracks,))
        if env.emits_cost:
            if env.cost_fn not in VALID_COST_FNS:
                raise SpecError("cost_fn must be one of %s; got %r" % (VALID_COST_FNS, env.cost_fn))
            if env.cost_budget is None or env.cost_budget <= 0:
                raise SpecError("cost-emitting env needs a positive cost_budget")

        # --- key routing ---
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
        if not a_keys <= c_keys:
            raise SpecError("asymmetric policies require critic_keys ⊇ actor_keys; "
                            "actor has %s the critic lacks" % sorted(a_keys - c_keys))
        if policy.cnn is None and "camera" in (a_keys | c_keys):
            raise SpecError("a vector policy cannot consume the raw 'camera' key; "
                            "add an encoder stage or use a camera policy")
        if policy.cnn is not None and "camera" not in a_keys:
            raise SpecError("a camera policy's actor must read the 'camera' key")

        # --- encoder ---
        if self.encoder.kind == "frozen_cnn":
            if env.modality != "camera":
                raise SpecError("FrozenCNNToFeatureVector requires an upstream camera env")
            if self.encoder.checkpoint is None:
                raise SpecError("FrozenCNNToFeatureVector requires a checkpoint path")
            if policy.cnn is not None:
                raise SpecError("FrozenCNNToFeatureVector requires a downstream vector "
                                "policy (VectorPolicy/AsymmetricVectorPolicy)")

        # --- obs DR coherence ---
        if (self.obs_dr.image_aug or self.obs_dr.camera_jitter) and env.modality != "camera":
            raise SpecError("DomainRandomizationCamera requires a camera env")

        # --- action DR ---
        if self.action_dr.delay_steps < 0:
            raise SpecError("delay_steps must be >= 0")
        if self.action_dr.steer_noise < 0 or self.action_dr.speed_noise < 0:
            raise SpecError("action noise magnitudes must be >= 0")

        # --- algorithm coherence ---
        algo = self.algorithm
        if algo is None:
            raise SpecError("algorithm missing: build() must run _infer_algorithm")
        if env.emits_cost and algo.kind == "ppo":
            warnings.warn(
                "cost-emitting env trained with plain PPO: the cost stream is "
                "collected but unconstrained (was this intentional?)",
                stacklevel=2)
        if not env.emits_cost and algo.kind == "ppo_lagrangian":
            raise SpecError("PPOLagrangian requires a SafeRL* env that emits a cost signal")
        if algo.kind == "ppo_lagrangian" and not algo.lagrangian.get("budget"):
            raise SpecError("PPOLagrangian needs a budget (explicit or from the env stage)")
        if (algo.kind == "ppo_lagrangian" and env.cost_budget is not None
                and algo.lagrangian.get("budget") not in (None, env.cost_budget)):
            raise SpecError(
                "conflicting budgets: env.cost_budget=%r vs algorithm.lagrangian"
                "['budget']=%r — sweep 'env.cost_budget' (ablation.override keeps "
                "them in sync)" % (env.cost_budget, algo.lagrangian.get("budget")))

        return self
