"""K.3: opt-in per-signal actor/critic obs routing.

Two layers, both CPU-only (no Genesis sim):

1. **Spec layer** — declaring ``obs_routing`` exposes ``obs_actor``/``obs_critic``
   keys, validates the base/actor/critic block selections + the privileged-critic
   invariant, and feeds the K.5 learnability check per-signal. The default
   (``obs_routing=None``) path is unchanged (single ``state`` key).
2. **Assembly layer** — ``RoutedFeatures`` builds the two vectors from the shared
   ``FEATURE_BLOCKS``, so ``critic=["*"]`` reproduces the ``SelectFeatures`` vector
   and the actor is a strict, narrower view.
"""

import json
import warnings

import pytest
import torch

from deepracer_genesis.envs.features import (
    FEATURE_BLOCKS,
    RoutedFeatures,
    SelectFeatures,
)
from deepracer_genesis.envs.signals import SignalBus
from deepracer_genesis.experiment import (
    AsymmetricVectorPolicy,
    FeatureEnvironment,
    VectorPolicy,
)
from deepracer_genesis.experiment.spec import (
    AlgorithmSpec,
    EnvSpec,
    ExperimentSpec,
    PolicySpec,
    SpecError,
)

BASE = ("v_forward", "v_lateral", "yaw_rate", "lateral", "heading", "last_action")
ROUTING = {"base": BASE, "actor": ("v_forward", "lateral", "heading"), "critic": ("*",)}


def _routed_pipeline(routing=ROUTING):
    return (FeatureEnvironment(num_envs=8, obs_routing=routing)
            >> AsymmetricVectorPolicy(actor_keys=("obs_actor",),
                                      critic_keys=("obs_critic",)))


def _spec(env, policy):
    return ExperimentSpec(env=env, policy=policy, algorithm=AlgorithmSpec())


# ---------------------------------------------------------------- spec layer
def test_default_obs_routing_is_none_single_state_key():
    spec = (FeatureEnvironment(num_envs=8) >> VectorPolicy()).build()
    assert spec.env.obs_routing is None
    assert spec.available_keys() == ("state",)
    assert spec.to_dict()["env"]["obs_routing"] is None


def test_routed_exposes_obs_actor_and_critic_keys():
    spec = _routed_pipeline().build()
    keys = spec.available_keys()
    assert "obs_actor" in keys and "obs_critic" in keys and "state" not in keys


def test_routed_spec_serializes_and_id_is_stable():
    spec = _routed_pipeline().build(seed=0)
    d = spec.to_dict()
    assert d["env"]["obs_routing"] is not None
    json.dumps(d)                                   # must stay JSON
    assert _routed_pipeline().build(seed=0).id() == spec.id()


def test_routing_rejected_on_camera_env():
    env = EnvSpec(modality="camera", render="madrona", obs_routing=ROUTING)
    policy = PolicySpec(actor_keys=("obs_actor",), critic_keys=("obs_critic",))
    with pytest.raises(SpecError, match="feature env"):
        _spec(env, policy).validate()


def test_routing_rejects_unknown_base_block():
    env = EnvSpec(modality="feature",
                  obs_routing={"base": ("nope",), "actor": ("*",), "critic": ("*",)})
    policy = PolicySpec(actor_keys=("obs_actor",), critic_keys=("obs_critic",))
    with pytest.raises(SpecError, match="unknown block"):
        _spec(env, policy).validate()


def test_routing_rejects_actor_block_not_in_base():
    env = EnvSpec(modality="feature",
                  obs_routing={"base": ("v_forward",), "actor": ("lateral",),
                               "critic": ("*",)})
    policy = PolicySpec(actor_keys=("obs_actor",), critic_keys=("obs_critic",))
    with pytest.raises(SpecError, match="not in base"):
        _spec(env, policy).validate()


def test_routing_requires_privileged_critic():
    """The critic must see >= the actor's blocks (obs_critic supersets obs_actor)."""
    env = EnvSpec(modality="feature",
                  obs_routing={"base": BASE, "actor": ("v_forward", "yaw_rate"),
                               "critic": ("v_forward",)})
    policy = PolicySpec(actor_keys=("obs_actor",), critic_keys=("obs_critic",))
    with pytest.raises(SpecError, match="privileged"):
        _spec(env, policy).validate()


def test_signals_for_keys_routed():
    spec = _routed_pipeline().build()
    assert spec._signals_for_keys({"obs_critic"}) == {"*"}          # critic=["*"]
    assert spec._signals_for_keys({"obs_actor"}) == {
        "v_forward", "lateral", "half_width", "heading_err"}         # actor blocks


def test_routed_star_critic_no_learnability_warning():
    """critic=["*"] recovers every signal -> the default reward is learnable."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _routed_pipeline().build()


def test_routed_explicit_critic_warns_unlearnable():
    """A critic that carries only kinematic blocks cannot recover the default
    reward's d_progress/off_track -> a precise 'unlearnable' warning."""
    routing = {"base": BASE, "actor": ("v_forward",),
               "critic": ("v_forward", "lateral", "heading")}
    pipe = (FeatureEnvironment(num_envs=8, obs_routing=routing)
            >> AsymmetricVectorPolicy(actor_keys=("obs_actor",),
                                      critic_keys=("obs_critic",)))
    with pytest.warns(UserWarning, match="unlearnable"):
        pipe.build()


# ------------------------------------------------------------- assembly layer
class _FakeEnv:
    """Minimal env carrying what the (non-lookahead) blocks read + a bus."""

    def __init__(self, with_bus=True):
        n = 4
        self.num_envs = n
        self.device = torch.device("cpu")
        self.lookahead_k = 3
        self.v_forward = torch.tensor([0.5, 1.0, 1.5, 2.0])
        self.v_lateral = torch.tensor([0.0, -0.1, 0.2, -0.3])
        self.yaw_rate = torch.tensor([0.0, 0.5, -0.5, 1.0])
        self.up_z = torch.ones(n)
        self.lateral = torch.tensor([0.0, 0.2, -0.4, 0.5])
        self.half_width = torch.full((n,), 0.6)
        self.heading_err = torch.tensor([0.0, 0.3, -0.3, 0.1])
        self.d_progress = torch.tensor([0.01, 0.02, 0.03, 0.04])
        self.actions = torch.tensor([[0.1, 0.2], [-0.1, 0.3], [0.0, -0.2], [0.2, 0.1]])
        self.last_actions = torch.zeros(n, 2)
        self.dir_sign = torch.tensor([1.0, 1.0, -1.0, -1.0])
        self.cfg = {"action": {"max_speed": 4.0},
                    "termination": {"wheel_margin": 0.0}}
        self.signals = SignalBus(self) if with_bus else None


def test_routed_critic_star_equals_select_over_base():
    env = _FakeEnv()
    routed = RoutedFeatures(env, {"base": BASE, "actor": ("v_forward", "lateral"),
                                  "critic": ("*",)})
    select = SelectFeatures(env, {"features": BASE})
    assert torch.equal(routed.compute_critic(), select.compute())
    assert routed.critic_dim == select.compute().shape[1]


def test_routed_actor_is_a_narrower_view():
    env = _FakeEnv()
    routed = RoutedFeatures(env, {"base": BASE,
                                  "actor": ("v_forward", "lateral", "heading"),
                                  "critic": ("*",)})
    assert routed.actor_dim == 4                     # 1 + 1 + 2
    assert routed.compute_actor().shape == (env.num_envs, 4)
    assert routed.actor_dim < routed.critic_dim


def test_routed_selection_resolves_to_base_order():
    env = _FakeEnv()
    r = RoutedFeatures(env, {"base": BASE, "actor": ("heading", "v_forward"),
                             "critic": ("*",)})
    assert r._actor == ("v_forward", "heading")      # base order, not selection order
    manual = torch.cat([FEATURE_BLOCKS["v_forward"].compute(env),
                        FEATURE_BLOCKS["heading"].compute(env)], dim=1)
    assert torch.equal(r.compute_actor(), manual)
