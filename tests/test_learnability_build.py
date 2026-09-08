"""Build-time learnability check wiring (Part K.4/K.5).

`spec.validate()` warns (never raises) when the reward/cost read signals the
critic can't see, or lean on a non-pixel-observable signal a pixel-only actor
cannot carry. These are pure spec-level tests (no sim): they construct
`ExperimentSpec` dataclasses directly and assert the warning behavior.
"""

import warnings

import pytest

from deepracer_genesis.envs.rewards import (
    COST_READS,
    cost_reads,
    deepracer,
    reads,
    reward_reads,
)
from deepracer_genesis.experiment.spec import (
    AlgorithmSpec,
    EnvSpec,
    ExperimentSpec,
    PolicySpec,
)


# ----------------------------------------------------------- reads metadata
def test_default_reward_declares_reads():
    r = reward_reads(deepracer)
    assert r == frozenset({
        "d_progress", "v_forward", "lateral", "half_width", "heading_err",
        "actions", "action_rate", "off_track", "up_z"})


def test_undeclared_custom_reward_reads_empty():
    def custom(env):
        return {}
    assert reward_reads(custom) == frozenset()


def test_reads_decorator_attaches_frozenset_without_changing_call():
    @reads("v_forward", "off_track")
    def r(env):
        return {"x": env}
    assert r.reads == frozenset({"v_forward", "off_track"})
    assert r("sentinel") == {"x": "sentinel"}   # behavior unchanged


def test_cost_reads_mapping():
    assert cost_reads("offtrack") == COST_READS["offtrack"]
    assert "v_forward" in cost_reads("offtrack_or_overspeed")
    assert cost_reads(None) == frozenset()
    assert cost_reads("nonexistent") == frozenset()


# --------------------------------------------------- spec.validate() wiring
def _feature_policy(actor=("state",), critic=("state",)):
    return PolicySpec(actor_keys=actor, critic_keys=critic)


def _spec(env, policy):
    return ExperimentSpec(env=env, policy=policy, algorithm=AlgorithmSpec())


def test_default_reward_default_config_does_not_warn():
    """The default deepracer reward on the default feature config: no warning."""
    spec = _spec(EnvSpec(modality="feature"), _feature_policy())
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # any warning fails the test
        spec.validate()


def test_reward_reading_signal_hidden_from_critic_warns():
    """A reward whose declared reads are not critic-visible triggers a warning.

    A pixel-only critic (`critic_keys=("camera",)`) recovers only the
    pixel-observable signals; a reward reading `d_progress` (not pixel-
    observable) is therefore unlearnable.
    """
    @reads("d_progress", "lateral")
    def progress_reward(env):
        return {"progress": env.d_progress}

    env = EnvSpec(modality="camera", render="madrona", reward=progress_reward,
                  reward_scales={"progress": 1.0})
    # pixel-only actor AND critic -> critic cannot see d_progress
    policy = PolicySpec(actor_keys=("camera",), critic_keys=("camera",),
                        cnn={"channels": (32,)})
    with pytest.warns(UserWarning, match="unlearnable.*d_progress"):
        _spec(env, policy).validate()


def test_privileged_critic_asymmetry_is_silent_at_build():
    """The intended privileged-critic pattern must NOT warn at build.

    Asymmetric camera policy: the critic sees ``state`` (all signals) while the
    pixel-only actor cannot carry the non-pixel ``d_progress``. This is the
    recommended asymmetric design (the critic can still score the reward), so
    ``validate()`` stays silent — the soft "cannot act on it" advisory is not
    emitted at build (it remains testable directly via ``check_learnability``,
    see ``test_signals.py``)."""
    @reads("d_progress")
    def progress_reward(env):
        return {"progress": env.d_progress}

    env = EnvSpec(modality="camera", render="madrona", reward=progress_reward,
                  reward_scales={"progress": 1.0})
    policy = PolicySpec(actor_keys=("camera",), critic_keys=("camera", "state"),
                        cnn={"channels": (32,)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # any warning (incl. the soft one) fails
        _spec(env, policy).validate()


def test_undeclared_custom_reward_warns_that_check_is_skipped():
    """A custom reward with no declared reads warns loudly (P12.d): an opaque
    reward is exactly the case K.5 should check, so skipping is never silent."""
    def custom(env):
        return {"x": env.d_progress}

    env = EnvSpec(modality="camera", render="madrona", reward=custom,
                  reward_scales={"x": 1.0})
    policy = PolicySpec(actor_keys=("camera",), critic_keys=("camera",),
                        cnn={"channels": (32,)})
    with pytest.warns(UserWarning, match="declares no signal reads"):
        _spec(env, policy).validate()


def test_cost_reads_checked_against_pixel_only_critic():
    """An emitted cost's reads are checked too: a pixel-only critic can't see
    the cost's non-pixel signals (v_forward) -> unlearnable warning."""
    env = EnvSpec(modality="camera", render="madrona", emits_cost=True,
                  cost_fn="offtrack_or_overspeed", cost_budget=25.0)
    policy = PolicySpec(actor_keys=("camera",), critic_keys=("camera",),
                        cnn={"channels": (32,)})
    algo = AlgorithmSpec(lagrangian={"budget": 25.0})
    with pytest.warns(UserWarning, match="unlearnable.*v_forward"):
        ExperimentSpec(env=env, policy=policy, algorithm=algo).validate()
