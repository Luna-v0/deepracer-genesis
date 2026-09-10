"""First-class custom rewards (P12): verbatim tensors, ``total``, ``_`` diagnostics,
and spec-hashed ``reward_params`` — all GPU-free (stub env / pure spec objects).
"""

import hashlib
import json

import pytest
import torch

from deepracer_genesis.envs import mdp
from deepracer_genesis.experiment.spec import (
    AlgorithmSpec,
    EnvSpec,
    ExperimentSpec,
    PolicySpec,
)
from deepracer_genesis.experiment.stages import RewardShaping

N = 4


class _StubEnv:
    """Minimal env double exposing what ``mdp.compute_reward`` reads/writes."""

    def __init__(self, fn, scales=None, params=None):
        self.reward_terms = fn
        self.reward_scales = dict(scales or {})
        self.reward_params = dict(params or {})
        self.rew_buf = torch.zeros(N)
        self.episode_sums = {k: torch.zeros(N) for k in self.reward_scales}
        self.d_progress = torch.arange(N, dtype=torch.float32)


# ----------------------------------------------------- verbatim (monolithic)
def test_bare_tensor_is_the_reward_verbatim():
    env = _StubEnv(lambda e: e.d_progress * 3.0)
    mdp.compute_reward(env)
    assert torch.equal(env.rew_buf, env.d_progress * 3.0)
    assert torch.equal(env.episode_sums["total"], env.rew_buf)
    mdp.compute_reward(env)   # episode sums accumulate across steps
    assert torch.equal(env.episode_sums["total"], env.d_progress * 6.0)


def test_bare_tensor_accepts_column_shape():
    env = _StubEnv(lambda e: e.d_progress.reshape(N, 1))
    mdp.compute_reward(env)
    assert torch.equal(env.rew_buf, env.d_progress)


def test_bare_tensor_with_scales_raises():
    env = _StubEnv(lambda e: e.d_progress, scales={"progress": 10.0})
    with pytest.raises(ValueError, match="bare tensor.*reward_scales"):
        mdp.compute_reward(env)


def test_total_term_without_scales_is_the_reward_verbatim():
    env = _StubEnv(lambda e: {"total": e.d_progress * 2.0, "_pace": -e.d_progress})
    mdp.compute_reward(env)
    assert torch.equal(env.rew_buf, env.d_progress * 2.0)
    assert torch.equal(env.episode_sums["total"], env.d_progress * 2.0)
    assert torch.equal(env.episode_sums["_pace"], -env.d_progress)


def test_total_with_stray_unscaled_term_raises():
    env = _StubEnv(lambda e: {"total": e.d_progress, "pace": e.d_progress})
    with pytest.raises(ValueError, match="unscaled term.*'pace'"):
        mdp.compute_reward(env)


def test_named_terms_without_scales_raises_with_guidance():
    def my_reward(e):
        return {"pace": e.d_progress}

    env = _StubEnv(my_reward)
    with pytest.raises(ValueError, match="my_reward.*no scales"):
        mdp.compute_reward(env)


# ------------------------------------------------------ diagnostic channels
def test_diagnostic_terms_are_logged_but_never_summed():
    env = _StubEnv(
        lambda e: {"progress": e.d_progress, "_lat_speed": e.d_progress * 5.0},
        scales={"progress": 2.0})
    mdp.compute_reward(env)
    assert torch.equal(env.rew_buf, env.d_progress * 2.0)   # _lat_speed excluded
    assert torch.equal(env.episode_sums["progress"], env.d_progress * 2.0)
    assert torch.equal(env.episode_sums["_lat_speed"], env.d_progress * 5.0)


def test_scale_referencing_diagnostic_term_raises():
    env = _StubEnv(lambda e: {"_x": e.d_progress}, scales={"_x": 1.0})
    with pytest.raises(ValueError, match="diagnostic term"):
        mdp.compute_reward(env)


def test_weighted_sum_path_unchanged():
    env = _StubEnv(
        lambda e: {"a": e.d_progress, "b": -e.d_progress},
        scales={"a": 10.0, "b": 0.5})
    mdp.compute_reward(env)
    assert torch.equal(env.rew_buf, env.d_progress * 10.0 - env.d_progress * 0.5)


# ------------------------------------------------------------ reward_params
def test_reward_params_reach_the_fn_via_env():
    env = _StubEnv(
        lambda e: -(e.d_progress - e.reward_params["target"]).abs(),
        params={"target": 1.6})
    mdp.compute_reward(env)
    assert torch.equal(env.rew_buf, -(env.d_progress - 1.6).abs())


def _spec(**env_kw):
    return ExperimentSpec(
        env=EnvSpec(modality="feature", **env_kw),
        policy=PolicySpec(actor_keys=("state",), critic_keys=("state",)),
        algorithm=AlgorithmSpec())


def test_reward_shaping_stage_threads_params_into_the_spec():
    def fn(e):
        return e.d_progress

    spec = RewardShaping(fn=fn, params={"target": 1.6}).apply(_spec())
    assert spec.env.reward is fn
    assert spec.env.reward_scales == {}
    assert spec.env.reward_params == {"target": 1.6}


def test_reward_params_are_content_hashed_and_recorded():
    base, with_params = _spec(), _spec(reward_params={"target": 1.6})
    assert base.id() != with_params.id()
    assert with_params.to_dict()["env"]["reward_params"] == {"target": 1.6}
    # differing params = different runs (the closure-constant blind spot)
    assert with_params.id() != _spec(reward_params={"target": 2.0}).id()


def test_empty_reward_params_keep_pre_p12_content_hashes():
    """Golden ids captured BEFORE the reward_params field existed: existing
    run dirs / eval records must keep resolving to the same specs."""
    assert _spec().id() == "24bb86d09cbc"
    camera = ExperimentSpec(
        env=EnvSpec(modality="camera", render="madrona",
                    reward_scales={"progress": 12.0}),
        policy=PolicySpec(actor_keys=("camera",), critic_keys=("camera", "state"),
                          cnn={"channels": (32,)}),
        algorithm=AlgorithmSpec())
    assert camera.id() == "074585c63896"
    # and the exemption is only for the UNSET defaults, not a hash exclusion
    payload = {k: v for k, v in _spec(reward_params={"t": 1}).to_dict().items()
               if k not in ("group", "variant")}
    # mirror id()'s prune list
    for late, unset in (("crash_penalty", None), ("episode_length_s", None),
                        ("max_laps", None)):
        if payload["env"].get(late) is unset:
            payload["env"].pop(late, None)
    manual = hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    assert _spec(reward_params={"t": 1}).id() == manual
