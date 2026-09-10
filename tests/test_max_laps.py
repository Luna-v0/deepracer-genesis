"""Lap-quota episode end (`max_laps`): a bootstrapped truncation, GPU-free.

Env 0 is mid-lap, env 1 has met the quota, env 2 exceeded it.
"""

import pytest
import torch

from deepracer_genesis.envs import mdp
from deepracer_genesis.experiment.builder import Builder
from deepracer_genesis.experiment.spec import (
    AlgorithmSpec,
    ExperimentSpec,
    PolicySpec,
    SpecError,
)
from deepracer_genesis.experiment.stages import (
    CameraEnvironment,
    FeatureEnvironment,
    SafeRLCameraEnvironment,
    VectorPolicy,
)

N = 3


class _StubEnv:
    """Exposes what ``check_termination`` reads/writes; all cars on-track."""

    def __init__(self, max_laps, emit_cost=False):
        self.cfg = {"termination": {"off_track_margin": 0.10,
                                    "crash_penalty": -10.0,
                                    "max_laps": max_laps}}
        self.emit_cost = emit_cost
        self.cost_fn = "offtrack"
        self.lateral = torch.zeros(N)
        self.half_width = torch.full((N,), 0.35)
        self.up_z = torch.ones(N)
        self.laps = torch.tensor([0.4, 2.0, 2.3])
        self.episode_length_buf = torch.full((N,), 10)
        self.max_episode_length = 1500
        self.rew_buf = torch.zeros(N)
        self.episode_sums = {}
        self.state_buf = torch.zeros(N, 3)
        self.cost_buf = torch.zeros(N)
        self.cost_episode_sum = torch.zeros(N)


def test_lap_quota_truncates_without_penalty():
    env = _StubEnv(max_laps=2)
    mdp.check_termination(env)
    assert env.reset_buf.tolist() == [False, True, True]
    # a truncation, not a failure: bootstrapped via time_outs, no -10
    assert env.time_out_buf.tolist() == [False, True, True]
    assert torch.equal(env.rew_buf, torch.zeros(N))
    assert torch.equal(env.episode_sums["crash_penalty"], torch.zeros(N))


def test_none_keeps_episodes_endless():
    env = _StubEnv(max_laps=None)
    mdp.check_termination(env)
    assert not env.reset_buf.any()


def test_quota_applies_on_the_cost_path_too():
    env = _StubEnv(max_laps=2, emit_cost=True)
    mdp.check_termination(env)
    assert env.reset_buf.tolist() == [False, True, True]


# ----------------------------------------------------------------- spec knob
def _spec(**env_kw):
    spec = ExperimentSpec(policy=PolicySpec(actor_keys=("state",),
                                            critic_keys=("state",)),
                          algorithm=AlgorithmSpec())
    return FeatureEnvironment(**env_kw).apply(spec)


def test_max_laps_threads_from_stage_to_cfg_and_is_hash_pruned_when_unset():
    cfg = Builder(_spec(max_laps=3)).sim_cfg()
    assert cfg["termination"]["max_laps"] == 3
    assert Builder(_spec()).sim_cfg()["termination"]["max_laps"] is None
    assert _spec().env.max_laps is None
    assert _spec().id() != _spec(max_laps=3).id()   # unset default pinned by golden ids


@pytest.mark.parametrize("bad", [0, -1, 1.5])
def test_max_laps_must_be_a_positive_int(bad):
    with pytest.raises(SpecError, match="max_laps"):
        _spec(max_laps=bad).validate()


def test_max_laps_reaches_camera_and_safe_rl_envs_too():
    # the check itself lives in check_termination (modality-agnostic); this
    # pins that every environment stage actually threads the knob
    base = ExperimentSpec(
        policy=PolicySpec(actor_keys=("camera",),
                          critic_keys=("camera", "state"),
                          cnn={"channels": (32,)}),
        algorithm=AlgorithmSpec())
    cam = CameraEnvironment(max_laps=2).apply(base)
    assert cam.env.max_laps == 2
    assert Builder(cam).sim_cfg()["termination"]["max_laps"] == 2
    assert SafeRLCameraEnvironment(max_laps=2).apply(base).env.max_laps == 2
