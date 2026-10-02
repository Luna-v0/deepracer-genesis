"""The terminal crash penalty closes the per-term breakdown (P11) — GPU-free.

Env 0 drives clean, env 1 is off-track, env 2 is flipped, env 3 timed out.
"""

import pytest
import torch

from deepracer_genesis.envs import mdp
from deepracer_genesis.experiment.builder import Builder
from deepracer_genesis.experiment.spec import (
    AlgorithmSpec,
    EnvSpec,
    ExperimentSpec,
    PolicySpec,
)
from deepracer_genesis.experiment.stages import RewardShaping

N = 4
PENALTY = -10.0


class _StubEnv:
    """Exposes what the non-cost ``check_termination`` path reads/writes."""

    def __init__(self):
        self.cfg = {"termination": {"off_track_margin": 0.10,
                                    "crash_penalty": PENALTY}}
        self.emit_cost = False
        self.lateral = torch.tensor([0.0, 1.0, 0.0, 0.0])
        self.half_width = torch.full((N,), 0.35)
        self.up_z = torch.tensor([1.0, 1.0, -1.0, 1.0])
        self.episode_length_buf = torch.tensor([10, 10, 10, 1500])
        self.max_episode_length = 1500
        self.rew_buf = torch.zeros(N)
        self.episode_sums = {}
        self.state_buf = torch.zeros(N, 3)
        # compute_reward inputs for the closes-the-books test
        self.reward_terms = lambda e: {"progress": torch.ones(N)}
        self.reward_scales = {"progress": 2.0}
        self.reward_params = {}


def test_penalty_reaches_rew_buf_and_episode_sums():
    env = _StubEnv()
    mdp.check_termination(env)
    assert torch.equal(env.rew_buf, torch.tensor([0.0, PENALTY, PENALTY, 0.0]))
    assert torch.equal(env.episode_sums["crash_penalty"], env.rew_buf)
    assert env.time_out_buf.tolist() == [False, False, False, True]  # no penalty


def test_breakdown_sums_to_the_reward_the_learner_saw():
    env = _StubEnv()
    env.episode_sums = {"progress": torch.zeros(N)}
    total = torch.zeros(N)
    for _ in range(3):
        mdp.compute_reward(env)
        mdp.check_termination(env)
        total += env.rew_buf
    summed = sum(env.episode_sums.values())
    assert torch.equal(summed, total), "per-term sums != effective reward"


def test_crash_penalty_is_a_reserved_term_name():
    env = _StubEnv()
    env.reward_scales = {"crash_penalty": 1.0}
    env.reward_terms = lambda e: {"crash_penalty": torch.ones(N)}
    with pytest.raises(ValueError, match="reserved"):
        mdp.compute_reward(env)


# ----------------------------------------------------------------- spec knob
def _spec(**env_kw):
    return ExperimentSpec(
        env=EnvSpec(modality="feature", **env_kw),
        policy=PolicySpec(actor_keys=("state",), critic_keys=("state",)),
        algorithm=AlgorithmSpec())


def test_reward_shaping_sets_crash_penalty_and_none_leaves_it_alone():
    spec = RewardShaping(crash_penalty=-3.0).apply(_spec())
    assert spec.env.crash_penalty == -3.0
    kept = RewardShaping(scales={"progress": 1.0}).apply(spec)
    assert kept.env.crash_penalty == -3.0   # None does not clobber


def test_crash_penalty_is_hashed_only_when_set_and_reaches_the_cfg():
    assert _spec().id() != _spec(crash_penalty=-3.0).id()
    cfg = Builder(_spec(crash_penalty=-3.0)).sim_cfg()
    assert cfg["termination"]["crash_penalty"] == -3.0
    assert Builder(_spec()).sim_cfg()["termination"]["crash_penalty"] == PENALTY
