"""Sign conventions of the built-in ``deepracer`` reward terms.

Pure-torch/CPU over a stub env; scales are positive, so penalties carry a minus.
"""

import torch

from deepracer_genesis.configs.cfgs import get_env_cfg
from deepracer_genesis.envs.mdp import compute_reward
from deepracer_genesis.envs.rewards import deepracer

DT = 0.02                # sim dt 0.01 x decimation 2 (configs/cfgs.py)
HALF_WIDTH = 0.6
WHEEL_MARGIN = 0.1       # a wheel leaves the road at |lateral| >= 0.5
SCALES = get_env_cfg()["reward"]["reward_scales"]


class _StubEnv:
    """Minimal stand-in exposing exactly what ``rewards.deepracer`` reads.

    Rows are identical apart from ``lateral``, isolating the on/off-track split.

    Args:
        lateral: ``(N,)`` signed distance from the centerline, in meters.
    """

    def __init__(self, lateral: torch.Tensor) -> None:
        n = lateral.shape[0]
        self.dt = DT
        self.lateral = lateral
        self.half_width = torch.full((n,), HALF_WIDTH)
        self.d_progress = torch.full((n,), 0.05)
        self.v_forward = torch.full((n,), 1.5)
        self.heading_err = torch.full((n,), 0.2)
        self.actions = torch.tensor([[0.3, 0.4]]).expand(n, 2).contiguous()
        self.last_actions = torch.zeros(n, 2)
        self.cfg = {
            "action": {"max_speed": 4.0},
            "termination": {"wheel_margin": WHEEL_MARGIN},
        }
        # the mdp.compute_reward accumulation surface
        self.reward_terms = deepracer
        self.reward_scales = SCALES
        self.rew_buf = torch.zeros(n)
        self.episode_sums = {name: torch.zeros(n) for name in SCALES}


def test_off_track_term_is_a_penalty():
    """Zero while all wheels are on the road, ``-dt`` once one leaves it."""
    env = _StubEnv(torch.tensor([0.0, 0.49, 0.51, -0.55]))
    off_track = deepracer(env)["off_track"]
    assert torch.allclose(off_track, torch.tensor([0.0, 0.0, -DT, -DT]))


def test_going_off_track_lowers_the_weighted_reward():
    """Scales are positive, so a positive term would pay a car for going off."""
    env = _StubEnv(torch.tensor([0.49, 0.51]))   # just inside / just outside
    compute_reward(env)
    on_track, off_track = env.rew_buf[0], env.rew_buf[1]
    assert off_track < on_track
    # the gap is the off_track penalty plus a smaller `centered` drift
    assert (on_track - off_track) > SCALES["off_track"] * DT


def test_penalty_terms_keep_their_sign():
    """heading/steering/action_rate are penalties too — a flip must not pass."""
    env = _StubEnv(torch.tensor([0.0, 0.49, 0.51, -0.55]))
    terms = deepracer(env)
    for name in ("heading", "steering", "action_rate", "off_track"):
        assert (terms[name] <= 0).all(), f"{name} must never be a bonus"
    # the stub drives all three with nonzero inputs, so none may be dead
    for name in ("heading", "steering", "action_rate"):
        assert (terms[name] < 0).all(), f"{name} stopped penalizing"
