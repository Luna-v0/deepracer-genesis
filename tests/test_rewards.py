"""Sign conventions of the built-in reward terms (GPU-free stub env).

The convention in ``rewards.deepracer`` is that SIGNS LIVE IN THE TERMS and
``reward_scales`` stay positive. A 2026-08 bug had the ``off_track`` term
positive, so the default +2.0 scale *rewarded* edge-riding at 4x the
centered bonus — these tests pin every term's sign so that cannot recur.
"""

import torch

from deepracer_genesis.envs.rewards import deepracer


class _StubEnv:
    """Minimal env double exposing the tensors ``rewards.deepracer`` reads.

    Attributes:
        Everything is (2,)-shaped: car 0 drives centered and clean, car 1
        rides beyond the wheel margin with steering/heading activity.
    """

    def __init__(self):
        self.dt = 0.02
        self.cfg = {"termination": {"wheel_margin": 0.08},
                    "action": {"max_speed": 3.0}}
        self.lateral = torch.tensor([0.0, 0.30])
        self.half_width = torch.tensor([0.35, 0.35])
        self.d_progress = torch.tensor([0.05, 0.05])
        self.v_forward = torch.tensor([1.5, 1.5])
        self.heading_err = torch.tensor([0.0, 0.4])
        self.actions = torch.tensor([[0.0, 0.5], [0.6, 0.5]])
        self.last_actions = torch.zeros(2, 2)


def test_offtrack_term_is_a_penalty():
    """Edge-riding must be strictly worse than staying inside the margin."""
    terms = deepracer(_StubEnv())
    assert terms["off_track"][0] == 0.0          # centered car: no penalty
    assert terms["off_track"][1] < 0.0           # edge-rider: negative term


def test_penalty_terms_are_nonpositive_and_bonuses_nonnegative():
    """Positive scales must never flip a term's intent."""
    terms = deepracer(_StubEnv())
    for name in ("heading", "steering", "action_rate", "off_track"):
        assert (terms[name] <= 0).all(), f"{name} must be a penalty term"
    for name in ("progress", "speed", "centered"):
        assert (terms[name] >= 0).all(), f"{name} must be a bonus term"
