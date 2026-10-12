"""Ackermann steering geometry (physics/limits.ackermann_angles).

Pure-torch on CPU — no Genesis. Expected magnitudes were verified independently
against the URDF geometry (L=0.163974 m, t=0.159202 m): a +30° center command
gives left(inner)=38.7359°, right(outer)=24.2734°.
"""

import math

import torch

from deepracer_genesis.physics.limits import (
    FRONT_TRACK_M,
    MAX_STEER_RAD,
    WHEELBASE_M,
    ackermann_angles,
)


def test_straight_is_zero():
    left, right = ackermann_angles(torch.zeros(4, 1))
    assert torch.allclose(left, torch.zeros_like(left))
    assert torch.allclose(right, torch.zeros_like(right))


def test_left_turn_inner_is_left_and_steers_more():
    left, right = ackermann_angles(torch.full((1, 1), MAX_STEER_RAD))  # +30° = left
    assert left.item() > right.item() > 0                              # left is inner
    assert math.isclose(math.degrees(left.item()), 38.7359, abs_tol=1e-2)
    assert math.isclose(math.degrees(right.item()), 24.2734, abs_tol=1e-2)


def test_right_turn_is_mirror_image():
    left, right = ackermann_angles(torch.full((1, 1), -MAX_STEER_RAD))  # -30° = right
    assert right.item() < left.item() < 0                               # right is inner
    assert math.isclose(math.degrees(right.item()), -38.7359, abs_tol=1e-2)
    assert math.isclose(math.degrees(left.item()), -24.2734, abs_tol=1e-2)


def test_ackermann_condition():
    # cot(outer) - cot(inner) == t / L, for a left turn (left = inner)
    left, right = ackermann_angles(torch.full((1, 1), math.radians(20.0)))
    cot = lambda a: 1.0 / math.tan(a)
    assert math.isclose(cot(right.item()) - cot(left.item()),
                        FRONT_TRACK_M / WHEELBASE_M, abs_tol=1e-4)


def test_finite_over_full_action_range():
    delta = torch.linspace(-MAX_STEER_RAD, MAX_STEER_RAD, 201).unsqueeze(1)
    left, right = ackermann_angles(delta)
    assert torch.isfinite(left).all() and torch.isfinite(right).all()


def test_preserves_batch_shape():
    delta = torch.randn(8, 1) * 0.1
    left, right = ackermann_angles(delta)
    assert left.shape == (8, 1) and right.shape == (8, 1)
