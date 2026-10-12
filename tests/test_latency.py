"""Stateful temporal camera DR (randomization.latency.FrameLatency)."""

import torch

from deepracer_genesis.randomization.latency import FrameLatency


def _frame(val, n=2, c=3, h=4, w=4):
    return torch.full((n, c, h, w), float(val))


def test_zero_latency_zero_drop_is_passthrough():
    fl = FrameLatency(2, latency=0, drop=0.0, device="cpu")
    for v in (0.1, 0.2, 0.3):
        assert torch.equal(fl.advance(_frame(v)), _frame(v))


def test_latency_one_delays_by_a_step():
    fl = FrameLatency(2, latency=1, drop=0.0, device="cpu")
    assert torch.equal(fl.advance(_frame(1)), _frame(1))   # seed shows current
    assert torch.equal(fl.advance(_frame(2)), _frame(1))   # then 1 step behind
    assert torch.equal(fl.advance(_frame(3)), _frame(2))


def test_latency_two_delays_by_two_steps():
    fl = FrameLatency(2, latency=2, drop=0.0, device="cpu")
    emitted = [fl.advance(_frame(v)) for v in (1, 2, 3, 4)]
    vals = [e[0, 0, 0, 0].item() for e in emitted]
    assert vals == [1.0, 1.0, 1.0, 2.0]   # frame from 2 steps ago after seed


def test_reset_clears_cross_episode_bleed():
    fl = FrameLatency(2, latency=1, drop=0.0, device="cpu")
    fl.advance(_frame(1))
    fl.reset(torch.tensor([0]))            # env 0 respawns
    out = fl.advance(_frame(2))
    assert out[0, 0, 0, 0].item() == 2.0   # reset env sees its fresh frame
    assert out[1, 0, 0, 0].item() == 1.0   # env 1 still 1 step behind


def test_reseed_restarts_only_the_given_envs():
    """A respawn re-render mid-step must not advance the other envs again."""
    fl = FrameLatency(2, latency=1, drop=0.0, device="cpu")
    fl.advance(_frame(1))
    fl.reseed(torch.tensor([0]), _frame(2))
    out = fl.advance(_frame(3))
    assert out[0, 0, 0, 0].item() == 2.0   # one step behind its fresh frame
    assert out[1, 0, 0, 0].item() == 1.0   # one step behind, not two


def test_frame_drop_repeats_previous():
    torch.manual_seed(0)
    fl = FrameLatency(2, latency=0, drop=1.0, device="cpu")   # always drop
    assert torch.equal(fl.advance(_frame(1)), _frame(1))      # first can't drop
    assert torch.equal(fl.advance(_frame(2)), _frame(1))      # repeats previous
    assert torch.equal(fl.advance(_frame(3)), _frame(1))


def test_shape_preserved():
    fl = FrameLatency(5, latency=1, drop=0.05, device="cpu")
    f = torch.rand(5, 3, 8, 12)
    assert fl.advance(f).shape == f.shape
