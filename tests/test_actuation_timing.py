"""The actuation-timing queue (randomization.actuation_timing), no sim (ADR 0005).

Latency is exact in whole 10 ms substeps, jitter is uniform, drops hold the
previous command, envs are independent, landings never reorder, a reset
touches only the given envs, and the same seed replays the same draws.
"""

import math
from itertools import pairwise

import pytest
import torch

from deepracer_genesis.randomization.actuation_timing import (
    ActuationTiming,
    max_pending,
)

DT = 0.01          # physics step: the latency resolution
DECIMATION = 2     # the 50 Hz control period


def _timing(n=2, latency=(30.0, 30.0), jitter=0.0, drop=0.0, seed=0,
            decimation=DECIMATION):
    t = ActuationTiming(n, latency_ms=latency, jitter_ms=jitter, drop=drop,
                        sim_dt=DT, decimation=decimation, device="cpu", seed=seed)
    t.reset(torch.arange(n))
    return t


def _cmd(value, n=2):
    return torch.full((n, 2), float(value))


def _first_substep_with(t, value, horizon=40):
    """Substep index (0 = the issue substep) at which ``value`` is first applied."""
    for k in range(horizon):
        if t.advance()[0, 0].item() == value:
            return k
    return None


# --------------------------------------------------------- latency exact in ms
@pytest.mark.parametrize("latency_ms", [0.0, 10.0, 30.0, 70.0, 110.0])
def test_latency_lands_after_ceil_latency_over_dt_substeps(latency_ms):
    t = _timing(latency=(latency_ms, latency_ms))
    t.issue(_cmd(1.0))
    assert _first_substep_with(t, 1.0) == math.ceil(latency_ms / (DT * 1000))


def test_a_latency_beyond_one_period_lands_in_the_right_later_step():
    """70 ms at a 20 ms period: the command lands 3 decisions later, substep 1."""
    t = _timing(latency=(70.0, 70.0))
    applied = []
    for step in range(5):
        t.issue(_cmd(step + 1))
        for _ in range(DECIMATION):
            applied.append(t.advance()[0, 0].item())
    # substeps 0..6 still the initial zero, substep 7 = command 1, 9 = command 2
    assert applied[:7] == [0.0] * 7
    assert applied[7] == 1.0 and applied[9] == 2.0 and applied[8] == 1.0


def test_zero_latency_applies_at_the_issue_substep():
    t = _timing(latency=(0.0, 0.0))
    t.issue(_cmd(0.5))
    assert t.advance()[0, 0].item() == 0.5


# ------------------------------------------------------------------- jitter
def test_jitter_is_uniform_within_plus_minus_the_magnitude():
    """Landing offsets over many decisions spread uniformly over L +- jitter."""
    t = _timing(n=1, latency=(50.0, 50.0), jitter=20.0, decimation=20)
    offsets = []
    for _ in range(2000):
        t.reset(torch.tensor([0]))
        t.issue(_cmd(1.0, n=1))
        offsets.append(_first_substep_with(t, 1.0))
    # L ~ U(30, 70) ms, so ceil(L / 10) is 4, 5, 6 or 7 with probability 1/4
    # each (exactly 30 ms -> 3 has probability zero)
    counts = {k: offsets.count(k) for k in range(4, 8)}
    assert set(offsets) == set(range(4, 8))
    for k in range(4, 8):
        assert abs(counts[k] / len(offsets) - 0.25) < 0.05


def test_landings_are_monotone_so_commands_never_reorder():
    t = _timing(n=1, latency=(30.0, 30.0), jitter=10.0, decimation=1)
    applied = []
    for step in range(60):
        t.issue(_cmd(step + 1, n=1))
        applied.append(t.advance()[0, 0].item())
    assert applied == sorted(applied)
    assert applied[-1] > 0


# --------------------------------------------------------------------- drops
def test_a_dropped_decision_holds_the_previous_command():
    t = _timing(n=1, latency=(0.0, 0.0), drop=1.0)
    t.issue(_cmd(1.0, n=1))             # always dropped: nothing lands
    assert t.advance()[0, 0].item() == 0.0
    assert t.pending().tolist() == [0]


def test_drop_share_matches_the_probability():
    t = _timing(n=1, latency=(0.0, 0.0), drop=0.1, decimation=1)
    held = 0
    for step in range(4000):
        t.issue(_cmd(step + 1, n=1))
        held += t.advance()[0, 0].item() != step + 1
    assert abs(held / 4000 - 0.1) < 0.02


def test_measured_hold_durations_mean_one_period():
    """Hold durations: mean T, and the share of holds >= 2T is about p."""
    p = 0.1
    t = _timing(n=1, latency=(20.0, 20.0), drop=p, decimation=DECIMATION)
    applied = []
    for step in range(3000):
        t.issue(_cmd(step + 1, n=1))
        for _ in range(DECIMATION):
            applied.append(t.advance()[0, 0].item())
    changes = [i for i in range(1, len(applied)) if applied[i] != applied[i - 1]]
    holds = [b - a for a, b in pairwise(changes)]
    mean_hold = sum(holds) / len(holds)
    assert abs(mean_hold / DECIMATION - 1 / (1 - p)) < 0.05
    assert abs(sum(h >= 2 * DECIMATION for h in holds) / len(holds) - p) < 0.03


# ------------------------------------------------------- per-env independence
def test_envs_draw_independent_latencies_and_switch_at_different_substeps():
    t = _timing(n=8, latency=(10.0, 90.0))
    t.issue(_cmd(1.0, n=8))
    first = [None] * 8
    for k in range(12):
        out = t.advance()
        for e in range(8):
            if first[e] is None and out[e, 0].item() == 1.0:
                first[e] = k
    assert len(set(first)) > 1
    assert all(f == math.ceil(t.latency_s[e].item() * 100 - 1e-6) for e, f in enumerate(first))


# --------------------------------------------------------------------- reset
def test_reset_clears_only_the_given_envs():
    t = _timing(n=2, latency=(30.0, 30.0))
    t.issue(_cmd(1.0))
    t.reset(torch.tensor([0]))              # env 0 respawns mid-flight
    assert t.pending().tolist() == [0, 1]
    for _ in range(4):                      # 30 ms = 3 substeps, lands on the 4th
        out = t.advance()
    assert out[0, 0].item() == 0.0          # env 0 lost its queued command
    assert out[1, 0].item() == 1.0          # env 1 landed it on time


def test_reset_resamples_the_base_latency():
    t = _timing(n=1, latency=(10.0, 90.0))
    first = t.latency_s.clone()
    for _ in range(5):
        t.reset(torch.tensor([0]))
        if not torch.equal(t.latency_s, first):
            break
    assert not torch.equal(t.latency_s, first)
    assert 0.01 <= t.latency_s.item() <= 0.09


# --------------------------------------------------------------- determinism
def test_same_seed_replays_the_same_draws():
    def trace(seed):
        t = _timing(n=4, latency=(10.0, 90.0), jitter=10.0, drop=0.2, seed=seed)
        out = []
        for step in range(20):
            t.issue(_cmd(step + 1, n=4))
            out.append(torch.stack([t.advance() for _ in range(DECIMATION)]))
        return torch.stack(out)
    assert torch.equal(trace(3), trace(3))
    assert not torch.equal(trace(3), trace(4))


def test_the_queue_draws_nothing_from_the_global_rng():
    torch.manual_seed(0)
    before = torch.rand(3)
    torch.manual_seed(0)
    t = _timing(n=4, latency=(10.0, 90.0), jitter=10.0, drop=0.2)
    for step in range(10):
        t.issue(_cmd(step, n=4))
        for _ in range(DECIMATION):          # one issue per control period
            t.advance()
    assert torch.equal(torch.rand(3), before)


# ------------------------------------------------------------------ capacity
def test_capacity_bounds_the_pending_commands():
    t = _timing(n=1, latency=(110.0, 110.0), decimation=1)
    assert t.capacity == max_pending(110.0, 0.0, DT, 1) == 13
    for step in range(40):                 # one issue per substep, never overflows
        t.issue(_cmd(step + 1, n=1))
        t.advance()
    assert t.pending().item() <= t.capacity


@pytest.mark.parametrize("bad", [
    {"latency": (-1.0, 10.0)}, {"latency": (50.0, 10.0)}, {"jitter": -1.0},
    {"drop": 1.5},
])
def test_invalid_ranges_are_refused(bad):
    with pytest.raises(ValueError):
        _timing(**bad)
