"""Unit tests for the torchrl-agnostic episodic aggregation (Phase 1)."""

import math

import torch

from deepracer_genesis.experiment.evaluator import aggregate_episodes


L = 10.0   # track length used throughout


def _streams(T, N):
    return dict(
        reward=torch.zeros(T, N),
        done=torch.zeros(T, N, dtype=torch.bool),
        progress_delta=torch.zeros(T, N),
        offtrack=torch.zeros(T, N, dtype=torch.bool),
    )


def test_no_completed_episode():
    m = aggregate_episodes(control_dt=0.02, track_length=L, **_streams(10, 4))
    assert m == {"episodes": 0}


def test_single_episode_accounting():
    s = _streams(10, 1)
    s["reward"][:, 0] = 1.0
    s["progress_delta"][:, 0] = 2.0            # 10 m over the 5-step episode = 1 lap
    s["done"][4, 0] = True
    m = aggregate_episodes(control_dt=0.02, track_length=L, **s)
    assert m["episodes"] == 1
    assert m["mean_return"] == 5.0             # steps 0..4 only
    assert m["mean_progress_m"] == 10.0
    assert m["completion_rate"] == 1.0
    assert math.isclose(m["lap_time_s"], 5 * 0.02, rel_tol=1e-6)
    assert m["offtrack_rate"] == 0.0


def test_laps_from_progress_not_line_crossings():
    # a car spawning just before the finish line gains almost no progress:
    # crossing the line must NOT count as a completed lap
    s = _streams(5, 1)
    s["progress_delta"][0, 0] = 0.3            # crossed the line, tiny progress
    s["done"][4, 0] = True
    m = aggregate_episodes(control_dt=0.02, track_length=L, **s)
    assert m["completion_rate"] == 0.0
    assert math.isnan(m["lap_time_s"])


def test_partial_trailing_episode_excluded():
    s = _streams(10, 1)
    s["reward"][:, 0] = 1.0
    s["done"][3, 0] = True                     # one finished episode (4 steps)
    m = aggregate_episodes(control_dt=0.02, track_length=L, **s)
    assert m["episodes"] == 1
    assert m["mean_return"] == 4.0             # trailing 6 steps not counted


def test_accumulators_reset_between_episodes():
    s = _streams(8, 1)
    s["reward"][:, 0] = 1.0
    s["done"][2, 0] = True                     # 3 steps
    s["done"][6, 0] = True                     # 4 steps
    s["offtrack"][6, 0] = True
    m = aggregate_episodes(control_dt=0.02, track_length=L, **s)
    assert m["episodes"] == 2
    assert m["mean_return"] == 3.5
    assert m["offtrack_rate"] == 0.5


def test_cost_metrics():
    s = _streams(6, 2)
    s["done"][5, :] = True
    cost = torch.zeros(6, 2)
    cost[:, 0] = 1.0                           # ep cost 6 (> budget)
    cost[:, 1] = 0.5                           # ep cost 3 (< budget)
    m = aggregate_episodes(control_dt=0.02, track_length=L,
                           cost=cost, cost_budget=5.0, **s)
    assert m["mean_cost"] == 4.5
    assert m["cost_violation_rate"] == 0.5
    assert m["budget_satisfied"] is True       # mean 4.5 <= 5.0


def test_multi_lap_lap_time_and_per_env_track_length():
    s = _streams(10, 2)
    s["done"][9, :] = True
    s["progress_delta"][:, 0] = 2.0            # 20 m = 2 laps of L=10
    s["progress_delta"][:, 1] = 2.0            # 20 m = 1 lap of L=20
    m = aggregate_episodes(control_dt=1.0, track_length=torch.tensor([10.0, 20.0]), **s)
    assert m["completion_rate"] == 1.0
    # lap times: env0 10s/2laps = 5s; env1 10s/1lap = 10s -> mean 7.5
    assert math.isclose(m["lap_time_s"], 7.5, rel_tol=1e-6)
