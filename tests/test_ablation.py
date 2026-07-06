"""Unit tests for sweep/grid/seeds + coupled-field sync (Phase 6)."""

import pytest

import experiments  # noqa: F401
from deepracer_genesis.experiment import SpecError, run
from deepracer_genesis.experiment.ablation import grid, override, seeds, sweep


def test_sweep_tags_and_values():
    base = run("safe_feature", build_only=True)
    sw = sweep(base, "env.cost_budget", [10.0, 25.0, 50.0])
    assert [s.env.cost_budget for s in sw] == [10.0, 25.0, 50.0]
    assert {s.ablation_group for s in sw} == {"sweep_cost_budget"}
    assert [s.variant for s in sw] == ["cost_budget=10.0", "cost_budget=25.0",
                                       "cost_budget=50.0"]


def test_budget_sync_env_to_algorithm_and_back():
    base = run("safe_feature", build_only=True)
    s = override(base, "env.cost_budget", 10.0)
    assert s.algorithm.lagrangian["budget"] == 10.0
    s2 = override(base, "algorithm.lagrangian.budget", 40.0)
    assert s2.env.cost_budget == 40.0


def test_divergent_budgets_rejected():
    base = run("safe_feature", build_only=True)
    s = override(base, "algorithm.lagrangian", dict(base.algorithm.lagrangian,
                                                    budget=99.0))
    with pytest.raises(SpecError, match="conflicting budgets"):
        s.validate()


def test_seeds_fan_out():
    base = run("feature_baseline", build_only=True)
    fan = seeds(base, 3)
    assert [s.seed for s in fan] == [0, 1, 2]
    assert len({s.id() for s in fan}) == 3          # seed is configuration


def test_grid_drops_invalid_combos():
    base = run("cam_baseline", build_only=True)
    g = grid(base, {"env.render": ["madrona", "nyx"],
                    "env.tracks": [("reinvent_base",),
                                   ("reinvent_base", "reInvent2019_track")]})
    variants = {s.variant for s in g}
    assert len(g) == 3                               # nyx x hetero dropped
    assert not any("nyx" in v and "reInvent2019" in v for v in variants)


def test_override_unknown_path():
    base = run("feature_baseline", build_only=True)
    with pytest.raises((SpecError, TypeError)):
        override(base, "env.does_not_exist", 1)
