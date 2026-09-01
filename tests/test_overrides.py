"""Unit tests for override() + coupled-field sync.

Self-contained: builds its specs inline via the DSL so it does not depend on any
authored experiments/examples package.
"""

import pytest

from deepracer_genesis.experiment import (
    FeatureEnvironment,
    SafeRLFeatureEnvironment,
    SpecError,
    VectorPolicy,
)
from deepracer_genesis.experiment.overrides import override


def _safe_feature():
    return (SafeRLFeatureEnvironment(cost="offtrack", budget=25.0, num_envs=64)
            >> VectorPolicy(keys=("state",))).build()


def _feature():
    return (FeatureEnvironment(num_envs=8) >> VectorPolicy(keys=("state",))).build()


def test_budget_sync_env_to_algorithm_and_back():
    base = _safe_feature()
    s = override(base, "env.cost_budget", 10.0)
    assert s.algorithm.lagrangian["budget"] == 10.0
    s2 = override(base, "algorithm.lagrangian.budget", 40.0)
    assert s2.env.cost_budget == 40.0


def test_divergent_budgets_rejected():
    base = _safe_feature()
    s = override(base, "algorithm.lagrangian", dict(base.algorithm.lagrangian,
                                                    budget=99.0))
    with pytest.raises(SpecError, match="conflicting budgets"):
        s.validate()


def test_override_unknown_path():
    base = _feature()
    with pytest.raises((SpecError, TypeError)):
        override(base, "env.does_not_exist", 1)
