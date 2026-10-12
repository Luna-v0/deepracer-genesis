"""Shared HPO/DR search-space types (Part H).

Pure, no-sim checks: DR ``sample`` shapes/bounds, HPO ``suggest`` routing via a
fake trial, ``to_cfg`` shapes (tuple vs scalar) the DR code depends on, and the
guardrails the verifier pinned down (log needs positive lo; Choice.sample and
SymRange.suggest raise).
"""

import pytest

from deepracer_genesis.randomization.spaces import (
    Choice,
    FloatRange,
    IntRange,
    SymRange,
)


class _FakeTrial:
    """Records optuna-style suggest_* calls and returns the low end."""

    def __init__(self):
        self.calls = []

    def suggest_float(self, name, lo, hi, log=False):
        self.calls.append(("float", name, lo, hi, log))
        return lo

    def suggest_int(self, name, lo, hi):
        self.calls.append(("int", name, lo, hi))
        return lo

    def suggest_categorical(self, name, values):
        self.calls.append(("cat", name, tuple(values)))
        return values[0]


# ------------------------------------------------------------------ FloatRange
def test_floatrange_sample_shape_and_bounds():
    s = FloatRange(0.6, 1.4)
    t = s.sample(1000, "cpu")
    assert t.shape == (1000,)
    assert t.min() >= 0.6 and t.max() <= 1.4


def test_floatrange_log_sample_positive_and_in_range():
    s = FloatRange(1e-4, 1e-1, log=True)
    t = s.sample(1000, "cpu")
    assert (t > 0).all()
    assert t.min() >= 1e-4 - 1e-9 and t.max() <= 1e-1 + 1e-9


def test_floatrange_log_rejects_nonpositive_lo():
    with pytest.raises(ValueError, match="strictly-positive"):
        FloatRange(0.0, 1.0, log=True)
    with pytest.raises(ValueError, match="strictly-positive"):
        FloatRange(-1.0, 1.0, log=True)


def test_floatrange_rejects_inverted():
    with pytest.raises(ValueError, match="hi < lo"):
        FloatRange(2.0, 1.0)


def test_floatrange_suggest_and_cfg():
    trial = _FakeTrial()
    s = FloatRange(0.6, 1.4, log=True)
    assert s.suggest(trial, "friction") == 0.6
    assert trial.calls == [("float", "friction", 0.6, 1.4, True)]
    assert s.to_cfg() == (0.6, 1.4)


# -------------------------------------------------------------------- IntRange
def test_intrange_sample_inclusive_bounds():
    s = IntRange(0, 3)
    t = s.sample(2000, "cpu")
    assert t.shape == (2000,)
    assert int(t.min()) == 0 and int(t.max()) == 3   # hi is inclusive


def test_intrange_suggest_and_cfg():
    trial = _FakeTrial()
    s = IntRange(1, 5)
    assert s.suggest(trial, "delay") == 1
    assert trial.calls == [("int", "delay", 1, 5)]
    assert s.to_cfg() == (1, 5)


# -------------------------------------------------------------------- SymRange
def test_symrange_sample_symmetric():
    s = SymRange(0.5)
    t = s.sample(2000, "cpu")
    assert t.shape == (2000,)
    assert t.min() >= -0.5 and t.max() <= 0.5
    assert t.min() < 0 < t.max()          # spans both signs


def test_symrange_cfg_is_scalar_not_tuple():
    """The DR code branches on shape: SymRange must emit a bare magnitude."""
    assert SymRange(0.3).to_cfg() == 0.3


def test_symrange_suggest_raises():
    with pytest.raises(NotImplementedError, match="DR-native"):
        SymRange(0.3).suggest(_FakeTrial(), "mass_shift")


def test_symrange_rejects_negative():
    with pytest.raises(ValueError, match=">= 0"):
        SymRange(-0.1)


# ---------------------------------------------------------------------- Choice
def test_choice_suggest_routes_categorical():
    trial = _FakeTrial()
    s = Choice(["relu", "elu"])
    assert s.suggest(trial, "act") == "relu"
    assert trial.calls == [("cat", "act", ("relu", "elu"))]


def test_choice_sample_raises():
    with pytest.raises(NotImplementedError, match="HPO-only"):
        Choice([1, 2, 3]).sample(4, "cpu")


def test_choice_to_cfg_raises():
    with pytest.raises(NotImplementedError):
        Choice([1, 2]).to_cfg()


def test_choice_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        Choice([])


# ------------------------------------------------- DR-side wiring (Part H)
def test_dr_physics_accepts_spaces_matching_raw_cfg_shapes():
    """DomainRandomizationPhysics with Space objects emits the same
    cfg['rand'] shapes (tuple vs scalar) as raw tuples/scalars."""
    from deepracer_genesis.experiment import (
        FeatureEnvironment,
        VectorPolicy,
    )
    from deepracer_genesis.experiment.stages import DomainRandomizationPhysics

    raw = (FeatureEnvironment(num_envs=8)
           >> DomainRandomizationPhysics(friction=(0.6, 1.4), mass=0.2, com=0.01,
                                         gains=(0.8, 1.2), armature=(0.0, 0.01))
           >> VectorPolicy()).build()
    spaced = (FeatureEnvironment(num_envs=8)
              >> DomainRandomizationPhysics(friction=FloatRange(0.6, 1.4),
                                            mass=SymRange(0.2), com=SymRange(0.01),
                                            gains=FloatRange(0.8, 1.2),
                                            armature=FloatRange(0.0, 0.01))
              >> VectorPolicy()).build()
    assert raw.obs_dr.physics == spaced.obs_dr.physics
    # friction is a (lo, hi) tuple; mass is a bare scalar magnitude
    assert spaced.obs_dr.physics["friction_range"] == (0.6, 1.4)
    assert spaced.obs_dr.physics["mass_shift_kg"] == 0.2
