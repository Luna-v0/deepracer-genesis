"""Feature-set selection is a real parameter (Part C.2 / Finding 1).

These are pure width/layout checks that need no Genesis sim: they exercise the
`feature_dim`/`feature_layout`/`resolve_feature_set` surface the env sizes its
state buffer from (`base_env.py`), so a regression that silently ignores the
selected `FeatureSet` (the original Finding 1 bug) fails here.
"""

from deepracer_genesis.envs.features import (
    ClassicFeatures,
    PerceptionFeatures,
    feature_dim,
    resolve_feature_set,
)


def test_resolve_defaults_to_classic():
    assert resolve_feature_set(None) is ClassicFeatures
    assert resolve_feature_set(PerceptionFeatures) is PerceptionFeatures


def test_perception_changes_observation_width():
    """The whole point of Finding 1: selecting perception must change the
    state-vector width the env allocates, not be silently ignored."""
    classic = feature_dim(None, lookahead_k=10)
    perception = feature_dim(PerceptionFeatures, lookahead_k=10)
    assert classic != perception, (
        "PerceptionFeatures produced the same width as ClassicFeatures — the "
        "feature_set parameter is being ignored (Finding 1 regression)")


def test_explicit_classic_matches_default():
    assert feature_dim(ClassicFeatures, lookahead_k=10) == feature_dim(None, lookahead_k=10)


# --------------------------------------------- SelectFeatures (Part K.2)
def test_select_features_full_matches_classic_width():
    from deepracer_genesis.envs.features import SelectFeatures
    FULL = ("v_forward", "v_lateral", "yaw_rate", "lateral", "heading",
            "last_action", "lookahead_xy")
    assert (feature_dim(SelectFeatures, lookahead_k=10, params={"features": FULL})
            == feature_dim(ClassicFeatures, lookahead_k=10))


def test_select_features_subset_widths():
    from deepracer_genesis.envs.features import SelectFeatures
    MINIMAL = ("v_forward", "lateral", "heading")
    KINEMATIC = ("v_forward", "v_lateral", "yaw_rate", "lateral", "heading", "last_action")
    assert feature_dim(SelectFeatures, lookahead_k=10, params={"features": MINIMAL}) == 4
    assert feature_dim(SelectFeatures, lookahead_k=10, params={"features": KINEMATIC}) == 8
    # lookahead width scales with lookahead_k
    d5 = feature_dim(SelectFeatures, lookahead_k=5, params={"features": ("lookahead_xy",)})
    assert d5 == 10


def test_select_features_rejects_unknown_and_empty():
    import pytest

    from deepracer_genesis.envs.features import SelectFeatures
    with pytest.raises(ValueError, match="unknown feature block"):
        feature_dim(SelectFeatures, params={"features": ("v_forward", "bogus")})
    with pytest.raises(ValueError, match="non-empty"):
        feature_dim(SelectFeatures, params={"features": ()})
