"""Feature-vector experiments — the baseline everything else is measured against."""

from deepracer_genesis.experiment import (
    DomainRandomizationPhysics,
    FeatureEnvironment,
    VectorPolicy,
    experiment,
)


@experiment
def feature_baseline():
    """Phase-1 baseline: state-vector PPO, no DR, no vision."""
    return (
        FeatureEnvironment(lookahead_k=10, num_envs=1024)
        >> VectorPolicy(keys=("state",))
    ).build(seed=0, ablation_group="baselines", variant="feature")


@experiment
def feature_dr():
    """dr_effect_feature pairing: physics-DR treatment on the feature env."""
    return (
        FeatureEnvironment(lookahead_k=10, num_envs=1024)
        >> DomainRandomizationPhysics()
        >> VectorPolicy(keys=("state",))
    ).build(seed=0, ablation_group="dr_effect_feature", variant="physics_dr")


@experiment
def feature_nodr():
    """dr_effect_feature pairing: the no-DR baseline (same config as
    feature_baseline; shares its content id, tagged into this group)."""
    return (
        FeatureEnvironment(lookahead_k=10, num_envs=1024)
        >> VectorPolicy(keys=("state",))
    ).build(seed=0, ablation_group="dr_effect_feature", variant="no_dr")
