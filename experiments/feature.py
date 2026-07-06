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
    """Feature baseline + physics DR (isolates the DR effect without vision)."""
    return (
        FeatureEnvironment(lookahead_k=10, num_envs=1024)
        >> DomainRandomizationPhysics()
        >> VectorPolicy(keys=("state",))
    ).build(seed=0, ablation_group="dr_effect", variant="physics_dr")
