"""``episode_length_s`` is a spec knob (P3), hash-exempt at its unset default."""

from deepracer_genesis.experiment.builder import Builder
from deepracer_genesis.experiment.spec import AlgorithmSpec, ExperimentSpec, PolicySpec
from deepracer_genesis.experiment.stages import FeatureEnvironment, VectorPolicy


def _spec(**env_kw):
    spec = ExperimentSpec(policy=PolicySpec(actor_keys=("state",),
                                            critic_keys=("state",)),
                          algorithm=AlgorithmSpec())
    return FeatureEnvironment(**env_kw).apply(spec)


def test_episode_length_threads_from_stage_to_cfg():
    cfg = Builder(_spec(episode_length_s=60.0)).sim_cfg()
    assert cfg["sim"]["episode_length_s"] == 60.0
    assert Builder(_spec()).sim_cfg()["sim"]["episode_length_s"] == 30.0


def test_episode_length_is_hashed_only_when_set():
    # pre-P3 hash stability at the unset default is pinned by the golden ids
    # in test_custom_rewards.py; here: setting the knob must change the id
    assert _spec().env.episode_length_s is None
    assert _spec().id() != _spec(episode_length_s=60.0).id()
