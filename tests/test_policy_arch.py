"""Architecture/topology plumbing: spec mlp dict -> rsl-rl train cfg.

Covers what the mlp dict can now express (depth/width, activation,
LSTM/GRU recurrence) and the guard rails, all GPU-free: spec building and
cfg translation are pure-python.
"""

import pytest

from deepracer_genesis.experiment import (AsymmetricCameraPolicy,
                                          CameraEnvironment,
                                          FeatureEnvironment, SpecError,
                                          VectorPolicy)
from deepracer_genesis.experiment.rsl_backend import spec_to_train_cfg


def _feature_spec(mlp=None):
    """A minimal validated feature-vector spec with the given mlp dict."""
    pipe = FeatureEnvironment(num_envs=16) >> VectorPolicy(mlp=mlp)
    return pipe.build(seed=0, total_env_steps=1000, eval_every_steps=0)


def test_hidden_and_activation_reach_both_nets():
    """Depth/width AND activation land on actor and critic configs."""
    cfg = spec_to_train_cfg(_feature_spec(
        mlp={"hidden": (512, 512, 256, 128), "activation": "tanh"}))
    for net in ("actor", "critic"):
        assert cfg[net]["hidden_dims"] == [512, 512, 256, 128]
        assert cfg[net]["activation"] == "tanh"


def test_default_mlp_keeps_cfg_defaults():
    """An unset mlp dict must not touch the cfg (spec-hash stability)."""
    cfg = spec_to_train_cfg(_feature_spec())
    assert cfg["actor"]["class_name"] == "MLPModel"
    assert cfg["actor"]["activation"] == "elu"


def test_rnn_switches_to_rnn_model():
    """An rnn config selects RNNModel with its cell/size/layer knobs."""
    cfg = spec_to_train_cfg(_feature_spec(
        mlp={"hidden": (128,), "rnn": {"type": "gru", "hidden": 192,
                                       "layers": 2}}))
    for net in ("actor", "critic"):
        assert cfg[net]["class_name"] == "RNNModel"
        assert cfg[net]["rnn_type"] == "gru"
        assert cfg[net]["rnn_hidden_dim"] == 192
        assert cfg[net]["rnn_num_layers"] == 2


def test_rnn_defaults_are_lstm():
    """An empty rnn dict means a 1-layer 256-wide LSTM."""
    cfg = spec_to_train_cfg(_feature_spec(mlp={"rnn": {}}))
    assert cfg["actor"]["rnn_type"] == "lstm"
    assert cfg["actor"]["rnn_hidden_dim"] == 256
    assert cfg["actor"]["rnn_num_layers"] == 1


def test_camera_policy_refuses_recurrence():
    """rsl-rl's RNNModel has no CNN trunk — camera + rnn must not build."""
    pipe = (CameraEnvironment(render="madrona", num_envs=8)
            >> AsymmetricCameraPolicy(actor_keys=("camera",),
                                      critic_keys=("camera", "state"),
                                      mlp={"rnn": {"type": "lstm"}}))
    with pytest.raises(SpecError, match="RNNModel has no CNN trunk"):
        pipe.build(seed=0, total_env_steps=1000, eval_every_steps=0)


def test_bad_rnn_type_refused():
    """Unsupported cell types fail at validate, not inside rsl-rl."""
    with pytest.raises(SpecError, match="lstm.*gru"):
        _feature_spec(mlp={"rnn": {"type": "transformer"}})
