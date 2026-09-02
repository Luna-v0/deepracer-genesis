"""Behaviour of the perception package: the split, the model, the jitter, the seams.

Nothing here touches ``data/``; the CNN is built and checkpointed in-test.
"""

import numpy as np
import pytest
import torch

from deepracer_genesis.perception import augment
from deepracer_genesis.perception.dataset import (DATASET_TRACKS, HOLDOUT_TRACKS,
                                                  TRAINING_TRACKS, track_names)
from deepracer_genesis.perception.features import (CHANNEL_NAMES, SIGMA,
                                                   CNNPerceptionFeatures,
                                                   NoisyPerceptionFeatures)
from deepracer_genesis.perception.model import PerceptionCNN


class _FakeEnv:
    """Minimum surface a FeatureSet constructor touches."""

    def __init__(self, *, vision=True, frame_stack=4, device="cpu", num_envs=2):
        self.vision = vision
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.cfg = {"vision": {"frame_stack": frame_stack}}
        self.camera_stack = None


def _checkpoint(tmp_path, in_channels=12, n_targets=7):
    """Save a randomly-initialised CNN and return its path."""
    path = tmp_path / "cnn.pt"
    torch.save(PerceptionCNN(in_channels=in_channels,
                             n_targets=n_targets).state_dict(), path)
    return path


# ----------------------------------------------------------------- the split
def test_the_holdout_split_is_disjoint_and_complete():
    assert set(HOLDOUT_TRACKS) & set(TRAINING_TRACKS) == set()
    assert set(HOLDOUT_TRACKS) | set(TRAINING_TRACKS) == set(DATASET_TRACKS)
    assert len(HOLDOUT_TRACKS) >= 5


def test_the_policy_and_the_cnn_hold_out_the_same_tracks():
    """The whole pipeline is measured on tracks neither half ever trained on."""
    from experiments.perception.train_policy_with_noise import (TEST_TRACKS,
                                                                TRAIN_TRACKS)

    assert set(TEST_TRACKS) == set(track_names(HOLDOUT_TRACKS))
    assert set(TRAIN_TRACKS) & set(TEST_TRACKS) == set()


# ----------------------------------------------------------------- the model
def test_the_model_maps_a_frame_stack_to_the_seven_channels():
    net = PerceptionCNN()
    y = net(torch.zeros(2, 12, 120, 160))
    assert y.shape == (2, len(CHANNEL_NAMES)) == (2, len(SIGMA))


def test_the_model_refuses_a_frame_too_small_for_its_convolutions():
    """Fails at construction with the shape, not at the first matmul."""
    with pytest.raises(ValueError, match="too small"):
        PerceptionCNN(input_hw=(16, 16))


# ---------------------------------------------------------------- the jitter
def test_augmenting_keeps_a_frame_a_frame():
    rng = np.random.default_rng(0)
    frame = rng.random((120, 160, 3), dtype=np.float32)
    out = augment.apply(frame.copy(), augment.sample_camera(rng), rng)

    assert out.shape == frame.shape and out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert not np.allclose(out, frame)


def test_augmenting_does_not_mutate_the_callers_frame():
    """apply() works in place internally; the array handed to it must survive."""
    rng = np.random.default_rng(0)
    frame = rng.random((8, 8, 3), dtype=np.float32)
    before = frame.copy()
    augment.apply(frame, augment.sample_camera(rng), rng)
    assert np.array_equal(frame, before)


def test_one_camera_state_treats_two_frames_alike():
    """Exposure cannot flicker inside a stack; only sensor noise may differ."""
    frame = np.full((60, 80, 3), 0.5, dtype=np.float32)
    camera = augment.sample_camera(np.random.default_rng(0))
    a = augment.apply(frame.copy(), camera, np.random.default_rng(1))
    b = augment.apply(frame.copy(), camera, np.random.default_rng(2))
    assert np.abs(a - b).max() <= 6 * augment.NOISE[1]


def test_the_same_seed_reproduces_the_same_jitter():
    frame = np.full((8, 8, 3), 0.5, dtype=np.float32)
    def once():
        rng = np.random.default_rng(7)
        return augment.apply(frame.copy(), augment.sample_camera(rng), rng)
    assert np.array_equal(once(), once())


# ------------------------------------------------------------- the env seams
def test_cnn_features_require_an_explicit_checkpoint():
    """No magic default: a stale or absent checkpoint must not pass silently."""
    with pytest.raises(ValueError, match="requires params\\['checkpoint'\\]"):
        CNNPerceptionFeatures(_FakeEnv(), {})


def test_cnn_features_reject_a_non_camera_env():
    with pytest.raises(ValueError, match="needs a camera env"):
        CNNPerceptionFeatures(_FakeEnv(vision=False), {"checkpoint": "unused.pt"})


def test_cnn_features_reject_a_frame_stack_the_checkpoint_cannot_read(tmp_path):
    """frame_stack=1 used to fall through and serve privileged values silently."""
    ckpt = _checkpoint(tmp_path, in_channels=12)
    with pytest.raises(ValueError, match="stacks 1 frames"):
        CNNPerceptionFeatures(_FakeEnv(frame_stack=1), {"checkpoint": ckpt})


def test_cnn_features_use_the_device_they_are_given(tmp_path):
    """cnn_device used to be compared to 'mps' and otherwise discarded."""
    ckpt = _checkpoint(tmp_path)
    fs = CNNPerceptionFeatures(_FakeEnv(), {"checkpoint": ckpt, "cnn_device": "cpu"})
    assert fs.device == torch.device("cpu")
    with pytest.raises(RuntimeError):
        CNNPerceptionFeatures(_FakeEnv(), {"checkpoint": ckpt,
                                           "cnn_device": "not-a-device"})


def test_noisy_features_reject_an_unknown_channel_name():
    fs = NoisyPerceptionFeatures(_FakeEnv(), {"noise": 1.0,
                                              "noise_channels": ("latteral",)})
    with pytest.raises(ValueError, match="unknown noise_channels"):
        fs._sigma(torch.device("cpu"), torch.float32)


def test_noisy_sigma_zeroes_every_channel_not_chosen():
    fs = NoisyPerceptionFeatures(_FakeEnv(), {"noise_channels": ("speed",)})
    sigma = fs._sigma(torch.device("cpu"), torch.float32)
    assert sigma[CHANNEL_NAMES.index("speed")] == pytest.approx(SIGMA[2])
    assert sigma.sum() == pytest.approx(SIGMA[2])


def test_the_base_env_reports_no_camera_stack():
    """The public seam CNNPerceptionFeatures reads, instead of a private buffer."""
    from deepracer_genesis.envs.base_env import DeepRacerEnv

    assert DeepRacerEnv.camera_stack.fget(object()) is None
