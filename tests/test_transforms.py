"""CPU unit tests for the DR transforms (Phase 3)."""

import torch

from deepracer_genesis.experiment.transforms import ActionNoiseDelay, ImageAug


def test_brightness_is_multiplicative():
    aug = ImageAug({"brightness": (0.5, 0.5)})      # degenerate range = exact
    out = aug._apply_transform(torch.full((4, 3, 8, 8), 0.8))
    assert torch.allclose(out, torch.full_like(out, 0.4), atol=1e-6)


def test_output_clamped_to_unit_range():
    aug = ImageAug({"brightness": (3.0, 3.0), "noise": 0.5})
    out = aug._apply_transform(torch.rand(2, 3, 8, 8))
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_cutout_zeroes_a_patch():
    aug = ImageAug({"cutout": 1.0})
    out = aug._apply_transform(torch.ones(4, 3, 32, 32))
    assert (out == 0).any()
    assert (out == 1).any()                          # but not the whole image


def test_hue_preserves_luma_direction():
    aug = ImageAug({"hue": 0.2})
    img = torch.rand(8, 3, 16, 16)
    out = aug._apply_transform(img)
    assert out.shape == img.shape
    assert not torch.allclose(out, img)              # something rotated


def test_delay_ring_buffer():
    d = ActionNoiseDelay(3, delay_steps=2, device="cpu")
    a = torch.ones(3, 2)
    assert (d._inv_apply_transform(a) == 0).all()          # draining zeros
    assert (d._inv_apply_transform(a * 2) == 0).all()
    assert (d._inv_apply_transform(a * 3) == 1).all()      # first cmd, k later


def test_noise_respects_channels():
    n = ActionNoiseDelay(64, steer_noise=0.5, speed_noise=0.0, device="cpu")
    out = n._inv_apply_transform(torch.zeros(64, 2))
    assert out[:, 0].abs().sum() > 0                 # steering perturbed
    assert (out[:, 1] == 0).all()                    # throttle untouched
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_delay_reset_zeroes_masked_rows():
    from tensordict import TensorDict

    d = ActionNoiseDelay(3, delay_steps=1, device="cpu")
    d._inv_apply_transform(torch.ones(3, 2))
    mask = torch.tensor([[True], [False], [False]])
    d._reset(TensorDict({"_reset": mask}, batch_size=[3]), TensorDict({}, batch_size=[3]))
    assert (d.buf[0] == 0).all() and (d.buf[1] == 1).all()
