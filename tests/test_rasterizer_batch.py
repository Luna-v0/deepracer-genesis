"""RasterizerObsRenderer renders a real batch, not one env broadcast N times.

Builds a small CPU camera sim, so these are the slow tests of the suite.
"""

import numpy as np
import torch

from deepracer_genesis.envs.features import PerceptionFeatures
from deepracer_genesis.envs.renderers import _FLEET_CAR_THRESHOLD
from deepracer_genesis.experiment import CameraEnvironment, PPO, VectorPolicy
from deepracer_genesis.experiment.builder import Builder

RES = (160, 120)            # (W, H) — DeepRacer-native camera
SPECTATOR_RES = (480, 360)  # (W, H)


def _sim(num_envs, tracks, random_start=False, extra_cfg=None):
    """Build and step a CPU camera sim once, so the image buffer is populated.

    Args:
        num_envs: Parallel simulation instances.
        tracks: Track names; more than one triggers Part O spatial tiling.
        random_start: Whether episodes begin at a random track position.
        extra_cfg: Config sections merged in at construction (e.g. spectator).

    Returns:
        The stepped ``DeepRacerEnv``.
    """
    torch.manual_seed(0)    # spawn jitter is drawn from the global RNG
    spec = (CameraEnvironment(backend="cpu", resolution=RES, num_envs=num_envs,
                              tracks=tracks, feature_set=PerceptionFeatures,
                              random_start=random_start, frame_stack=4)
            >> VectorPolicy(keys=("state",)) >> PPO()).build(seed=0)
    sim = Builder(spec).sim(extra_cfg=extra_cfg)
    sim.get_observations()
    sim.step(torch.zeros(num_envs, 2, device=sim.device))
    return sim


def _pairwise_distinct(images):
    """Report whether every pair of frames differs somewhere.

    Args:
        images: An ``(N, C, H, W)`` batch of frames.

    Returns:
        True when no two frames are pixel-identical.
    """
    n = len(images)
    return all((images[i] - images[j]).abs().max() > 0
               for i in range(n) for j in range(i + 1, n))


def _car_mask(frame, background):
    """Mask the pixels of one frame that read as a car, not as empty track.

    Args:
        frame: An ``(H, W, 3)`` uint8 frame.
        background: The ``(H, W, 3)`` uint8 empty-track reference.

    Returns:
        An ``(H, W)`` boolean mask of the departing pixels.
    """
    delta = np.abs(frame.astype(np.int16) - background).sum(-1)
    return delta > _FLEET_CAR_THRESHOLD


def test_batched_camera_keeps_every_env_distinct():
    sim = _sim(4, ("reinvent_base",))
    img = sim.obs_image_buf

    w, h = RES
    assert img.shape == (4, 3, h, w)
    assert img.dtype == torch.float32
    assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0
    # the whole point of env_separate_rigid: N real renders, not env 0 tiled
    assert _pairwise_distinct(img)


def test_tiled_multi_track_frames_stay_on_their_own_tile():
    sim = _sim(6, ("reinvent_base", "reInvent2019_track"))
    img = sim.obs_image_buf
    offsets = sim.track.variant_offset                       # (V, 2)
    variant = sim.track.variant_idx                          # (N,)
    assert len(offsets) == 2 and set(variant.tolist()) == {0, 1}

    # every car sits on its home tile, so its camera only ever frames that track
    d = (sim.base_pos[:, None, :2] - offsets[None]).norm(dim=-1)   # (N, V)
    assert torch.equal(d.argmin(dim=1), variant)

    for i in range(len(img)):
        for j in range(i + 1, len(img)):
            if variant[i] != variant[j]:
                assert (img[i] - img[j]).abs().max() > 0


def test_spectator_composites_the_whole_fleet_into_one_frame():
    sim = _sim(4, ("reinvent_base",), random_start=True,
               extra_cfg={"vision": {"spectator": True,
                                     "spectator_res": SPECTATOR_RES}})
    raw = np.asarray(sim.spec_cam.render(rgb=True)[0])
    assert raw.ndim == 4, "env_separate_rigid batches the spectator camera too"

    w, h = SPECTATOR_RES
    frame = sim.render_spectator()
    assert frame.shape == (h, w, 3)

    # every car reaches the composite, not just env 0's (the collapsed contract)
    background = np.median(raw, axis=0).astype(np.uint8)
    composed = _car_mask(frame, background)
    for i in range(len(raw)):
        car = _car_mask(raw[i], background)
        assert car.sum() > 0
        assert (car & composed).sum() == car.sum()
