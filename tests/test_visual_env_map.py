"""Part P.1 env-map (Nyx sky) DR — the GPU-free slice.

The Nyx application (EnvironmentMapAsset/set_env_map) is GPU-blocked and
deferred; what lands and is tested here is the pure-torch sampler + the config/
catalog/spec/builder wiring that makes the knob reachable.
"""

import pytest
import torch

from deepracer_genesis.experiment import (
    AsymmetricCameraPolicy,
    CameraEnvironment,
    DomainRandomizationTrackAppearance,
)
from deepracer_genesis.experiment.builder import Builder
from deepracer_genesis.experiment.spec import (
    AlgorithmSpec,
    EnvSpec,
    ExperimentSpec,
    ObsDRSpec,
    PolicySpec,
    SpecError,
)
from deepracer_genesis.randomization.catalog import BY_NAME, by_layer
from deepracer_genesis.randomization.visual import sample_env_map


# ------------------------------------------------------------------ sampler
def test_env_map_tint_and_multiplier_within_range():
    tint, mult = sample_env_map(64, device="cpu")
    assert tint.shape == (64, 3) and mult.shape == (64,)
    assert tint.min() >= 0.35 and tint.max() < 0.75
    assert mult.min() >= 0.5 and mult.max() < 2.0


def test_env_map_custom_ranges():
    tint, mult = sample_env_map(32, tint_range=(0.1, 0.2), mult_range=(1.0, 1.5),
                                device="cpu")
    assert tint.min() >= 0.1 and tint.max() < 0.2
    assert mult.min() >= 1.0 and mult.max() < 1.5


def test_env_map_deterministic_with_generator():
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    t1, m1 = sample_env_map(16, device="cpu", generator=g1)
    t2, m2 = sample_env_map(16, device="cpu", generator=g2)
    assert torch.equal(t1, t2) and torch.equal(m1, m2)


# ------------------------------------------------------------------ catalog
def test_env_map_knobs_registered():
    for name in ("env_map_tint", "env_map_multiplier"):
        assert name in BY_NAME
        assert BY_NAME[name].layer == "visual"
        assert BY_NAME[name].signals == ("camera",)
    visual = {k.name for k in by_layer("visual")}
    assert {"env_map_tint", "env_map_multiplier"} <= visual


# ------------------------------------------------------- spec + builder wiring
def test_env_map_requires_camera_env():
    spec = ExperimentSpec(
        env=EnvSpec(modality="feature"),
        policy=PolicySpec(actor_keys=("state",), critic_keys=("state",)),
        algorithm=AlgorithmSpec(),
        obs_dr=ObsDRSpec(env_map={"tint": (0.35, 0.75)}))
    with pytest.raises(SpecError, match="sky"):
        spec.validate()


def _camera_policy():
    return AsymmetricCameraPolicy(actor_keys=("camera",),
                                  critic_keys=("camera", "state"))


def test_env_map_authoring_and_builder_wiring():
    # render='nyx': env maps exist only as Nyx build-time assets, and the
    # knob-compat matrix now (correctly) refuses env_map under Madrona.
    spec = (CameraEnvironment(render="nyx")
            >> DomainRandomizationTrackAppearance(
                env_map_tint=(0.35, 0.75), env_map_multiplier=(0.5, 2.0))
            >> _camera_policy()).build()
    assert spec.obs_dr.env_map == {"tint": (0.35, 0.75), "multiplier": (0.5, 2.0)}
    cfg = Builder(spec).sim_cfg()
    assert cfg["vision"]["env_map"] == {"tint": (0.35, 0.75), "multiplier": (0.5, 2.0)}


def test_no_env_map_leaves_cfg_default_empty():
    plain = (CameraEnvironment(render="madrona") >> _camera_policy()).build()
    assert plain.obs_dr.env_map == {}
    assert Builder(plain).sim_cfg()["vision"]["env_map"] == {}
