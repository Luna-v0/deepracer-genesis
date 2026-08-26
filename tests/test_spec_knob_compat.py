"""DR-honesty knob-compat tests: a knob either acts or refuses to build.

The compatibility matrix lives on the catalog knobs (``Knob.modalities`` /
``Knob.renderers``) and ``ExperimentSpec._validate_knob_compat`` enforces it,
so an unsupported knob fails the build loudly instead of silently sampling
nothing. Table-driven over that matrix, plus ``_active_knobs`` unit coverage,
the ``EnvSpec.effective_renderer``/builder parity, the builder's top-level DR
emission, and the run-identity lock.
"""

import dataclasses
import warnings

import pytest

from deepracer_genesis.experiment import (
    PPO,
    AsymmetricCameraPolicy,
    CameraEnvironment,
    DomainRandomizationActions,
    DomainRandomizationCamera,
    DomainRandomizationPhysics,
    DomainRandomizationTrackAppearance,
    FeatureEnvironment,
    SpecError,
    VectorPolicy,
)
from deepracer_genesis.experiment.builder import Builder
from deepracer_genesis.experiment.spec import (
    NEUTRAL_PHYSICS,
    EnvSpec,
    _active_knobs,
)

# ----------------------------------------------------------------- helpers

def _camera_policy():
    """The minimal asymmetric camera policy used across the tables."""
    return AsymmetricCameraPolicy(actor_keys=("camera",),
                                  critic_keys=("camera", "state"))


def _build_quiet(pipeline):
    """Build a pipeline with advisory UserWarnings silenced.

    The camera-on-CPU rasterizer path warns by design (debug path, not a
    throughput path); these tests assert knob compatibility, not that advisory.

    Args:
        pipeline: The ``>>`` pipeline to build.

    Returns:
        The built (validated) ExperimentSpec.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pipeline.build()


def _base_camera_spec():
    """A valid DR-free camera spec to hand-mutate via dataclasses.replace."""
    return (CameraEnvironment() >> _camera_policy()).build()


# --------------------------------------------------- compat matrix: PASS

PASS_CASES = {
    # physics DR under camera is fine while track_width stays at its
    # neutral (1.0, 1.0) default (the stage always emits every key)
    "camera_madrona_neutral_track_width": lambda: (
        CameraEnvironment(render="madrona")
        >> DomainRandomizationPhysics()
        >> _camera_policy()),
    # mount jitter acts under Madrona (batched attach-offset write)
    "camera_madrona_mount_jitter": lambda: (
        CameraEnvironment(render="madrona")
        >> DomainRandomizationCamera(camera_jitter=True)
        >> _camera_policy()),
    # env maps are Nyx build-time assets -> allowed under render='nyx'
    "camera_nyx_env_map": lambda: (
        CameraEnvironment(render="nyx")
        >> DomainRandomizationTrackAppearance(strength=0.3,
                                              env_map_tint=(0.3, 0.8))
        >> _camera_policy()),
    # new image knobs (cutout/noise) act under any camera renderer
    "camera_cutout_and_noise": lambda: (
        CameraEnvironment()
        >> DomainRandomizationCamera(cutout=0.3, noise=0.02)
        >> _camera_policy()),
    # track-width DR is the feature-mode geometry knob
    "feature_track_width": lambda: (
        FeatureEnvironment()
        >> DomainRandomizationPhysics(track_width=(0.9, 1.15))
        >> VectorPolicy()),
    # actuation DR is modality-agnostic
    "feature_action_dr": lambda: (
        FeatureEnvironment()
        >> VectorPolicy()
        >> DomainRandomizationActions(steer_noise=0.02)),
    # the per-env rasterizer supports mount jitter now (randomize_mount)
    "camera_cpu_rasterizer_mount_jitter": lambda: (
        CameraEnvironment(backend="cpu")
        >> DomainRandomizationCamera(camera_jitter=True)
        >> _camera_policy()),
}


@pytest.mark.parametrize("make", PASS_CASES.values(), ids=PASS_CASES.keys())
def test_compatible_knobs_build(make):
    _build_quiet(make())                        # must not raise


# ----------------------------------------------- compat matrix: SpecError

FAIL_CASES = {
    # camera mode: the mesh is fixed at build, the rulebook must not desync
    "track_width_under_camera": (lambda: (
        CameraEnvironment()
        >> DomainRandomizationPhysics(track_width=(0.9, 1.15))
        >> _camera_policy()), "track_width_scale"),
    # Nyx has ONE shared sensor offset -> per-env mount jitter is inert
    "mount_jitter_under_nyx": (lambda: (
        CameraEnvironment(render="nyx")
        >> DomainRandomizationCamera(camera_jitter=True)
        >> _camera_policy()), "camera_pitch_jitter"),
    # Madrona replicates lights identically into every world -> no env map
    "env_map_under_madrona": (lambda: (
        CameraEnvironment(render="madrona")
        >> DomainRandomizationTrackAppearance(env_map_tint=(0.35, 0.75))
        >> _camera_policy()), "env_map"),
}


@pytest.mark.parametrize("make,match", FAIL_CASES.values(), ids=FAIL_CASES.keys())
def test_incompatible_knobs_refuse_to_build(make, match):
    with pytest.raises(SpecError, match=match):
        make().build()


# a typo'd DR key would be silently ignored at runtime -> refuse to build
UNKNOWN_KEY_CASES = {
    "image_aug_typo": {"image_aug": {"brigthness": (0.7, 1.3)}},
    "physics_typo": {"physics": {"frictoin_range": (0.6, 1.4)}},
    "camera_jitter_typo": {"camera_jitter": {"pitch": 2.0}},
}


@pytest.mark.parametrize("dr_field", UNKNOWN_KEY_CASES.values(),
                         ids=UNKNOWN_KEY_CASES.keys())
def test_unknown_dr_keys_rejected(dr_field):
    base = _base_camera_spec()
    spec = dataclasses.replace(
        base, obs_dr=dataclasses.replace(base.obs_dr, **dr_field))
    with pytest.raises(SpecError, match="unknown key"):
        spec.validate()


# ------------------------------------------------------------ _active_knobs

def test_active_knobs_filters_neutral_physics():
    spec = (FeatureEnvironment()
            >> DomainRandomizationPhysics()
            >> VectorPolicy()).build()
    names = {knob.name for knob, _ in _active_knobs(spec)}
    assert "track_width_scale" not in names     # (1.0, 1.0) default is inert
    assert {"friction", "mass_shift", "steer_kp_scale"} <= names

    all_neutral = dataclasses.replace(
        spec, obs_dr=dataclasses.replace(spec.obs_dr,
                                         physics=dict(NEUTRAL_PHYSICS)))
    assert _active_knobs(all_neutral) == []     # fully neutral = no activation


def test_active_knobs_filters_falsy_image_values():
    base = _base_camera_spec()
    spec = dataclasses.replace(base, obs_dr=dataclasses.replace(
        base.obs_dr,
        image_aug={"brightness": (0.7, 1.3), "blur": 0.0, "hue": 0.0}))
    names = [knob.name for knob, _ in _active_knobs(spec)]
    assert names == ["brightness"]              # falsy values are not activations


def test_active_knobs_pixel_noise_and_action_dr():
    spec = (CameraEnvironment()
            >> DomainRandomizationCamera(pixel_noise=0.03)
            >> _camera_policy()
            >> DomainRandomizationActions(steer_noise=0.02, delay_steps=1)).build()
    entries = {knob.name: value for knob, value in _active_knobs(spec)}
    assert entries["pixel_noise"] == 0.03
    assert entries["steer_noise"] == 0.02
    assert entries["delay_steps"] == 1
    assert "speed_noise" not in entries         # 0.0 stays inactive


# ------------------------------------------- renderer resolution + parity

@pytest.mark.parametrize("modality,render,backend,expected", [
    ("feature", "none", "gpu", None),
    ("feature", "none", "cpu", None),
    ("camera", "madrona", "gpu", "madrona"),
    ("camera", "nyx", "gpu", "nyx"),
    ("camera", "madrona", "cpu", "rasterizer"),
    ("camera", "nyx", "cpu", "rasterizer"),     # cpu wins over explicit nyx
])
def test_effective_renderer_matrix(modality, render, backend, expected):
    env = EnvSpec(modality=modality, render=render, backend=backend)
    assert env.effective_renderer == expected


@pytest.mark.parametrize("render,backend,expected", [
    ("madrona", "gpu", "batch"),                # madrona keeps the cfg default
    ("nyx", "gpu", "nyx"),
    ("madrona", "cpu", "rasterizer"),
    ("nyx", "cpu", "rasterizer"),
])
def test_builder_renderer_parity(render, backend, expected):
    spec = _build_quiet(CameraEnvironment(render=render, backend=backend)
                        >> _camera_policy())
    with warnings.catch_warnings():             # Builder re-validates (cpu warns)
        warnings.simplefilter("ignore")
        cfg = Builder(spec).sim_cfg()
    assert cfg["vision"]["vision_renderer"] == expected


# --------------------------------------------------- builder DR emission

def test_builder_emits_top_level_dr_cfg():
    # formerly rsl_backend._dr_extra_cfg (deleted): every build path — eval,
    # preview, datasets, plain Builder.sim() — now sees the env-side DR keys
    spec = (CameraEnvironment()
            >> DomainRandomizationCamera(brightness=(0.7, 1.3))
            >> _camera_policy()
            >> DomainRandomizationActions(delay_steps=2)).build()
    cfg = Builder(spec).sim_cfg()
    assert cfg["image_aug"] == {"brightness": (0.7, 1.3)}
    assert cfg["action_dr"] == {"steer_noise": 0.0, "speed_noise": 0.0,
                                "delay_steps": 2}


def test_dr_free_spec_emits_no_dr_cfg():
    cfg = Builder(_base_camera_spec()).sim_cfg()
    assert "image_aug" not in cfg
    assert "action_dr" not in cfg


# ------------------------------------------------------- run-identity lock

def test_run_identity_locked():
    # Pins spec serialization / run identity: the DR-honesty change (catalog
    # modalities/renderers + knob-compat validation) must NOT shift existing
    # run ids — equal configs keep their run dirs across the change. If a
    # spec FIELD ever changes intentionally, update this constant consciously
    # in the same commit (it is a retrain-the-world event, not noise).
    spec = (CameraEnvironment() >> AsymmetricCameraPolicy() >> PPO()).build()
    assert spec.id() == "3ddf1ade78ca"
