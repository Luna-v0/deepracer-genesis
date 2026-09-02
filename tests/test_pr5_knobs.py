"""The public knobs PR #5 added: env `max_speed`, PPO `schedule`/`desired_kl`,
`ExperimentSpec.resume`, and the policy-driven vision/MLP model choice.

Spec-level only — no sim is built, so these stay fast and deterministic.
"""

import pytest

from deepracer_genesis.configs.cfgs import get_env_cfg
from deepracer_genesis.experiment import (
    PPO,
    AsymmetricCameraPolicy,
    CameraEnvironment,
    Experiment,
    FeatureEnvironment,
    VectorPolicy,
)
from deepracer_genesis.experiment.builder import Builder
from deepracer_genesis.experiment.rsl_backend import _PPO_KEY_MAP, spec_to_train_cfg
from deepracer_genesis.physics.limits import MAX_SPEED, MIN_SPEED


def _feature_spec(**env_kw):
    return (FeatureEnvironment(num_envs=8, **env_kw) >> VectorPolicy()).build()


def _camera_spec(**env_kw):
    return (CameraEnvironment(num_envs=8, **env_kw)
            >> AsymmetricCameraPolicy()).build()


ENV_STAGES = pytest.mark.parametrize("make_spec", [_feature_spec, _camera_spec],
                                     ids=["feature", "camera"])


# ------------------------------------------------------------------ max_speed
@ENV_STAGES
def test_max_speed_threads_stage_to_spec_to_sim_cfg(make_spec):
    spec = make_spec(max_speed=1.5)
    assert spec.env.max_speed == 1.5
    assert Builder(spec).sim_cfg()["action"]["max_speed"] == 1.5


@ENV_STAGES
def test_max_speed_none_keeps_the_physics_default(make_spec):
    spec = make_spec()
    assert spec.env.max_speed is None
    cfg = Builder(spec).sim_cfg()
    assert cfg["action"]["max_speed"] == MAX_SPEED
    assert cfg["action"]["max_speed"] == get_env_cfg()["action"]["max_speed"]


@ENV_STAGES
def test_max_speed_leaves_min_speed_at_the_physics_default(make_spec):
    """Pins today's asymmetry: the spec caps the TOP of the speed range only, so
    the floor stays MIN_SPEED and nothing checks max_speed > min_speed."""
    cfg = Builder(make_spec(max_speed=1.5)).sim_cfg()
    assert cfg["action"]["min_speed"] == MIN_SPEED
    assert cfg["action"]["min_speed"] < cfg["action"]["max_speed"]


def test_max_speed_survives_the_spec_round_trip():
    assert _feature_spec(max_speed=2.25).to_dict()["env"]["max_speed"] == 2.25


# ---------------------------------------------------------- PPO schedule knobs
def _train_cfg(*, ppo=None, policy=None, env=None):
    pipe = (env or FeatureEnvironment(num_envs=8)) >> (policy or VectorPolicy())
    if ppo is not None:
        pipe = pipe >> ppo
    return spec_to_train_cfg(pipe.build())


def test_ppo_schedule_and_desired_kl_reach_the_algorithm_cfg():
    algo = _train_cfg(ppo=PPO(schedule="fixed", desired_kl=0.02))["algorithm"]
    assert algo["schedule"] == "fixed"
    assert algo["desired_kl"] == 0.02


def test_ppo_schedule_defaults_are_adaptive_kl():
    algo = _train_cfg(ppo=PPO())["algorithm"]
    assert algo["schedule"] == "adaptive"
    assert algo["desired_kl"] == 0.01


def test_every_ppo_stage_knob_is_mapped_for_rsl():
    """A knob added to the PPO stage but not to _PPO_KEY_MAP would be silently
    dropped; horizon is the one key mapped outside it (num_steps_per_env)."""
    ppo = (FeatureEnvironment(num_envs=8) >> VectorPolicy()
           >> PPO()).build().algorithm.ppo
    assert set(ppo) - {"horizon"} <= set(_PPO_KEY_MAP)
    assert _train_cfg(ppo=PPO(horizon=7))["num_steps_per_env"] == 7


# --------------------------------------------------------------------- resume
class _Baseline(Experiment):
    def pipeline(self):
        return FeatureEnvironment(num_envs=8) >> VectorPolicy()


class _FineTune(_Baseline):
    resume = "runs/abc/model_1500.pt"


def test_resume_survives_the_authoring_passthrough():
    assert _FineTune().spec().resume == "runs/abc/model_1500.pt"
    assert _Baseline().spec().resume is None


def test_resume_is_settable_as_an_experiment_override():
    assert _Baseline(resume="runs/x/model_10.pt").spec().resume == "runs/x/model_10.pt"


# ------------------------------------------------------- vision model dispatch
def test_camera_env_read_only_as_state_gets_the_mlp_model():
    """The model class follows the POLICY: a camera env whose policy reads only
    the state vector needs MLPModel, not the CNN."""
    cfg = _train_cfg(env=CameraEnvironment(num_envs=8),
                     policy=VectorPolicy(keys=("state",)))
    assert cfg["actor"]["class_name"] == "MLPModel"
    assert cfg["critic"]["class_name"] == "MLPModel"
    assert "cnn_cfg" not in cfg["actor"]
    assert cfg["obs_groups"] == {"actor": ["state"], "critic": ["state"]}


def test_camera_policy_reading_the_camera_gets_the_cnn_model():
    cfg = _train_cfg(env=CameraEnvironment(num_envs=8),
                     policy=AsymmetricCameraPolicy())
    assert cfg["actor"]["class_name"] == "CNNModel"
    assert cfg["actor"]["cnn_cfg"]["output_channels"]
    assert cfg["obs_groups"] == {"actor": ["camera"], "critic": ["camera", "state"]}


def test_feature_env_gets_the_mlp_model():
    assert _train_cfg()["actor"]["class_name"] == "MLPModel"
