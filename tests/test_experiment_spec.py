"""Phase-0 unit tests (plan section 7): the declaration core, no torch."""

import json

import pytest

from deepracer_genesis.experiment import (
    PPO,
    AsymmetricCameraPolicy,
    AsymmetricVectorPolicy,
    CameraEnvironment,
    DomainRandomizationActions,
    DomainRandomizationCamera,
    DomainRandomizationPhysics,
    Experiment,
    ExperimentSpec,
    FeatureEnvironment,
    FrozenCNNToFeatureVector,
    PPOLagrangian,
    REGISTRY,
    SafeRLCameraEnvironment,
    SafeRLFeatureEnvironment,
    SpecError,
    VectorPolicy,
    build,
    experiment,
    run,
)

# ----------------------------------------------------------------- helpers

def env1_pipeline():
    """Plan section 1.5, Env 1 — as a raw pipeline."""
    return (
        CameraEnvironment(render="madrona", resolution=(160, 120))
        >> DomainRandomizationCamera(brightness=(0.7, 1.3), hue=0.05, blur=0.3,
                                     camera_jitter=True)
        >> AsymmetricCameraPolicy(actor_keys=("camera",), critic_keys=("camera", "state"))
        >> DomainRandomizationActions(steer_noise=0.02, speed_noise=0.05, delay_steps=1)
    )


def env2_pipeline(budget=25.0, ckpt="runs/x/best.pt"):
    """Plan section 1.5, Env 2."""
    return (
        SafeRLCameraEnvironment(render="madrona", cost="offtrack_or_overspeed",
                                budget=budget)
        >> DomainRandomizationCamera(brightness=(0.7, 1.3))
        >> FrozenCNNToFeatureVector(checkpoint=ckpt, output_dim=256)
        >> VectorPolicy(keys=("encoded", "state"))
        >> DomainRandomizationActions(steer_noise=0.02)
    )


# --------------------------------------------------------------- Env 1 & 2

def test_env1_builds_expected_spec():
    spec = env1_pipeline().build(seed=0)
    assert spec.env.modality == "camera"
    assert spec.env.render == "madrona"
    assert spec.env.emits_cost is False
    assert spec.obs_dr.image_aug["brightness"] == (0.7, 1.3)
    assert spec.obs_dr.camera_jitter            # jitter=True expands to defaults
    assert spec.policy.actor_keys == ("camera",)
    assert spec.policy.critic_keys == ("camera", "state")
    assert spec.policy.cnn is not None
    assert spec.action_dr.delay_steps == 1
    assert spec.algorithm.kind == "ppo"         # inferred: no cost signal


def test_env2_builds_expected_spec():
    spec = env2_pipeline().build(seed=0)
    assert spec.env.emits_cost is True
    assert spec.env.cost_fn == "offtrack_or_overspeed"
    assert spec.encoder.kind == "frozen_cnn"
    assert spec.encoder.out_key == "encoded"
    assert spec.policy.cnn is None              # vector policy downstream
    assert spec.policy.actor_keys == ("encoded", "state")
    assert spec.algorithm.kind == "ppo_lagrangian"   # inferred from emits_cost
    assert spec.algorithm.lagrangian["budget"] == 25.0


def test_env2_as_class_idiom_matches_function_idiom():
    class _Env2(Experiment):       # leading _ opts out of the registry
        budget = 25.0
        def spec(self):
            return env2_pipeline(budget=self.budget).build(seed=0)

    assert _Env2().spec() == env2_pipeline().build(seed=0)
    assert _Env2(budget=10.0).spec().algorithm.lagrangian["budget"] == 10.0
    with pytest.raises(AttributeError):
        _Env2(nonexistent=1)


# ------------------------------------------------------------ identity/hash

def test_id_stable_and_config_sensitive():
    a = env1_pipeline().build(seed=0)
    b = env1_pipeline().build(seed=0)
    assert a == b and a.id() == b.id()
    c = env1_pipeline().build(seed=1)
    assert c.id() != a.id()
    d = env1_pipeline().build(seed=0, variant="v2")
    assert d.id() != a.id()


def test_to_dict_json_serializable_and_run_dir():
    spec = env2_pipeline().build(seed=3, ablation_group="safety", variant="tight")
    json.dumps(spec.to_dict())                      # must not raise
    assert spec.run_dir() == f"runs/safety/tight-3-{spec.id()}"


# -------------------------------------------------------------- validation

def test_pipeline_must_start_with_environment():
    with pytest.raises(SpecError, match="first stage"):
        (VectorPolicy() >> FeatureEnvironment()).build()


def test_exactly_one_policy():
    with pytest.raises(SpecError, match="exactly one Policy"):
        (FeatureEnvironment() >> VectorPolicy() >> VectorPolicy()).build()
    with pytest.raises(SpecError, match="exactly one Policy"):
        Pipeline_no_policy = FeatureEnvironment() >> DomainRandomizationActions()
        Pipeline_no_policy.build()


def test_frozen_cnn_requires_camera_env():
    with pytest.raises(SpecError, match="camera env"):
        (FeatureEnvironment()
         >> FrozenCNNToFeatureVector(checkpoint="x.pt")
         >> VectorPolicy(keys=("encoded", "state"))).build()


def test_frozen_cnn_requires_vector_policy():
    with pytest.raises(SpecError, match="vector"):
        (CameraEnvironment()
         >> FrozenCNNToFeatureVector(checkpoint="x.pt")
         >> AsymmetricCameraPolicy()).build()


def test_asymmetry_requires_critic_superset():
    with pytest.raises(SpecError, match="critic_keys"):
        (CameraEnvironment()
         >> AsymmetricCameraPolicy(actor_keys=("camera", "state"),
                                   critic_keys=("camera",))).build()


def test_unknown_key_rejected():
    with pytest.raises(SpecError, match="not produced"):
        (FeatureEnvironment()
         >> AsymmetricVectorPolicy(actor_keys=("state",),
                                   critic_keys=("state", "privileged"))).build()


def test_vector_policy_cannot_eat_raw_camera():
    with pytest.raises(SpecError, match="raw 'camera'"):
        (CameraEnvironment() >> VectorPolicy(keys=("camera",))).build()


def test_nyx_heterogeneous_rejected():
    with pytest.raises(SpecError, match="Madrona-only"):
        (CameraEnvironment(render="nyx",
                           tracks=("reinvent_base", "reInvent2019_track"))
         >> AsymmetricCameraPolicy()).build()
    # same combination on madrona is fine
    (CameraEnvironment(render="madrona",
                       tracks=("reinvent_base", "reInvent2019_track"))
     >> AsymmetricCameraPolicy()).build()


def test_lagrangian_without_cost_env_rejected():
    with pytest.raises(SpecError, match="cost signal"):
        (FeatureEnvironment() >> VectorPolicy() >> PPOLagrangian(budget=5.0)).build()


def test_plain_ppo_on_cost_env_warns():
    with pytest.warns(UserWarning, match="unconstrained"):
        spec = (SafeRLFeatureEnvironment(budget=25.0)
                >> VectorPolicy() >> PPO()).build()
    assert spec.algorithm.kind == "ppo"


def test_explicit_lagrangian_budget_filled_from_env():
    spec = (SafeRLFeatureEnvironment(budget=42.0)
            >> VectorPolicy() >> PPOLagrangian()).build()
    assert spec.algorithm.lagrangian["budget"] == 42.0


# ---------------------------------------------------------- registry / run

def test_registry_and_run_dispatcher():
    import experiments  # noqa: F401  registrations fire

    assert "cam_baseline" in REGISTRY
    assert "SafeTransfer" in REGISTRY

    by_name = run("cam_baseline", build_only=True)
    by_fn = run(REGISTRY["cam_baseline"], build_only=True)
    assert by_name == by_fn
    assert by_name.algorithm.kind == "ppo"

    with_override = run("cam_baseline", build_only=True, seed=3)
    assert with_override.seed == 3
    assert with_override.id() != by_name.id()

    st = run("SafeTransferTight", build_only=True)
    assert st.algorithm.lagrangian["budget"] == 10.0

    with pytest.raises(SpecError, match="unknown experiment"):
        run("does_not_exist", build_only=True)


def test_run_accepts_spec_and_pipeline():
    spec = env1_pipeline().build()
    assert run(spec, build_only=True) == spec
    assert run(env1_pipeline(), build_only=True) == spec


def test_build_rejects_forgotten_build():
    @experiment(name="_forgot_build")
    def forgot():
        return env1_pipeline()          # returns a Pipeline, not a spec
    with pytest.raises(SpecError, match="forget"):
        build("_forgot_build")


def test_default_spec_invalid_without_stages():
    with pytest.raises(SpecError):
        ExperimentSpec().validate()
