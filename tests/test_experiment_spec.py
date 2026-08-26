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
    SafeRLCameraEnvironment,
    SafeRLFeatureEnvironment,
    SpecError,
    VectorPolicy,
    build,
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
    assert not spec.algorithm.requires_cost     # inferred: no cost signal


def test_pixel_noise_and_temporal_dr_flow_to_cfg():
    from deepracer_genesis.experiment.builder import Builder
    spec = (
        CameraEnvironment(render="madrona", resolution=(160, 120))
        >> DomainRandomizationCamera(pixel_noise=0.03, latency_steps=2, frame_drop=0.05)
        >> AsymmetricCameraPolicy(actor_keys=("camera",), critic_keys=("camera", "state"))
    ).build(seed=0)
    assert spec.obs_dr.pixel_noise == 0.03
    assert spec.obs_dr.image_aug["latency_steps"] == 2
    assert spec.obs_dr.image_aug["frame_drop"] == 0.05
    cfg = Builder(spec).sim_cfg()
    assert cfg["vision"]["pixel_noise"] == 0.03           # reachable from the DSL now


def test_pixel_noise_requires_camera_env():
    pipe = (
        FeatureEnvironment()
        >> DomainRandomizationCamera(pixel_noise=0.02)
        >> VectorPolicy(keys=("state",))
    )
    with pytest.raises(SpecError, match="camera env"):
        pipe.build(seed=0)


def test_env2_builds_expected_spec():
    spec = env2_pipeline().build(seed=0)
    assert spec.env.emits_cost is True
    assert spec.env.cost_fn == "offtrack_or_overspeed"
    assert spec.encoder.kind == "frozen_cnn"
    assert spec.encoder.out_key == "encoded"
    assert spec.policy.cnn is None              # vector policy downstream
    assert spec.policy.actor_keys == ("encoded", "state")
    assert spec.algorithm.requires_cost              # inferred from emits_cost
    assert spec.algorithm.lagrangian["budget"] == 25.0


def test_env2_as_class_idiom_matches_function_idiom():
    class _Env2(Experiment):       # not listed in EXPERIMENTS -> not registered
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
    assert c.id() != a.id()                     # seed IS configuration
    d = env1_pipeline().build(seed=0, total_env_steps=1)
    assert d.id() != a.id()                     # any config field changes id


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


def test_nyx_tiled_multitrack_allowed():
    # Tiled multi-track camera works under Nyx too (verified end to end by
    # scripts/verify_nyx_tiling.py): tiled variants are plain meshes on
    # separate world tiles, ordinary scene content for the path tracer. The
    # old blanket spec rejection only ever applied to the heterogeneous
    # (superimposed) path, which envs/scene.py still refuses at build time
    # when tiling is explicitly disabled. Building only WARNS (tiling cost).
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = (CameraEnvironment(render="nyx",
                                  tracks=("reinvent_base", "reInvent2019_track"))
                >> AsymmetricCameraPolicy()).build()
    assert spec.env.render == "nyx" and len(spec.env.tracks) == 2


def test_lagrangian_without_cost_env_rejected():
    with pytest.raises(SpecError, match="cost signal"):
        (FeatureEnvironment() >> VectorPolicy() >> PPOLagrangian(budget=5.0)).build()


def test_plain_ppo_on_cost_env_warns():
    with pytest.warns(UserWarning, match="unconstrained"):
        spec = (SafeRLFeatureEnvironment(budget=25.0)
                >> VectorPolicy() >> PPO()).build()
    assert not spec.algorithm.requires_cost


def test_explicit_lagrangian_budget_filled_from_env():
    spec = (SafeRLFeatureEnvironment(budget=42.0)
            >> VectorPolicy() >> PPOLagrangian()).build()
    assert spec.algorithm.lagrangian["budget"] == 42.0


# ------------------------------------------------- class-based run dispatch


class _FeatureExp(Experiment):
    """Inline Experiment class for the dispatch tests (no registry)."""
    def pipeline(self):
        return FeatureEnvironment(num_envs=8) >> VectorPolicy(keys=("state",))


class _SafeExp(Experiment):
    budget = 25.0

    def spec(self):
        return (SafeRLFeatureEnvironment(budget=self.budget)
                >> VectorPolicy(keys=("state",))).build(seed=self.seed)


def test_run_dispatches_class_and_instance():
    by_cls = run(_FeatureExp, build_only=True)
    by_inst = run(_FeatureExp(), build_only=True)
    assert by_cls == by_inst
    assert not by_cls.algorithm.requires_cost

    with_override = run(_FeatureExp, build_only=True, seed=3)
    assert with_override.seed == 3
    assert with_override.id() != by_cls.id()


def test_run_rejects_string_target():
    with pytest.raises(SpecError, match="referenced by class"):
        run("some_name", build_only=True)


def test_experiment_class_overrides_via_run():
    st = run(_SafeExp, build_only=True, budget=10.0, seed=2)
    assert st.algorithm.lagrangian["budget"] == 10.0
    assert st.seed == 2
    with pytest.raises(SpecError, match="unknown override"):
        run(_FeatureExp, build_only=True, nonexistent_field=1)


def test_id_ignores_bookkeeping_tags_and_is_cross_process_stable():
    import subprocess
    import sys

    a = env1_pipeline().build(seed=0, ablation_group="g1", variant="v1")
    b = env1_pipeline().build(seed=0, ablation_group="g2", variant="v2")
    assert a.id() == b.id()                     # tags are not configuration
    assert a.run_dir() != b.run_dir()           # but runs land separately

    code = ("from examples import CameraMadronaDr; "
            "from deepracer_genesis.experiment import run; "
            "print(run(CameraMadronaDr, build_only=True).id())")
    out = [subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True).stdout.strip() for _ in range(2)]
    assert out[0] == out[1] and len(out[0]) == 12   # sha1, not salted hash()


def test_run_accepts_spec_and_pipeline():
    spec = env1_pipeline().build()
    assert run(spec, build_only=True) == spec
    assert run(env1_pipeline(), build_only=True) == spec


def test_build_rejects_forgotten_build():
    def forgot():
        return env1_pipeline()          # returns a Pipeline, not a spec
    with pytest.raises(SpecError, match="forget"):
        build(forgot)                   # a factory that forgot to call .build()


def test_default_spec_invalid_without_stages():
    with pytest.raises(SpecError):
        ExperimentSpec().validate()


def test_random_direction_routes_to_env_spec_and_changes_id():
    from deepracer_genesis.experiment import FeatureEnvironment, VectorPolicy

    base = (FeatureEnvironment(num_envs=8) >> VectorPolicy()).build()
    both = (FeatureEnvironment(num_envs=8, random_direction=True)
            >> VectorPolicy()).build()
    assert base.env.random_direction is False
    assert both.env.random_direction is True
    assert base.id() != both.id()       # driving direction is configuration


def test_appearance_dr_routes_and_validates():
    from deepracer_genesis.experiment import (AsymmetricCameraPolicy,
                                              CameraEnvironment,
                                              DomainRandomizationTrackAppearance,
                                              FeatureEnvironment, VectorPolicy)

    spec = (CameraEnvironment(num_envs=8)
            >> DomainRandomizationTrackAppearance(strength=0.4)
            >> AsymmetricCameraPolicy(actor_keys=("camera",),
                                      critic_keys=("camera", "state"))).build()
    assert spec.obs_dr.appearance == {"world_color": 0.4}

    with pytest.raises(SpecError, match="camera"):      # feature env: no render
        (FeatureEnvironment(num_envs=8)
         >> DomainRandomizationTrackAppearance()
         >> VectorPolicy()).build()

    # Part O: multi-track camera (madrona) is now sound via spatial tiling —
    # it builds, but warns about the K× render/memory tax (benchmark gate)
    with pytest.warns(UserWarning, match="spatial tiling"):
        (CameraEnvironment(num_envs=8, tracks=("reinvent_base",
                                               "reInvent2019_track"))
         >> AsymmetricCameraPolicy(actor_keys=("camera",),
                                   critic_keys=("camera", "state"))).build()
    (FeatureEnvironment(num_envs=8, tracks=("reinvent_base",
                                            "reInvent2019_track"))
     >> VectorPolicy()).build()


def test_rsl_backend_maps_spec_and_gates_dispatch():
    from deepracer_genesis.experiment import (AsymmetricCameraPolicy,
                                              CameraEnvironment, PPO,
                                              VectorPolicy)
    from deepracer_genesis.experiment.rsl_backend import (rsl_supported,
                                                          spec_to_train_cfg)

    # feature + symmetric + continuous PPO => migrated (rsl-rl) scope
    spec = (FeatureEnvironment(num_envs=8) >> VectorPolicy()
            >> PPO(epochs=7, lr=1e-3, horizon=32, gae_lambda=0.9)).build()
    assert rsl_supported(spec)
    cfg = spec_to_train_cfg(spec)
    assert cfg["obs_groups"] == {"actor": ["state"], "critic": ["state"]}
    assert cfg["num_steps_per_env"] == 32                  # horizon -> num_steps_per_env
    assert cfg["algorithm"]["num_learning_epochs"] == 7    # epochs -> num_learning_epochs
    assert cfg["algorithm"]["learning_rate"] == 1e-3       # lr -> learning_rate
    assert cfg["algorithm"]["lam"] == 0.9                  # gae_lambda -> lam

    # camera + asymmetric critic is migrated too (obs_groups is native to rsl-rl)
    cam = (CameraEnvironment(num_envs=8)
           >> AsymmetricCameraPolicy(actor_keys=("camera",),
                                     critic_keys=("camera", "state"))).build()
    assert rsl_supported(cam)
    cam_cfg = spec_to_train_cfg(cam)
    assert cam_cfg["obs_groups"] == {"actor": ["camera"], "critic": ["camera", "state"]}

    # action-DR is migrated (applied env-side, not a TorchRL transform)
    from deepracer_genesis.experiment import (DomainRandomizationActions,
                                              SafeRLFeatureEnvironment)
    act_dr = (FeatureEnvironment(num_envs=8) >> VectorPolicy()
              >> DomainRandomizationActions(steer_noise=0.1, delay_steps=2)).build()
    assert rsl_supported(act_dr)

    # cost / frozen-CNN still route to the TorchRL Trainer (their phases pending)
    cost = (SafeRLFeatureEnvironment(num_envs=8, budget=5.0)
            >> VectorPolicy()).build()
    assert not rsl_supported(cost)


def test_track_width_dr_routes_and_defaults_off():
    from deepracer_genesis.experiment import VectorPolicy

    # explicit range routes to obs_dr.physics under the track_width_scale key
    spec = (FeatureEnvironment(num_envs=8)
            >> DomainRandomizationPhysics(track_width=(0.9, 1.15))
            >> VectorPolicy()).build()
    assert spec.obs_dr.physics["track_width_scale"] == (0.9, 1.15)

    # left unset, the knob is neutral (off) so existing DR runs are unchanged
    spec_off = (FeatureEnvironment(num_envs=8)
                >> DomainRandomizationPhysics()
                >> VectorPolicy()).build()
    assert spec_off.obs_dr.physics["track_width_scale"] == (1.0, 1.0)


def test_track_width_scale_in_catalog_as_geometry_layer():
    from deepracer_genesis.randomization.catalog import BY_NAME, by_layer

    knob = BY_NAME["track_width_scale"]
    assert knob.layer == "geometry"
    assert knob.cfg_key == "rand.track_width_scale"
    assert "half_width" in knob.signals
    assert knob in by_layer("geometry")


def test_discrete_action_space_routes_and_validates():
    from deepracer_genesis.experiment import FeatureEnvironment, VectorPolicy
    from deepracer_genesis.experiment.stages import (DomainRandomizationActions,
                                                     discrete_grid)

    grid_acts = discrete_grid(steer_bins=3, speed_bins=2)
    spec = (FeatureEnvironment(num_envs=8)
            >> VectorPolicy(actions=grid_acts)).build()
    assert len(spec.policy.actions) == 6
    assert all(len(a) == 2 for a in spec.policy.actions)

    with pytest.raises(SpecError, match="continuous"):   # action DR needs continuous
        (FeatureEnvironment(num_envs=8)
         >> VectorPolicy(actions=grid_acts)
         >> DomainRandomizationActions(steer_noise=0.02)).build()

    with pytest.raises(SpecError, match="pairs"):
        (FeatureEnvironment(num_envs=8)
         >> VectorPolicy(actions=((-2.0, 0.0), (1.0, 0.0)))).build()
