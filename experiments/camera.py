"""Camera experiments — Env 1 of the plan (section 1.5)."""

from deepracer_genesis.experiment import (
    AsymmetricCameraPolicy,
    CameraEnvironment,
    DomainRandomizationActions,
    DomainRandomizationCamera,
    DomainRandomizationPhysics,
    DomainRandomizationTrackAppearance,
    experiment,
)


def _cam_chain(render="madrona", num_envs=128, dr=True):
    """The Env-1 pipeline; `dr=False` strips every DR stage."""
    chain = CameraEnvironment(render=render, resolution=(160, 120), num_envs=num_envs)
    if dr:
        if render == "madrona":
            # scene-level color/texture DR is heterogeneous-morph based
            chain = chain >> DomainRandomizationTrackAppearance(variants=16)
        chain = (chain
                 >> DomainRandomizationCamera(brightness=(0.7, 1.3), hue=0.05, blur=0.3,
                                              camera_jitter=True)
                 >> DomainRandomizationPhysics())
    chain = chain >> AsymmetricCameraPolicy(actor_keys=("camera",),
                                            critic_keys=("camera", "state"))
    if dr:
        chain = chain >> DomainRandomizationActions(steer_noise=0.02, speed_noise=0.05,
                                                    delay_steps=1)
    return chain


@experiment
def cam_baseline():
    """Env 1: end-to-end camera, asymmetric critic, full DR, unconstrained PPO."""
    return _cam_chain("madrona").build(seed=0, ablation_group="camera", variant="madrona")


@experiment
def cam_nyx():
    """Render-method ablation partner for cam_baseline (only render differs)."""
    return _cam_chain("nyx", num_envs=64).build(seed=0, ablation_group="camera",
                                                variant="nyx")


@experiment
def cam_full_dr():
    """dr_effect pairing: the full-DR treatment (same config as cam_baseline)."""
    return _cam_chain("madrona").build(seed=0, ablation_group="dr_effect",
                                       variant="full_dr")


@experiment
def cam_plain():
    """dr_effect pairing: the no-DR baseline."""
    return _cam_chain("madrona", dr=False).build(seed=0, ablation_group="dr_effect",
                                                 variant="no_dr")


@experiment
def cam_multitrack():
    """Heterogeneous training: each parallel env simulates + renders its own
    track (Genesis balanced block assignment across the three tracks)."""
    return (
        CameraEnvironment(render="madrona", resolution=(160, 120), num_envs=126,
                          tracks=("reinvent_base", "reInvent2019_track",
                                  "2022_reinvent_champ"))
        >> AsymmetricCameraPolicy(actor_keys=("camera",),
                                  critic_keys=("camera", "state"))
    ).build(seed=0, ablation_group="tracks", variant="hetero3")
