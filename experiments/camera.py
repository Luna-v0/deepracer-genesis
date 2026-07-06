"""Camera experiments — Env 1 of the plan (section 1.5)."""

from deepracer_genesis.experiment import (
    AsymmetricCameraPolicy,
    CameraEnvironment,
    DomainRandomizationActions,
    DomainRandomizationCamera,
    experiment,
)


@experiment
def cam_baseline():
    """Env 1: end-to-end camera, asymmetric critic, full DR, unconstrained PPO."""
    return (
        CameraEnvironment(render="madrona", resolution=(160, 120))
        >> DomainRandomizationCamera(brightness=(0.7, 1.3), hue=0.05, blur=0.3,
                                     camera_jitter=True)
        >> AsymmetricCameraPolicy(actor_keys=("camera",), critic_keys=("camera", "state"))
        >> DomainRandomizationActions(steer_noise=0.02, speed_noise=0.05, delay_steps=1)
    ).build(seed=0, ablation_group="camera", variant="madrona")


@experiment
def cam_nyx():
    """Render-method ablation partner for cam_baseline."""
    return (
        CameraEnvironment(render="nyx", resolution=(160, 120), num_envs=64)
        >> DomainRandomizationCamera(brightness=(0.7, 1.3), hue=0.05, blur=0.3)
        >> AsymmetricCameraPolicy(actor_keys=("camera",), critic_keys=("camera", "state"))
        >> DomainRandomizationActions(steer_noise=0.02, speed_noise=0.05, delay_steps=1)
    ).build(seed=0, ablation_group="camera", variant="nyx")


@experiment
def cam_plain():
    """cam_baseline stripped of every DR stage (no_dr vs full_dr pairing)."""
    return (
        CameraEnvironment(render="madrona", resolution=(160, 120))
        >> AsymmetricCameraPolicy(actor_keys=("camera",), critic_keys=("camera", "state"))
    ).build(seed=0, ablation_group="dr_effect", variant="no_dr")
