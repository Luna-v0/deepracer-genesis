"""Watch-it-live examples: the interactive Genesis viewer (Part M), one per
observation modality.

Both open the GUI viewer (``view="gui"``) so you watch the cars drive as they
train on the rsl-rl backend, with holdout eval + charts (Part N):

    WatchLiveFeature   GPU feature-vector env + viewer + physics DR
    WatchLiveCamera    GPU camera(render="madrona") env + viewer + camera DR,
                       asymmetric policy (actor=pixels, critic=pixels+state)

``view="gui"`` needs a display and a small ``num_envs`` — it is a debug/watch
path, not a throughput path. Run one::

    python examples/watch_live.py                                    # feature
    python -m deepracer_genesis.experiment examples.watch_live:WatchLiveCamera
"""

from deepracer_genesis.experiment import (
    AsymmetricCameraPolicy,
    CameraEnvironment,
    DomainRandomizationCamera,
    DomainRandomizationPhysics,
    Evaluation,
    Experiment,
    FeatureEnvironment,
    VectorPolicy,
)


class WatchLiveFeature(Experiment):
    """GPU feature env + interactive viewer + physics DR + holdout eval + charts.

    Attributes:
        num_envs: watchable in the viewer yet enough parallelism to learn and
            use the GPU (rsl-rl leaves plenty of headroom; see the perf review).
    """

    num_envs = 64
    seed = 0
    total_env_steps = 3_000_000
    eval_every_steps = 500_000
    group = "examples"
    variant = "watch_live_feature"

    def pipeline(self):
        return (
            FeatureEnvironment(num_envs=self.num_envs, backend="gpu",
                               view="gui",              # GPU + viewer (Part M)
                               realtime_factor=0)       # 0 = uncapped: run as fast as the GPU allows
            >> DomainRandomizationPhysics()
            >> VectorPolicy(keys=("state",))
            >> Evaluation(real_tracks=("reinvent_base", "Oval_track"),
                          eval_num_envs=32, charts=True)             # Part N
        )


class WatchLiveCamera(Experiment):
    """GPU camera env + interactive viewer + camera DR, asymmetric policy.

    The viewer shows the cars driving while the actor learns from the onboard
    RGB camera (critic also sees privileged state). A small ``num_envs`` keeps
    the viewer + per-env camera render watchable; ``realtime_factor=1`` paces it
    to real time so the motion is legible.

    Attributes:
        num_envs: small, so the camera render + viewer stay watchable.
    """

    num_envs = 16
    seed = 0
    total_env_steps = 3_000_000
    eval_every_steps = 500_000
    group = "examples"
    variant = "watch_live_camera"

    def pipeline(self):
        return (
            CameraEnvironment(render="madrona", resolution=(160, 120),
                              num_envs=self.num_envs, view="gui",     # camera + viewer
                              realtime_factor=1.0)      # real-time pacing so it's watchable
            >> DomainRandomizationCamera(brightness=(0.8, 1.2), hue=0.03)
            >> AsymmetricCameraPolicy(actor_keys=("camera",),
                                      critic_keys=("camera", "state"))
            >> Evaluation(charts=True)                                 # Part N charts
        )


#: back-compat alias — the original WatchLive was the feature-vector variant
WatchLive = WatchLiveFeature


if __name__ == "__main__":
    WatchLiveFeature().run()
