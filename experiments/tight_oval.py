"""Train a forward-driving policy on the tight home-room oval (``donut_track``).

Feasibility probe for a 1.5 x 1.6 m room track: the centerline turn radius
(~0.40 m) sits above the car's full-lock minimum (~0.284 m), and a privileged
P-controller already laps it forward, so this checks that a *learned* policy
can complete it forward — no reverse required. Holdout eval runs on the same
track to report lap completion.

Run::

    python -m deepracer_genesis.experiment experiments.tight_oval:TightOval
"""

from deepracer_genesis.envs.features import SelectFeatures
from deepracer_genesis.experiment import (
    AsymmetricCameraPolicy,
    CameraEnvironment,
    DomainRandomizationCamera,
    Evaluation,
    Experiment,
    FeatureEnvironment,
    VectorPolicy,
)

FULL = ("v_forward", "v_lateral", "yaw_rate", "lateral", "heading",
        "last_action", "lookahead_xy")


class TightOval(Experiment):
    """Continuous-control PPO on ``donut_track``, forward direction only.

    Blends: GPU feature-vector env on the single generated oval + the FULL
    feature vector + a shared-encoder MLP policy + holdout eval on the same
    track to read out lap completion.
    """

    total_env_steps = 2_000_000
    eval_every_steps = 500_000
    group = "tight_oval"
    variant = "tight_oval_fwd"

    def pipeline(self):
        return (
            FeatureEnvironment(num_envs=1024, tracks=("donut_track",),
                               feature_set=SelectFeatures,
                               feature_params={"features": FULL})
            >> VectorPolicy(keys=("state",))
            >> Evaluation(real_tracks=("donut_track",), charts=True)
        )


class TightOvalLive(Experiment):
    """Watch cars learn to lap ``donut_track`` live in the interactive viewer.

    Same env/policy as :class:`TightOval`, but ``view="gui"`` opens a Genesis
    window and ``realtime_factor=1`` paces it to real time so it is watchable.
    A fresh policy trains from scratch, so you see the cars flail and then
    improve. Run it from a terminal with a display and Ctrl-C when you have
    seen enough::

        python -m deepracer_genesis.experiment experiments.tight_oval:TightOvalLive

    Small ``num_envs`` keeps the window legible; raise ``realtime_factor`` to
    fast-forward, or set it to 0 to run uncapped (learns in seconds, too fast
    to watch).
    """

    num_envs = 512
    total_env_steps = 10_000_000
    eval_every_steps = 1_000_000
    group = "tight_oval"
    variant = "tight_oval_live"

    def pipeline(self):
        return (
            CameraEnvironment(render="nyx", resolution=(160, 120), num_envs=64,
                              tracks=("donut_track",))
            >> DomainRandomizationCamera(brightness=(0.8, 1.2), hue=0.03)
            >> AsymmetricCameraPolicy(actor_keys=("camera",),
                                      critic_keys=("camera", "state"))
                >> Evaluation(charts=True, real_tracks=("donut_track",), gui=True)                              # Part N charts
        )


if __name__ == "__main__":
    TightOvalLive().run()
