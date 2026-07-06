"""Single-file experiment template — copy me.

One class defines EVERYTHING about a run: the hyperparameters (as class
attributes — a variant is a subclass or `MyExperiment(budget=10)`), the
env / DR / policy pipeline (the `>>` chain), and how long to train / how
often to evaluate. Then it just runs — no command line needed:

    uv run experiments/template.py                 # this file, directly
    run(MyExperiment, seed=3)                      # or from any Python code
    python -m deepracer_genesis.experiment MyExperiment   # CLI, if you want it
"""

from deepracer_genesis.experiment import (
    DomainRandomizationActions,
    DomainRandomizationCamera,
    DomainRandomizationPhysics,
    AsymmetricCameraPolicy,
    CameraEnvironment,
    Experiment,
    run,
)


class MyExperiment(Experiment):
    # ---- training configuration ----------------------------------------
    seed = 0
    total_env_steps = 10_000_000
    eval_every_steps = 1_000_000        # deterministic eval every 1M steps
    ablation_group = "my_study"
    variant = "baseline"

    # ---- experiment-specific hyperparameters ----------------------------
    render = "madrona"                  # or "nyx"
    num_envs = 128
    brightness = (0.7, 1.3)
    action_delay = 1

    # ---- the pipeline ----------------------------------------------------
    def pipeline(self):
        return (
            CameraEnvironment(render=self.render, resolution=(160, 120),
                              num_envs=self.num_envs)
            >> DomainRandomizationCamera(brightness=self.brightness, hue=0.05,
                                         blur=0.3, camera_jitter=True)
            >> DomainRandomizationPhysics()
            >> AsymmetricCameraPolicy(actor_keys=("camera",),
                                      critic_keys=("camera", "state"))
            >> DomainRandomizationActions(steer_noise=0.02,
                                          delay_steps=self.action_delay)
        )


class MyExperimentNoDelay(MyExperiment):
    variant = "no_delay"
    action_delay = 0


if __name__ == "__main__":
    run(MyExperiment)
    # run(MyExperimentNoDelay)        # queue variants in the same file
