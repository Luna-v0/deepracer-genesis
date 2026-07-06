"""Safe-RL experiments — Env 2 of the plan (representation transfer + CMDP)."""

from deepracer_genesis.experiment import (
    DomainRandomizationActions,
    DomainRandomizationCamera,
    Experiment,
    FrozenCNNToFeatureVector,
    SafeRLCameraEnvironment,
    SafeRLFeatureEnvironment,
    VectorPolicy,
    experiment,
)


@experiment
def safe_feature():
    """Phase-4 development target: PPO-Lagrangian on the (fast) feature env."""
    return (
        SafeRLFeatureEnvironment(cost="offtrack", budget=25.0, num_envs=1024)
        >> VectorPolicy(keys=("state",))
    ).build(seed=0, ablation_group="safety", variant="feature_lagrangian")


class SafeTransfer(Experiment):
    """Env 2: Safe-RL camera env, DR, frozen-CNN -> vector transfer, PPO-Lagrangian.

    Isolates two questions at once: do frozen visual features suffice vs.
    end-to-end vision, and does the constraint hold on transferred
    representations?
    """

    render = "madrona"
    budget = 25.0
    ckpt = "runs/camera/madrona-0/best.pt"   # a cam_baseline checkpoint
    seed = 0

    def spec(self):
        return (
            SafeRLCameraEnvironment(render=self.render,
                                    cost="offtrack_or_overspeed", budget=self.budget)
            >> DomainRandomizationCamera(brightness=(0.7, 1.3))
            >> FrozenCNNToFeatureVector(checkpoint=self.ckpt, output_dim=256)
            >> VectorPolicy(keys=("encoded", "state"))
            >> DomainRandomizationActions(steer_noise=0.02)
        ).build(seed=self.seed, ablation_group="safety", variant=f"transfer_{self.render}")


class SafeTransferNyx(SafeTransfer):
    render = "nyx"


class SafeTransferTight(SafeTransfer):
    budget = 10.0
