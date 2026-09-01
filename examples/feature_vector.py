"""Feature-vector examples: the fastest path, on CPU and on GPU with DR.

Each example is an ``Experiment`` subclass — hyperparameters are class
attributes, the pipeline is in ``pipeline()``, and you run it directly. The
structure every example follows (env -> obs choice -> DR -> policy -> algorithm
-> eval) is the ``>>`` chain returned by ``pipeline()``.

Run one::

    python examples/feature_vector.py                     # runs FeatureGpuDr
    python -m deepracer_genesis.experiment examples.feature_vector:FeatureCpu
    from examples.feature_vector import FeatureCpu; FeatureCpu().run()
"""

from deepracer_genesis.envs.features import SelectFeatures
from deepracer_genesis.experiment import (
    DomainRandomizationPhysics,
    Evaluation,
    Experiment,
    FeatureEnvironment,
    VectorPolicy,
)
from deepracer_genesis.randomization.spaces import FloatRange, SymRange

# a feature vector is an ordered selection of named blocks (envs/features.py)
POSE_ONLY = ("v_forward", "lateral", "heading")
FULL = ("v_forward", "v_lateral", "yaw_rate", "lateral", "heading",
        "last_action", "lookahead_xy")


class FeatureCpu(Experiment):
    """Feature vector on the CPU backend — no GPU needed (Part M).

    Blends: backend="cpu" + a pose-only feature selection + plain PPO. A small
    num_envs keeps CPU throughput reasonable.
    """

    total_env_steps = 2_000_000
    eval_every_steps = 500_000
    group = "examples"
    variant = "feature_cpu"

    def pipeline(self):
        return (
            FeatureEnvironment(num_envs=64, backend="cpu",
                               feature_set=SelectFeatures,
                               feature_params={"features": POSE_ONLY})
            >> VectorPolicy(keys=("state",))
        )


class FeatureGpuDr(Experiment):
    """Feature vector on GPU with physics domain randomization.

    Blends: backend="gpu" + the FULL feature vector + physics DR declared with
    the shared Space types (Part H) + an out-of-loop holdout eval (Part N).
    """

    total_env_steps = 5_000_000
    eval_every_steps = 1_000_000
    group = "examples"
    variant = "feature_gpu_dr"

    def pipeline(self):
        return (
            FeatureEnvironment(num_envs=1024, feature_set=SelectFeatures,
                               feature_params={"features": FULL})
            >> DomainRandomizationPhysics(friction=FloatRange(0.6, 1.4),
                                          mass=SymRange(0.2), com=SymRange(0.01))
            >> VectorPolicy(keys=("state",))
            >> Evaluation(real_tracks=("Oval_track",), charts=True)
        )


if __name__ == "__main__":
    FeatureGpuDr().run()
