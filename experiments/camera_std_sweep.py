"""Diagnostic sweep: stop the camera policy's action-std explosion.

Evidence (runs/examples/camera_madrona_dr-0-33bbaf4cc441): std grows
monotonically from ~iter 100 (1.3) to 18.3 at iter 2600 while deterministic
eval stays at 0% completion. Hypothesis: entropy_coef=0.01 pays +log(std) per
dim with nothing opposing it — the vision policy gradient is too weak to pull
std back. Short runs (~600 iters) are enough to see whether the drift stops;
the failed run had std>4 by iter 500.

Run: uv run python experiments/camera_std_sweep.py
"""

import sys

sys.path.insert(0, ".")   # examples.* lives at the project root
from examples.camera import CameraMadronaDr  # noqa: E402

from deepracer_genesis.experiment import PPO  # noqa: E402


class _ShortCamera(CameraMadronaDr):
    total_env_steps = 2_500_000
    eval_every_steps = 1_250_000
    group = "camera_std_sweep"


class SweepEntropyZero(_ShortCamera):
    """No entropy bonus at all: if std stays ~1, entropy was the whole story."""
    variant = "entropy_0"

    def pipeline(self):
        return super().pipeline() >> PPO(entropy_coef=0.0)


class SweepEntropySmall(_ShortCamera):
    """10x smaller bonus: keeps some exploration pressure if zero over-corrects."""
    variant = "entropy_1e3"

    def pipeline(self):
        return super().pipeline() >> PPO(entropy_coef=0.001)


class SweepStdCeiling(_ShortCamera):
    """Round-2 fix: keep the default entropy bonus but clamp std to [0.1, 1.0].

    Round 1 showed the unbounded std is bistable (0.01 -> 18+, <=0.001 ->
    0.02-0.06). A ceiling gives the entropy term a stable fixed point: std
    rides up to 1.0 and pins there — constant healthy exploration; the floor
    prevents premature determinism.
    """
    variant = "std_ceiling"

    def pipeline(self):
        # A pipeline allows exactly ONE policy stage, so restate
        # CameraMadronaDr's pipeline with the distribution override in place
        # (keep in sync with examples/camera.py).
        from deepracer_genesis.experiment import (
            AsymmetricCameraPolicy,
            CameraEnvironment,
            DomainRandomizationActions,
            DomainRandomizationCamera,
            DomainRandomizationPhysics,
            DomainRandomizationTrackAppearance,
        )
        return (
            CameraEnvironment(render="madrona", resolution=(160, 120), num_envs=128)
            >> DomainRandomizationTrackAppearance(strength=0.6)
            >> DomainRandomizationCamera(brightness=(0.7, 1.3), hue=0.05, blur=0.3,
                                         camera_jitter=True)
            >> DomainRandomizationPhysics()
            >> AsymmetricCameraPolicy(actor_keys=("camera",),
                                      critic_keys=("camera", "state"),
                                      distribution={"std_range": (0.1, 1.0)})
            >> DomainRandomizationActions(steer_noise=0.02, speed_noise=0.05,
                                          delay_steps=1)
        )


class SweepStackStd(_ShortCamera):
    """Round 3: std ceiling + 4-frame stack.

    Round 2 fixed the std dynamics (pins at 1.0) but eval stayed at ~1.9 m
    progress — consistent with the actor being VELOCITY-BLIND: a single frame
    carries no ego-motion, while the critic privately sees velocities. Stack 4
    frames along channels (oldest first, prime by repeat — the deployment
    contract) so the actor can observe its own dynamics.
    """
    variant = "stack4_stdceil"

    def pipeline(self):
        from deepracer_genesis.experiment import (
            AsymmetricCameraPolicy,
            CameraEnvironment,
            DomainRandomizationActions,
            DomainRandomizationCamera,
            DomainRandomizationPhysics,
            DomainRandomizationTrackAppearance,
        )
        return (
            # 48 envs + 8 minibatches: the rollout storage holds the STACKED
            # obs (24 x N x 12 x 120 x 160 f32) AND each minibatch indexes a
            # dense copy of it — at N=128/mb=4 that is 2.8 GB + 0.35 GB copies
            # next to Genesis's ~5.4 GB, an OOM on 8 GB. Run with
            # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
            CameraEnvironment(render="madrona", resolution=(160, 120),
                              num_envs=48, frame_stack=4)
            >> DomainRandomizationTrackAppearance(strength=0.6)
            >> DomainRandomizationCamera(brightness=(0.7, 1.3), hue=0.05, blur=0.3,
                                         camera_jitter=True)
            >> DomainRandomizationPhysics()
            >> AsymmetricCameraPolicy(actor_keys=("camera",),
                                      critic_keys=("camera", "state"),
                                      distribution={"std_range": (0.1, 1.0)})
            >> DomainRandomizationActions(steer_noise=0.02, speed_noise=0.05,
                                          delay_steps=1)
            >> PPO(minibatches=8)   # smaller dense obs copies per update
        )


class SweepNoDr(_ShortCamera):
    """Round 4: same policy/algorithm as round 3, ZERO domain randomization.

    Decisive split: if the clean env learns to steer in 2.5M steps, the full
    DR stack (world-color remap + photometric + physics + action delay from
    step zero) is what stalls learning at this budget -> answer is curriculum
    or budget. If this ALSO plateaus at ~1.9 m, the problem is structural
    (reward/architecture/perception), not difficulty.
    """
    variant = "nodr_stack4"

    def pipeline(self):
        from deepracer_genesis.experiment import (
            AsymmetricCameraPolicy,
            CameraEnvironment,
            PPO,
        )
        return (
            CameraEnvironment(render="madrona", resolution=(160, 120),
                              num_envs=48, frame_stack=4)
            >> AsymmetricCameraPolicy(actor_keys=("camera",),
                                      critic_keys=("camera", "state"),
                                      distribution={"std_range": (0.1, 1.0)})
            >> PPO(minibatches=8)
        )


_ALL = {c.variant: c for c in (SweepEntropyZero, SweepEntropySmall,
                               SweepStdCeiling, SweepStackStd, SweepNoDr)}

if __name__ == "__main__":
    # Sequential on purpose: one 8 GB GPU. Pass variant names to select.
    names = sys.argv[1:] or ["entropy_0", "entropy_1e3"]
    for name in names:
        print(f"\n=== {name} ===", flush=True)
        _ALL[name]().run()
