"""Policy fine-tuned through the frozen perception CNN.

Only the source of the seven camera-recoverable channels changes, so whatever
this run costs is the cost of learned perception.
"""

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from deepracer_genesis.experiment import (
    PPO,
    CameraEnvironment,
    Evaluation,
    Experiment,
    VectorPolicy,
    run,
)

from deepracer_genesis.perception.features import CNNPerceptionFeatures

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = REPO_ROOT / "runs" / "cnn"


class CNNPerceptionPolicy(Experiment):
    seed = 0
    # start from a policy trained on the simulator's exact values. Measured, the
    # CNN's error costs 0-1% on tracks the policy can already drive, so there is
    # little for this run to absorb: it demonstrates that a policy survives being
    # driven by the camera, it does not make it better.
    resume = str(CHECKPOINTS / "reference_policy.pt")
    total_env_steps = 3_000_000
    eval_every_steps = 300_000
    ablation_group = "cnn"
    variant = "cnn_perception"

    # 6 tracks spanning the whole range of corner severity, from gentlest to
    # tightest: it is severity that decides off-tracks (correlation +0.76)
    tracks = ("2022_march_open",       # k_std 0.10  very easy
              "arctic_open",           #       0.18  easy
              "jyllandsringen_open",   #       0.22  medium
              "hamption_pro",          #       0.25  hard
              "thunder_hill_pro",      #       0.27  very hard
              "Tokyo_Training_track")  #       0.42  extreme
    # sized for a 10-core laptop; raise it on a bigger box, the loop fits in
    # one core and 1.4 GB
    num_envs = 64
    max_speed = 2.0
    lr = 3e-5          # we are refining a policy that already drives
    # the adaptive schedule retunes lr from the measured KL; on 96 samples per
    # minibatch that KL is noise and lr random-walks up to its 1e-2 cap, which
    # destroys the resumed policy. Pin it.
    schedule = "fixed"
    feature_set = CNNPerceptionFeatures   # PerceptionFeatures = perfect perception
    # the checkpoint is explicit on purpose: a missing or stale one would
    # silently turn this into a perfect-perception run
    feature_params = {"checkpoint": str(CHECKPOINTS / "perception_jittered.pt")}

    def pipeline(self):
        return (
            CameraEnvironment(
                feature_set=self.feature_set,
                feature_params=self.feature_params,
                resolution=(160, 120),
                frame_stack=4,
                tracks=self.tracks,
                num_envs=self.num_envs,
                backend="cpu",          # "gpu" where Madrona is available
                max_speed=self.max_speed,
            )
            >> VectorPolicy(keys=("state",))
            >> PPO(lr=self.lr, schedule=self.schedule)
            >> Evaluation(real_tracks=self.tracks, eval_num_envs=8)
        )


if __name__ == "__main__":
    record = run(CNNPerceptionPolicy, root="runs")
    m = record.metrics
    print(f"\ncompletion {m['completion_rate']:.3f}  progress {m['mean_progress_m']:.1f} m"
          f"  offtrack {m['offtrack_rate']:.2f}")
