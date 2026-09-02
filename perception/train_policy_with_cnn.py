"""Policy trained through the real perception CNN.

    caffeinate -i .venv/bin/python -m perception.train_policy_with_cnn

The env renders the camera, the frozen CNN turns it into the 7 channels, and PPO
fine-tunes a policy that already knows how to drive on the exact values. Nothing
else changes, so whatever the run costs is the cost of the learned perception.
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

from perception.cnn_features import CNNPerceptionFeatures

REPO_ROOT = Path(__file__).resolve().parent.parent


class CNNPerceptionPolicy(Experiment):
    seed = 0
    # start from a policy trained on the simulator's exact values. Measured, the
    # CNN's error costs 0-1% on tracks the policy can already drive, so there is
    # little for this run to absorb: it demonstrates that a policy survives being
    # driven by the camera, it does not make it better.
    resume = str(REPO_ROOT / "perception" / "reference_policy.pt")
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
    # mac: 883 steps/s measured at 16 envs. The loop fits in one core out of ten
    # and 1.4 GB out of 16, so raising the env count costs no wall time and takes
    # the PPO batch from 384 to 1536 samples per update. Raise it on a bigger box.
    num_envs = 64
    max_speed = 2.0
    lr = 3e-5          # we are refining a policy that already drives
    # the adaptive schedule retunes lr from the measured KL; on 96 samples per
    # minibatch that KL is noise and lr random-walks up to its 1e-2 cap, which
    # destroys the resumed policy. Pin it.
    schedule = "fixed"
    feature_set = CNNPerceptionFeatures   # PerceptionFeatures = perfect perception
    feature_params = None                 # e.g. {"cnn_device": "cpu"}

    def pipeline(self):
        return (
            CameraEnvironment(
                feature_set=self.feature_set,
                feature_params=self.feature_params,
                resolution=(160, 120),
                frame_stack=4,
                tracks=self.tracks,
                num_envs=self.num_envs,
                backend="cpu",          # mac: no Madrona; "gpu" on CUDA
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
