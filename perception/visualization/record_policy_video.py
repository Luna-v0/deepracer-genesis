"""Top-down videos of the policies trained with simulated CNN error.

    python -m perception.visualization.record_policy_video           # both noise levels, test tracks
    python -m perception.visualization.record_policy_video Monaco    # one track
"""

import sys
import warnings

warnings.filterwarnings("ignore")

from deepracer_genesis.experiment.visualize import rollout_video

from perception.train_policy_with_noise import TEST_TRACKS, NoisyPerceptionPolicy

NOISE_LEVELS = (0.0, 1.0)


def main():
    tracks = (sys.argv[1],) if len(sys.argv) > 1 else TEST_TRACKS
    for noise in NOISE_LEVELS:
        for track in tracks:
            try:
                out = rollout_video(NoisyPerceptionPolicy, root="runs", track=track,
                                    steps=1500, num_envs=1,
                                    out=f"runs/videos/noise{noise}", noise=noise)
                print(f"noise={noise} {track:16} -> {out}", flush=True)
            except FileNotFoundError as e:
                print(f"noise={noise} {track:16} -> no checkpoint ({e})")


if __name__ == "__main__":
    main()
