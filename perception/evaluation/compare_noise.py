"""Two trainings: perfect perception, then the CNN's error.

The two ends of ablation.py on their own, when only the headline number is
wanted.

    caffeinate -i .venv/bin/python -m perception.evaluation.compare_noise
"""

import warnings

warnings.filterwarnings("ignore")

from deepracer_genesis.experiment import run

from perception.train_policy_with_noise import NoisyPerceptionPolicy


def main():
    for noise in (0.0, 1.0):
        record = run(NoisyPerceptionPolicy, root="runs", noise=noise)
        m = record.metrics
        print(f"\nnoise={noise}  completion {m['completion_rate']:.2f}"
              f"  progress {m['mean_progress_m']:.1f} m"
              f"  speed {m['mean_speed_mps']:.2f} m/s"
              f"  offtrack {m['offtrack_rate']:.2f}", flush=True)


if __name__ == "__main__":
    main()
