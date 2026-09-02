"""A series of trainings to locate where the performance is lost.

Each run keeps the exact simulator values but corrupts a different subset of the
camera channels with the CNN's measured error, so the drop can be attributed to
"where am I" (lateral, heading) or to "what is ahead" (the two curvatures).

    caffeinate -i .venv/bin/python -m perception.evaluation.ablation
"""

import warnings

warnings.filterwarnings("ignore")

from deepracer_genesis.experiment import run

from perception.train_policy_with_noise import NoisyPerceptionPolicy

RUNS = (
    ("reference",        dict(noise=0.0)),
    ("full cnn error",   dict(noise=1.0)),
    ("half the error",   dict(noise=0.5)),
    ("where am i",       dict(noise=1.0, noise_channels=("lateral", "heading"))),
    ("what is ahead",    dict(noise=1.0, noise_channels=("curv@1m", "curv@3m"))),
    ("oval only",        dict(noise=1.0, tracks=("Oval_track",),
                              test_tracks=("Oval_track",))),
)


def main():
    for name, settings in RUNS:
        record = run(NoisyPerceptionPolicy, root="runs", **settings)
        m = record.metrics
        print(f"\n>>> {name:20} completion {m['completion_rate']:.3f}"
              f"  progress {m['mean_progress_m']:.1f} m"
              f"  offtrack {m['offtrack_rate']:.2f}", flush=True)


if __name__ == "__main__":
    main()
