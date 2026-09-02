"""Collect one track's worth of camera frames and their ground-truth targets.

    python -m perception.data_generation <track>

Writes ``data/<track>_v2/rollout_*.parquet``, one row per frame: the JPEG image,
the full feature vector it is paired with, and the env/episode ids the dataset
needs to know which frames may be stacked together.
"""

import sys
import warnings
from pathlib import Path

from deepracer_genesis.datasets.rollout import collect_rollout_dataset
from deepracer_genesis.envs.features import PerceptionFeatures
from deepracer_genesis.experiment import CameraEnvironment
from deepracer_genesis.experiment.stages import DomainRandomizationTrackAppearance

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
STEPS = 1500        # one full episode (30 s)


def main():
    track = sys.argv[1]
    world = (CameraEnvironment(
        backend="cpu",      # mac: no Madrona, so the CPU rasterizer; "gpu" on CUDA
        resolution=(160, 120),
        num_envs=32,        # 32 appearances and 32 starting points per track;
                            # the batched camera renders them in a single call
        tracks=(track,),
        feature_set=PerceptionFeatures,
        random_start=True,
    ) >> DomainRandomizationTrackAppearance(strength=0.6))

    collect_rollout_dataset(world, out=str(REPO_ROOT / "data" / f"{track}_v2"),
                            steps=STEPS, seed=0)


if __name__ == "__main__":
    main()
