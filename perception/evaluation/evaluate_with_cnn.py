"""Put an already-trained policy on the track, fed by the real CNN.

No training at all: this measures what the perception costs a policy that
already knows how to drive.

    python -m perception.evaluation.evaluate_with_cnn <path/model.pt>
"""

import sys
import warnings

warnings.filterwarnings("ignore")

from deepracer_genesis.experiment import run
from deepracer_genesis.experiment.evaluator import (
    build_single_track_sim,
    evaluate_on_tracks,
)
from deepracer_genesis.experiment.visualize import _rsl_actor

from perception.train_policy_with_cnn import CNNPerceptionPolicy

NUM_ENVS = 8


def main():
    ckpt = sys.argv[1]
    spec = run(CNNPerceptionPolicy, build_only=True)
    for track in CNNPerceptionPolicy.tracks:
        sim = build_single_track_sim(spec, track, NUM_ENVS)
        actor = _rsl_actor(spec, ckpt, sim)
        m = evaluate_on_tracks(actor, (track,),
                               sim_factory=lambda t, s=sim: s)[track]
        print(f"\n{track}  ({m['episodes']} episodes)")
        for k in ("completion_rate", "mean_progress_m", "mean_speed_mps",
                  "offtrack_rate", "mean_laps"):
            print(f"  {k:18} {m[k]:.3f}")


if __name__ == "__main__":
    main()
