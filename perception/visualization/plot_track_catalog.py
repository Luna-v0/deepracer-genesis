"""Contact sheet of every track in the dataset, seen from above.

    python -m perception.visualization.plot_track_catalog
"""

import math

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deepracer_genesis.envs.track import ASSETS_DIR, TRACKS
from deepracer_genesis.tools.track_builder import plot_track, track_metrics

from perception.dataset import DATASET_TRACKS, HOLDOUT_TRACKS, REPO_ROOT

COLUMNS = 6


def route(name):
    return np.load(f"{ASSETS_DIR}/{TRACKS[name][1]}").astype(np.float32)


def main():
    names = [t[:-3] for t in DATASET_TRACKS]
    rows = math.ceil(len(names) / COLUMNS)
    fig, axes = plt.subplots(rows, COLUMNS, figsize=(2.4 * COLUMNS, 2.6 * rows))
    for ax, name in zip(axes.flat, names):
        r = route(name)
        plot_track(r, ax=ax, dash_len=0.6, dash_gap=0.7)
        m = track_metrics(r)
        is_val = f"{name}_v2" in HOLDOUT_TRACKS
        ax.set_title(f"{name}\n{m['length_m']:.0f} m  |  {m['width_m']:.2f} m",
                     fontsize=6.5, pad=3, color="crimson" if is_val else "black")
    for ax in axes.flat[len(names):]:
        ax.axis("off")

    fig.suptitle(f"{len(names)} tracks  -  red: held out", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = REPO_ROOT / "runs" / "figures" / "track_catalog.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print("open:", out)


if __name__ == "__main__":
    main()
