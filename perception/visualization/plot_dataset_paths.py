"""Top-down view of a track plus the paths actually recorded in the dataset.

Shows how the collected episodes cover the track, and where they ended.

    python -m perception.visualization.plot_dataset_paths <dataset folder>
"""

import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS = REPO_ROOT / "deepracer_genesis" / "assets"


def route(name):
    direct = ASSETS / "routes" / f"{name}.npy"
    if direct.exists():
        return np.load(direct)[:, :2]
    return np.load(ASSETS / "tracks" / "generated" / name / "route.npy")[:, :2]


def main():
    folder = sys.argv[1]
    name = folder.replace("_dr", "").replace("_v2", "")
    center = route(name)

    files = sorted(glob.glob(str(REPO_ROOT / "data" / folder / "*.parquet")))
    table = pq.read_table(files[0], columns=["env", "pose", "done"])
    pose = np.array(table["pose"].to_pylist(), dtype=np.float32)
    env = np.array(table["env"].to_pylist())
    done = np.array(table["done"].to_pylist())

    plt.figure(figsize=(9, 9))
    plt.plot(center[:, 0], center[:, 1], "k-", lw=3, alpha=.35, label="track centre")
    for car in sorted(set(env))[:8]:
        m = env == car
        plt.plot(pose[m, 0], pose[m, 1], lw=1, alpha=.8)
    plt.scatter(pose[done, 0], pose[done, 1], c="red", s=60, zorder=5,
                label=f"off-tracks ({done.sum()})")
    plt.axis("equal")
    plt.legend()
    plt.title(f"{folder} - {len(set(env))} cars, {table.num_rows} frames")

    out = REPO_ROOT / "runs" / "figures" / f"dataset_paths_{folder}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=90, facecolor="white", bbox_inches="tight")
    print("open:", out)


if __name__ == "__main__":
    main()
