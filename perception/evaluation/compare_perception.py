"""Compare the same policy fed by the simulator, then by the CNN.

Same track, same camera, same checkpoint: only the source of the 7 values
changes. This is the measurement that says whether the learned perception costs
anything to the driving.

Each camera scene runs in its own subprocess. pyrender's OpenGL context is
process-global: when the garbage collector destroys a previous scene it takes
the live scene's context with it, and the next render fails on
`glBindFramebuffer: invalid operation`. A single-process loop therefore cannot
survive past one scene.

    python -m perception.evaluation.compare_perception <path/model.pt> [track ...] \
        [--video] [--cnn <path/cnn.pt>]
"""

import json
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

METRICS = ("episodes", "mean_return", "mean_progress_m", "mean_episode_s",
           "completion_rate", "mean_laps", "offtrack_rate", "lap_time_s",
           "mean_speed_mps")
SOURCES = ("sim", "cnn")
NUM_ENVS = 16
REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE = "perception.evaluation.compare_perception"


def _feature_set(source):
    from deepracer_genesis.envs.features import PerceptionFeatures

    from perception.cnn_features import CNNPerceptionFeatures
    return PerceptionFeatures if source == "sim" else CNNPerceptionFeatures


# ------------------------------------------------------------------- child
def evaluate_one(ckpt, track, source, cnn=None):
    """One track, one perception source, one scene. Prints RESULT <json>."""
    from deepracer_genesis.experiment import run
    from deepracer_genesis.experiment.evaluator import (build_single_track_sim,
                                                        evaluate_on_tracks)
    from deepracer_genesis.experiment.visualize import _rsl_actor

    from perception.train_policy_with_cnn import CNNPerceptionPolicy

    spec = run(CNNPerceptionPolicy, build_only=True,
               feature_set=_feature_set(source), tracks=(track,), num_envs=NUM_ENVS,
               feature_params={"checkpoint": cnn} if cnn else None)
    sim = build_single_track_sim(spec, track, NUM_ENVS)
    actor = _rsl_actor(spec, ckpt, sim)
    m = evaluate_on_tracks(actor, (track,), sim_factory=lambda t, s=sim: s)[track]
    print("RESULT " + json.dumps(m))


def record_one(ckpt, track, source, cnn=None):
    from deepracer_genesis.experiment.visualize import rollout_video

    from perception.train_policy_with_cnn import CNNPerceptionPolicy

    rollout_video(CNNPerceptionPolicy, root="runs", ckpt=ckpt, track=track,
                  steps=1500, num_envs=1, out=f"runs/videos/{source}",
                  feature_set=_feature_set(source), tracks=(track,),
                  feature_params={"checkpoint": cnn} if cnn else None)
    print("RESULT {}")


# ------------------------------------------------------------------ parent
def run_child(ckpt, mode, track, source, cnn=None):
    argv = [sys.executable, "-m", MODULE, ckpt, mode, track, source]
    if cnn:
        argv += ["--cnn", cnn]
    p = subprocess.run(argv, cwd=REPO_ROOT, capture_output=True, text=True)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    if line is None:
        print("  failed:\n" + "\n".join((p.stdout + p.stderr).splitlines()[-6:]))
        return None
    return json.loads(line[7:])


def print_track_table(track, res):
    print(f"\n=== {track} ===")
    print(f"{'metric':18} {'sim':>10} {'cnn':>10} {'delta':>9}")
    for k in METRICS:
        a, b = res["sim"][k], res["cnn"][k]
        delta = f"{100*(b-a)/a:+7.0f} %" if a else "       -"
        print(f"{k:18} {a:10.3f} {b:10.3f} {delta}")


def print_summary(summary):
    print(f"\n{'':24} {'offtrack rate':>21} {'progress (m)':>21}")
    print(f"{'track':24} {'sim':>10} {'cnn':>10} {'sim':>10} {'cnn':>10}")
    for track, r in summary:
        print(f"{track:24} {r['sim']['offtrack_rate']:10.2f} "
              f"{r['cnn']['offtrack_rate']:10.2f} "
              f"{r['sim']['mean_progress_m']:10.1f} "
              f"{r['cnn']['mean_progress_m']:10.1f}")
    n = len(summary)
    mean = lambda s, k: sum(r[s][k] for _, r in summary) / n
    print(f"{'MEAN':24} {mean('sim','offtrack_rate'):10.2f} "
          f"{mean('cnn','offtrack_rate'):10.2f} "
          f"{mean('sim','mean_progress_m'):10.1f} "
          f"{mean('cnn','mean_progress_m'):10.1f}")


def main():
    args = sys.argv[1:]
    ckpt = args.pop(0)

    cnn = None                      # which CNN feeds the "cnn" case
    if "--cnn" in args:
        i = args.index("--cnn")
        cnn = args[i + 1]
        del args[i:i + 2]

    if args and args[0] in ("--one", "--video-one"):
        mode, track, source = args[0], args[1], args[2]
        (evaluate_one if mode == "--one" else record_one)(ckpt, track, source, cnn)
        return

    video = "--video" in args
    tracks = [a for a in args if not a.startswith("--")]
    if not tracks:
        from perception.train_policy_with_cnn import CNNPerceptionPolicy
        tracks = list(CNNPerceptionPolicy.tracks)

    summary = []
    for track in tracks:
        res = {}
        for source in SOURCES:
            print(f"  {track:24} {source:8} ...", flush=True)
            m = run_child(ckpt, "--one", track, source, cnn)
            if m is None:
                break
            res[source] = m
            if video:
                run_child(ckpt, "--video-one", track, source, cnn)
        if len(res) != 2:
            print(f"\n=== {track} === incomplete, track skipped\n", flush=True)
            continue

        summary.append((track, res))
        print_track_table(track, res)
        if video:
            print(f"videos: runs/videos/sim|cnn/spectator_{track}.mp4")
        print(flush=True)

    if len(summary) > 1:
        print_summary(summary)


if __name__ == "__main__":
    main()
