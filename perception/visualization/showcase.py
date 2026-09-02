"""Measure, pick two tracks from the numbers, and build the comparison videos.

The point is to show what the learned perception costs. Same policy, same track,
same camera -- only the source of the 7 values changes (exact simulator vs CNN).

    caffeinate -di .venv/bin/python -m perception.visualization.showcase <path/model.pt>

Roughly 40 min:
  1. evaluate every track under both sources     (one subprocess per scene)
  2. pick, FROM THE NUMBERS, the track where the CNN helps most and the one
     where the two are closest -- no track is chosen by hand
  3. render the 4 fleet videos and stitch them side by side, two at a time
  4. write runs/showcase/showcase.json

One camera scene per subprocess: pyrender's OpenGL context is process-global,
and destroying a previous scene invalidates the live one.
"""

import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "runs" / "showcase"

TRACKS = ("2022_march_open", "arctic_open", "jyllandsringen_open",
          "hamption_pro", "thunder_hill_pro", "Tokyo_Training_track",
          "dubai_open", "Monaco")
SOURCES = ("sim", "cnn")
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"   # mac path; falls back below


def banner(text, width, to):
    """Render a title banner as a PNG.

    We do not use ffmpeg's `drawtext` filter: it is not compiled into every
    build (it is missing from the Homebrew one here), and its absence only shows
    up at run time.
    """
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype(FONT, 30)
    except OSError:
        font = ImageFont.load_default()
    im = Image.new("RGBA", (width, 62), (30, 33, 40, 225))
    d = ImageDraw.Draw(im)
    l, t, r, b = d.textbbox((0, 0), text, font=font)
    d.text(((width - (r - l)) / 2 - l, (62 - (b - t)) / 2 - t), text,
           font=font, fill=(255, 255, 255, 255))
    im.save(to)
    return to


def evaluate(ckpt, track, source):
    p = subprocess.run([sys.executable, "-m",
                        "perception.evaluation.compare_perception",
                        ckpt, "--one", track, source],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    line = next((x for x in p.stdout.splitlines() if x.startswith("RESULT ")), None)
    if line is None:
        print("      failed:", "\n".join((p.stdout + p.stderr).splitlines()[-4:]))
    return json.loads(line[7:]) if line else None


def record(ckpt, track, source):
    """Render the fleet video; returns its path, or None."""
    p = subprocess.run([sys.executable, "-m",
                        "perception.visualization.fleet_video", ckpt, track, source],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    f = REPO_ROOT / "runs" / "videos" / source / f"fleet_{track}.mp4"
    if not f.exists():
        print("      video failed:", "\n".join((p.stdout + p.stderr).splitlines()[-4:]))
        return None
    return f


def side_by_side(left, right, out):
    """Stitch two videos side by side, each under its own banner."""
    width = int(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
         "stream=width", "-of", "csv=p=0", str(left)],
        capture_output=True, text=True).stdout.strip())
    bl = banner("EXACT PERCEPTION", width, OUT_DIR / "_banner_left.png")
    br = banner("THROUGH THE CNN", width, OUT_DIR / "_banner_right.png")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(left), "-i", str(right),
         "-i", str(bl), "-i", str(br), "-filter_complex",
         "[0:v][2:v]overlay=0:0[l];[1:v][3:v]overlay=0:0[r];[l][r]hstack=inputs=2[v]",
         "-map", "[v]", "-c:v", "libx264", "-crf", "24", "-pix_fmt", "yuv420p",
         str(out)], check=True)
    return out


def pick_tracks(res):
    """The track where the CNN helps most, and the one where the two match best."""
    delta = {t: r["sim"]["offtrack_rate"] - r["cnn"]["offtrack_rate"]
             for t, r in res.items()}
    helps = max(delta, key=delta.get)
    if delta[helps] <= 0.02:      # no track where the CNN cuts off-tracks
        helps = max(res, key=lambda t: res[t]["cnn"]["mean_progress_m"]
                    - res[t]["sim"]["mean_progress_m"])
    similar = min((t for t in res if t != helps),
                  key=lambda t: abs(delta[t]) + abs(res[t]["cnn"]["mean_progress_m"]
                  - res[t]["sim"]["mean_progress_m"]) / 50)
    return helps, similar


def print_table(res):
    print(f"\n{'':24} {'offtrack rate':>21} {'progress (m)':>21}")
    print(f"{'track':24} {'sim':>10} {'cnn':>10} {'sim':>10} {'cnn':>10}")
    for t, r in res.items():
        print(f"{t:24} {r['sim']['offtrack_rate']:10.2f} "
              f"{r['cnn']['offtrack_rate']:10.2f} "
              f"{r['sim']['mean_progress_m']:10.1f} "
              f"{r['cnn']['mean_progress_m']:10.1f}")
    n = len(res)
    mean = lambda s, k: sum(r[s][k] for r in res.values()) / n
    print(f"{'MEAN':24} {mean('sim','offtrack_rate'):10.2f} "
          f"{mean('cnn','offtrack_rate'):10.2f} {mean('sim','mean_progress_m'):10.1f} "
          f"{mean('cnn','mean_progress_m'):10.1f}", flush=True)


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "perception/reference_policy.pt"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"\n1/3  measuring {len(TRACKS)} tracks under both perception sources\n")
    res = {}
    for i, track in enumerate(TRACKS, 1):
        r = {}
        for source in SOURCES:
            print(f"  {i:2}/{len(TRACKS)}  {track:24} {source:8}", flush=True)
            m = evaluate(ckpt, track, source)
            if m is None:
                break
            r[source] = m
        if len(r) == 2:
            res[track] = r
            (OUT_DIR / "showcase.json").write_text(json.dumps(res, indent=1))
    if len(res) < 2:
        print("not enough tracks measured, stopping")
        return
    print_table(res)

    helps, similar = pick_tracks(res)
    print("\n2/3  tracks selected, from the numbers")
    print(f"  cnn helps most : {helps}")
    print(f"  the two match  : {similar}\n", flush=True)

    print("3/3  videos (4 renders + 2 stitches)\n")
    videos = {}
    for label, track in (("helps", helps), ("similar", similar)):
        sides = [record(ckpt, track, source) for source in SOURCES]
        if None in sides:
            continue
        f = side_by_side(sides[0], sides[1], OUT_DIR / f"{label}_{track}.mp4")
        videos[label] = {"track": track, "file": str(f)}
        print(f"  {f}", flush=True)

    (OUT_DIR / "showcase.json").write_text(json.dumps(
        {"measurements": res, "videos": videos, "checkpoint": ckpt}, indent=1))
    print(f"\ndone in {(time.time()-t0)/60:.0f} min")
    print(f"everything is in {OUT_DIR}")


if __name__ == "__main__":
    main()
