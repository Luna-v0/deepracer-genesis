"""GPU verification: tiled multi-track variants render correctly under Nyx.

The old guard refused Nyx + multiple tracks outright, but its rationale only
applies to the HETEROGENEOUS (superimposed) path — Nyx has no per-env
visibility masking. Part O tiling never superimposes anything: each variant is
a plain mesh on its own world tile, ordinary scene content for a path tracer.
This script settles the question empirically so the zoo's constraint matrix
row can flip from "unsupported" to "per tile".

Checks (all cars teleported to the SAME waypoint of their own variant, Nyx
temporal settle applied by the editor session):

1. rulebook — per-env ``half_width`` equals the variant scale x the scale-1.0
   reference (the tiled localizer works per variant under Nyx).
2. same-tile determinism — two envs on the SAME variant at the same pose
   render near-identical frames (path-traced, denoise off, so ~exact).
3. cross-variant difference — a narrow-variant and a wide-variant frame
   differ at the same relative pose: the width is visible, per env, under Nyx.
4. non-degenerate — every env's frame has real content (std above floor).

Needs gs-nyx (+plugin) and a GPU. Exit 0 only if all checks pass::

    python scripts/verify_nyx_tiling.py
"""

from __future__ import annotations

import argparse
import os

import torch
from PIL import Image

from deepracer_genesis.tools.dr_editor.session import EditorSession
from deepracer_genesis.tools.track_builder import width_variants


def main() -> None:
    """Run the four Nyx-tiling checks and exit 0 only if all pass."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="donut_track")
    ap.add_argument("--scales", default="0.9,1.0,1.15")
    ap.add_argument("--num_envs", type=int, default=6)
    ap.add_argument("--waypoint", type=int, default=5)
    ap.add_argument("--out", default="logs/validation")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    scales = tuple(float(s) for s in args.scales.split(","))
    if 1.0 not in scales:
        ap.error("--scales must include 1.0 (the rulebook reference)")
    names = width_variants(args.track, scales)
    print(f"variants under nyx: {dict(zip(names, scales))}")

    session = EditorSession.from_defaults(
        track=",".join(names), renderer="nyx", num_envs=args.num_envs,
        spectator=False)
    session.teleport(waypoint=args.waypoint)
    env = session.env
    scale_env = torch.tensor(scales, device=env.device)[env.track.variant_idx]
    raw = session.raw()                                      # (N, 3, H, W)

    # 1 ---- rulebook: half_width follows the variant scale
    hw = env.half_width
    ref = hw[scale_env == 1.0][0]
    ok1 = bool(torch.allclose(hw, scale_env * ref, rtol=1e-4))
    print(f"per-env half_width: {[round(v, 4) for v in hw.tolist()]}  "
          f"(expected scale x {ref:.4f})")

    # 2/3 ---- same-tile ~identical, cross-variant different
    def pair(i: int, j: int) -> float:
        return (raw[i] - raw[j]).abs().mean().item()

    ev = env.track.variant_idx.tolist()
    n = env.num_envs
    same = [pair(i, j) for i in range(n) for j in range(i + 1, n)
            if ev[i] == ev[j]]
    lo_i = ev.index(int(torch.tensor(scales).argmin()))
    hi_i = ev.index(int(torch.tensor(scales).argmax()))
    cross = pair(lo_i, hi_i)
    ok2 = bool(same) and max(same) < 0.02
    ok3 = cross > 0.005
    print(f"same-variant max |Δ|: {max(same):.5f}   "
          f"narrow-vs-wide |Δ|: {cross:.5f}")

    # 4 ---- non-degenerate frames
    stds = raw.flatten(1).std(dim=1)
    ok4 = bool((stds > 0.02).all())
    print(f"per-env frame std: {[round(v, 3) for v in stds.tolist()]}")

    for i in range(n):
        Image.fromarray((raw[i].permute(1, 2, 0) * 255).byte().cpu().numpy()
                        ).save(os.path.join(args.out, f"nyx_tile_env{i}.png"))
    print(f"saved per-env onboard PNGs -> {args.out}")

    for tag, ok, msg in (
            ("rulebook", ok1, "half_width == scale x reference per env"),
            ("same-tile", ok2, "same-variant frames ~identical (< 0.02)"),
            ("cross-tile", ok3, "narrow vs wide frames differ (> 0.005)"),
            ("content", ok4, "all frames non-degenerate (std > 0.02)")):
        print(f"[{'PASS' if ok else 'FAIL'}] {tag}: {msg}")
    if ok1 and ok2 and ok3 and ok4:
        print("NYX-TILING PASS: tiled multi-track variants render per env under Nyx.")
        raise SystemExit(0)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
