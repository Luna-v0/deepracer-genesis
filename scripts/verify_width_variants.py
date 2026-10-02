"""GPU verification of baked width variants: visible per-env width + matching rulebook.

``track_width_scale`` is feature-only DR: it scales the rulebook while the baked
mesh stays fixed, so under camera the width itself must vary. ``width_variants``
bakes one generated track per width scale (same centerline, borders scaled) and
the multi-track tiling gives each env its own variant. This script parks every
car at the SAME waypoint of its own variant and proves the width difference is
real on both sides of the contract:

1. rulebook — ``env.half_width`` per env equals its variant scale times the
   scale-1.0 env's half-width.
2. visible (top-down) — the road-pixel count in each env's bird's-eye view is
   strictly monotone in the variant scale.
3. visible (onboard) — a narrow-variant and a wide-variant frame differ at the
   same relative pose, while two envs on the SAME variant render ~identically,
   so the frame difference IS the width.
4. off-track — cars parked at 1.05x the NARROW variant's half-width trip the
   env's own off-track signal on narrow-variant envs only (the grace margins
   are zeroed: a 5% overhang is ~1 cm at DeepRacer scale, far inside the
   default 8-10 cm margins).

Needs a working Madrona batch renderer (GPU). On a CUDA-13 toolkit, first run
``scripts/fix_madrona_cuda13.sh`` (see the README CUDA-13 note).

    python scripts/verify_width_variants.py
"""

from __future__ import annotations

import argparse
import os

import torch
from PIL import Image

from deepracer_genesis._gs import ensure_init
from deepracer_genesis.configs.cfgs import get_env_cfg
from deepracer_genesis.envs import DeepRacerEnv
from deepracer_genesis.tools.track_builder import width_variants


def _teleport(env: DeepRacerEnv, wp: int, lateral_m: torch.Tensor) -> None:
    """Park every car at waypoint ``wp`` of its own variant and refresh state.

    Places each env exactly at its variant's centerline waypoint (the padded
    ``MultiTrack`` centerlines already include the Part O tile offsets) plus a
    signed offset along the local left normal, facing the track tangent, then
    settles with ``_post_physics`` (no dynamics step) to refresh the localizer
    and the cameras — the sweep-dataset teleport recipe.

    Args:
        env: The built env whose cars are teleported.
        wp: Waypoint index probed on every variant (variants share a centerline).
        lateral_m: ``(N,)`` signed lateral offsets in metres.
    """
    mt = env.track
    ev = mt.variant_idx
    yaw = mt.track_yaw[ev, wp]
    qpos = torch.zeros(env.num_envs, 13, device=env.device)
    qpos[:, 0:2] = mt.center[ev, wp] + mt.normal[ev, wp] * lateral_m[:, None]
    qpos[:, 2] = env.cfg["spawn"]["spawn_height"]
    qpos[:, 3] = torch.cos(yaw / 2)
    qpos[:, 6] = torch.sin(yaw / 2)
    env.car.set_qpos(qpos)
    env._post_physics(torch.arange(env.num_envs, device=env.device))


def _save_png(rgb_hwc: torch.Tensor, path: str) -> None:
    """Save an ``(H, W, 3)`` uint8 RGB tensor as a PNG file.

    Args:
        rgb_hwc: The frame to save, channels-last uint8.
        path: Destination PNG path.
    """
    Image.fromarray(rgb_hwc.cpu().numpy()).save(path)


def main() -> None:
    """Run the four width-variant checks and exit 0 only if all pass."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="donut_track")
    ap.add_argument("--scales", default="0.9,1.0,1.15",
                    help="comma-separated width scales; must include the 1.0 reference")
    ap.add_argument("--num_envs", type=int, default=6)
    ap.add_argument("--out", default="logs/validation")
    ap.add_argument("--waypoint", type=int, default=5)
    args = ap.parse_args()

    scales = tuple(float(s) for s in args.scales.split(","))
    if 1.0 not in scales:
        ap.error("--scales must include 1.0 (the rulebook reference)")
    os.makedirs(args.out, exist_ok=True)

    # This tiny scene needs nowhere near madrona's default 4 GiB device-malloc
    # heap, and on a busy shared GPU that reservation fails the very first
    # kernel launch (CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES). Cap it; an explicit
    # MADRONA_MWGPU_DEVICE_HEAP_SIZE in the environment still wins.
    os.environ.setdefault("MADRONA_MWGPU_DEVICE_HEAP_SIZE", str(1 << 30))

    ensure_init("gpu")
    names = width_variants(args.track, scales)
    print(f"variants: {dict(zip(names, scales))}")

    env_cfg = get_env_cfg(vision=True, track=list(names), topdown=True)
    # deterministic spawn; the probes teleport to exact poses anyway
    env_cfg["spawn"]["random_start"] = False
    env_cfg["spawn"]["spawn_lateral_noise"] = 0.0
    env_cfg["spawn"]["spawn_yaw_noise"] = 0.0
    # zero the grace margins so the off-track signal tests the pure rulebook width
    env_cfg["termination"]["off_track_margin"] = 0.0
    env_cfg["termination"]["wheel_margin"] = 0.0
    env = DeepRacerEnv(num_envs=args.num_envs, env_cfg=env_cfg)
    env.reset_idx(torch.arange(args.num_envs, device=env.device))

    ev = env.track.variant_idx.tolist()                     # env -> variant
    scale_env = torch.tensor(scales, device=env.device)[env.track.variant_idx]
    scale_list = scale_env.tolist()
    print(f"per-env variant scales: {scale_list}")
    wp = args.waypoint
    _teleport(env, wp, torch.zeros(args.num_envs, device=env.device))  # on centerline

    # 1 ---- rulebook: per-env half_width follows the variant scale
    hw = env.half_width
    ref = hw[scale_env == 1.0][0]
    expected = scale_env * ref
    ok1 = torch.allclose(hw, expected, rtol=1e-4)
    print(f"per-env half_width:     {[round(v, 5) for v in hw.tolist()]}")
    print(f"expected (scale x ref): {[round(v, 5) for v in expected.tolist()]}")

    # 2 ---- visible top-down: road pixels strictly monotone in the scale
    top = env.render_topdown()                              # (N, H, W, 3) uint8
    lum = top.float().mean(dim=-1) / 255.0
    # under the Madrona light the dark road albedo renders at ~0.41 luminance;
    # the band excludes the white backdrop (~1.0), the yellow lane marks
    # (~0.75), and the dark car (~0.2)
    road_px = ((lum > 0.30) & (lum < 0.60)).sum(dim=(1, 2)).tolist()
    print(f"top-down road pixels:   {road_px}")
    ok2 = all(road_px[i] < road_px[j]
              for i in range(args.num_envs) for j in range(args.num_envs)
              if scale_list[i] < scale_list[j])
    for i in range(args.num_envs):
        _save_png(top[i], os.path.join(args.out, f"topdown_env{i}_{names[ev[i]]}.png"))

    # 3 ---- visible onboard: only the width separates the frames
    img = env.image_buf                                     # (N, 3, H, W) in [0, 1]
    by_variant: dict[int, list[int]] = {}
    for i, v in enumerate(ev):
        by_variant.setdefault(v, []).append(i)
    v_narrow, v_wide = scales.index(min(scales)), scales.index(max(scales))
    cross = (img[by_variant[v_narrow][0]] - img[by_variant[v_wide][0]]).abs().mean().item()
    same_pairs = [(img[g[0]] - img[g[1]]).abs().mean().item()
                  for g in by_variant.values() if len(g) >= 2]
    same = max(same_pairs) if same_pairs else 0.0
    print(f"onboard mean|diff|: narrow-vs-wide {cross:.4f}, same-variant max {same:.6f}")
    ok3 = cross > 0.005 and same < 1e-3
    for v, g in sorted(by_variant.items()):
        _save_png((img[g[0]].permute(1, 2, 0) * 255).byte(),
                  os.path.join(args.out, f"onboard_{names[v]}.png"))

    # 4 ---- off-track: 1.05x the NARROW half-width trips narrow envs only
    hw_narrow = float(env.track.half_width[v_narrow, wp])
    off_lat = torch.full((args.num_envs,), 1.05 * hw_narrow, device=env.device)
    _teleport(env, wp, off_lat)
    off = (env.signals["off_track"] > 0.5).tolist()         # the env's own signal
    expect_off = [s < 1.05 * min(scales) for s in scale_list]
    print(f"off_track at |lat|={1.05 * hw_narrow:.4f} m: {off} expected {expect_off}")
    ok4 = off == expect_off

    checks = [
        (ok1, "rulebook: half_width == variant scale x scale-1.0 reference (rtol 1e-4)"),
        (ok2, "top-down: road-pixel count strictly monotone in the width scale"),
        (ok3, "onboard: cross-width frames differ (>0.005), same-variant ~identical (<1e-3)"),
        (ok4, "off-track: 1.05x narrow half-width trips only narrow-variant envs"),
    ]
    for ok, desc in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {desc}")
    print(f"saved per-env top-down + per-variant onboard PNGs -> {args.out}")
    all_ok = all(ok for ok, _ in checks)
    print("WIDTH-VARIANTS PASS: per-env visible width with a matching rulebook."
          if all_ok else "WIDTH-VARIANTS FAIL: see the failed checks above.")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
