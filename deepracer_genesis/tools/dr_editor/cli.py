"""The editor CLI: one command per question, one labeled PNG per answer.

Commands (``python -m deepracer_genesis.tools.dr_editor <cmd>``):

- ``knobs``  — the registry truth table (schedule / liveness / renderers).
- ``sweep``  — one knob across its range at a fixed pose -> filmstrip PNG +
  pasteable stage code (``--bank`` = instant, Genesis-free).
- ``grid``   — N envs at ONE shared pose, onboard + top-down rows, one sim
  instant -> contact sheet (per-env knobs are invisible without this).
- ``stages`` — raw -> world colour -> pixel noise -> policy res -> aug ->
  latency -> stack, one strip (the honest "what the policy sees").
- ``prove``  — the automated check suite (also via
  ``python -m deepracer_genesis.validation.dr_check`` for CI).
- ``bank``   — record a raw frame bank for the offline tier.

Headless by design: artifacts land under ``--out`` (default
``logs/dr_editor``), stamped with their seed so any frame can be regenerated.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from . import emit, knobs as K
from .rng import stamp


def _parse_pick(knob, pick: str):
    """Parse a ``--pick`` value string into the knob's cfg shape.

    Args:
        knob: The knob being emitted.
        pick: ``"lo:hi"`` for range-shaped knobs or a single number.

    Returns:
        A ``(lo, hi)`` tuple or a scalar, matching the knob's cfg shape.
    """
    if ":" in pick:
        lo, hi = (float(x) for x in pick.split(":", 1))
        return (lo, hi)
    v = float(pick)
    return int(v) if v.is_integer() and knob.name in ("latency_steps",
                                                      "delay_steps") else v


def _session_updates(names: list[str], renderer: str) -> dict:
    """Build-time cfg updates needed so the listed knobs are live/provable.

    Args:
        names: Knob names the command will exercise.
        renderer: The session's renderer (catalog vocabulary of the CLI flag).

    Returns:
        Dotted-path cfg updates: env-map ranges (Nyx only) and physics rand
        entries (physics DR applies once per run at build) for any
        rebuild-class knob in the list.
    """
    updates: dict = {}
    env_map: dict = {}
    for name in names:
        kn = K.get(name)
        if kn.kind == "env_map" and renderer == "nyx":
            env_map["tint" if name == "env_map_tint" else "multiplier"] = \
                (kn.space.lo, kn.space.hi)
        elif kn.kind == "physics" and "camera" in kn.modalities:
            updates[f"rand.{kn.cfg_path.rsplit('.', 1)[1]}"] = kn.space.to_cfg()
            updates["rand.randomize"] = True
    if env_map:
        updates["vision.env_map"] = env_map
    return updates


def _build_session(args, names: Optional[list[str]] = None):
    """Construct the live session a command asked for.

    Args:
        args: Parsed CLI args (``target``/``track``/``renderer``/``num_envs``/
            ``res``).
        names: Knob names needing build-time cfg (rebuild-class).

    Returns:
        The built :class:`~.session.EditorSession`.
    """
    from .session import EditorSession

    updates = _session_updates(names or [], args.renderer)
    if args.target:
        return EditorSession.from_target(args.target, num_envs=args.num_envs,
                                         cfg_updates=updates)
    res = tuple(int(x) for x in args.res.split("x"))
    return EditorSession.from_defaults(
        track=args.track, renderer=args.renderer, num_envs=args.num_envs,
        res=res, cfg_updates=updates)


def _raw_frames(args, names: Optional[list[str]] = None):
    """One raw batch: from a bank (offline) or a fresh live session.

    Args:
        args: Parsed CLI args (uses ``bank``/``waypoint`` plus the session
            flags).
        names: Rebuild-class knob names for the live path.

    Returns:
        ``(raw, session_or_None, dr)`` — raw ``(N, 3, H, W)`` float frames,
        the live session when one was built, and the target's DR params
        (empty without ``--target``).
    """
    if args.bank:
        from .frames import FrameBank
        bank = FrameBank(args.bank)
        return bank.raw(0), None, {}
    session = _build_session(args, names)
    session.teleport(waypoint=args.waypoint)
    return session.raw(), session, session.dr


# ---------------------------------------------------------------- commands
def cmd_knobs(args) -> int:
    """Print the registry truth table."""
    rows = [k for k in K.REGISTRY.values()
            if args.layer in ("all", getattr(k.source, "layer", "static"))]
    hdr = f"{'knob':22} {'schedule':11} {'liveness':8} {'kind':11} " \
          f"{'renderers':26} cfg path"
    print(hdr + "\n" + "-" * len(hdr))
    for k in sorted(rows, key=lambda k: (k.liveness, k.name)):
        rend = ",".join(sorted(k.renderers)) if "camera" in k.modalities \
            else "feature-only"
        print(f"{k.name:22} {k.schedule:11} {k.liveness:8} {k.kind:11} "
              f"{rend:26} {k.cfg_path}")
        if k.note:
            print(f"{'':22}   {k.note}")
    return 0


def cmd_sweep(args) -> int:
    """Filmstrip one knob across its range at a fixed pose."""
    from .pipeline import replay_stages
    from .sheets import sheet

    knob = K.get(args.knob)
    if knob.liveness != "offline":
        print(f"'{args.knob}' is {knob.liveness}-class ({knob.kind}); sweep "
              f"handles offline image/visual knobs — use 'grid' or 'prove'.")
        return 2
    raw, _, _ = _raw_frames(args)
    frame = raw[args.env:args.env + 1]
    values = ([float(v) for v in args.values.split(",")] if args.values
              else K.sweep_values(knob, args.points))
    tiles = []
    for v in values:
        out = replay_stages(frame, K.dr_for_value(knob, v), seed=args.seed)
        tiles.append((f"{v:.3g}", out["image_aug"][0]))
    path = os.path.join(args.out, f"sweep_{knob.name}_{stamp(args.seed)}.png")
    sheet(tiles, path, title=f"{knob.name}  [{knob.schedule}]  "
          f"{'deterministic values' if knob.kind == 'aug_range' else f'seed {args.seed}'}")
    print(f"wrote {path}")
    pick = _parse_pick(knob, args.pick) if args.pick else (
        (values[0], values[-1]) if knob.kind == "aug_range" else values[-1])
    print("\npasteable stage code:\n  " + emit.emit_knob_code(knob, pick))
    if args.save_preset:
        dr = K.dr_for_value(knob, pick[1] if isinstance(pick, tuple) else pick)
        if knob.kind == "aug_range":
            dr = {"image_aug": {knob.name: pick}}
        print("preset -> " + emit.save_preset(args.save_preset, dr))
    return 0


def cmd_grid(args) -> int:
    """Contact-sheet N envs at one shared pose (onboard + top-down rows)."""
    from .pipeline import replay_stages
    from .prove import _range_dr
    from .sheets import paired_sheet

    knob = K.get(args.knob) if args.knob else None
    names = [args.knob] if args.knob else []
    session = _build_session(args, names)
    session.teleport(waypoint=args.waypoint)
    if knob and knob.kind == "mount":
        session.reroll_mount(
            knob.space.m if knob.name == "camera_pitch_jitter" else 0.0,
            knob.space.m if knob.name == "camera_pos_jitter" else 0.0,
            seed=args.seed)
    raw = session.raw()
    if knob and knob.liveness == "offline":
        raw = replay_stages(raw, _range_dr(knob), seed=args.seed)["image_aug"]
    top = session.topdown()
    n = raw.shape[0]
    rows = [("onboard", [(f"env{i}", raw[i]) for i in range(n)]),
            ("topdown", [(f"env{i}", top[i]) for i in range(n)])]
    label = knob.name if knob else "scene"
    path = os.path.join(args.out, f"grid_{label}_{stamp(args.seed)}.png")
    paired_sheet(rows, path, scale=2, title=f"{label} — {n} envs, one pose, "
                 f"one instant ({stamp(args.seed)})")
    print(f"wrote {path}")
    return 0


def cmd_stages(args) -> int:
    """One strip: every pipeline stage of one env at one instant."""
    from .pipeline import STAGES, replay_stages
    from .sheets import sheet

    raw, session, dr = _raw_frames(args)
    if args.preset:
        import json
        with open(args.preset if args.preset.endswith(".json") else
                  os.path.join(emit.PRESET_DIR, f"{args.preset}.json")) as f:
            dr = json.load(f)["dr"]
    if not any(dr.get(k) for k in ("image_aug", "world_color", "pixel_noise")):
        print("note: no DR params (pass --target or --preset); stages will "
              "be identical to raw")
    policy_res = (session.env.cfg["vision"].get("policy_res")
                  if session else None)
    fs = (session.env.cfg["vision"].get("frame_stack", 4) if session else 4)
    out = replay_stages(raw, dr, policy_res=policy_res, frame_stack=fs,
                        seed=args.seed)
    tiles = [(s if s != "stack" else f"stack (newest of {fs})",
              out[s][args.env]) for s in STAGES]
    path = os.path.join(args.out, f"stages_{stamp(args.seed)}.png")
    sheet(tiles, path, title=f"pipeline stages, env {args.env}, "
          f"{stamp(args.seed)} (latency=identity at a static instant)")
    print(f"wrote {path}")
    return 0


def cmd_prove(args) -> int:
    """Run the check suite for a comma-list of knobs; exit 0 iff all pass."""
    from .prove import run_suite

    names = [n.strip() for n in args.knobs.split(",")]
    session = _build_session(args, names)
    out_json = os.path.join(args.out, "dr_prove.json")
    verdicts = run_suite(session, names, seed=args.seed,
                         waypoint=args.waypoint, out_json=out_json)
    failed = 0
    for v in verdicts:
        tag = "PASS" if v.passed else "FAIL"
        failed += not v.passed
        print(f"[{tag}] {v.knob:20} {v.check:16} {v.detail}")
    print(f"verdicts -> {out_json}")
    return 1 if failed else 0


def cmd_bank(args) -> int:
    """Record a raw frame bank for the offline tier."""
    from .frames import FrameBank

    session = _build_session(args)
    wps = [int(w) for w in args.waypoints.split(",")]
    bank = FrameBank.record(session, args.bank_out, waypoints=wps)
    print(f"recorded {len(bank)} poses x {bank.meta['num_envs']} envs "
          f"({bank.meta['renderer']}, raw) -> {args.bank_out}")
    return 0


# ------------------------------------------------------------------ parser
def main(argv: Optional[list] = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (``prove`` returns 1 on any failed check).
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    ap = argparse.ArgumentParser(
        prog="python -m deepracer_genesis.tools.dr_editor",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, bank: bool = False):
        p.add_argument("--target", default=None,
                       help="module:ClassName Experiment whose config to inspect")
        p.add_argument("--track", default="reinvent_base")
        p.add_argument("--renderer", default="batch",
                       choices=("batch", "nyx", "rasterizer"))
        p.add_argument("--num-envs", type=int, default=12, dest="num_envs")
        p.add_argument("--res", default="160x120")
        p.add_argument("--waypoint", type=int, default=5)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out", default="logs/dr_editor")
        if bank:
            p.add_argument("--bank", default=None,
                           help="use a recorded frame bank (no GPU/Genesis)")

    p = sub.add_parser("knobs", help="print the knob truth table")
    p.add_argument("--layer", default="all")
    p.set_defaults(fn=cmd_knobs)

    p = sub.add_parser("sweep", help="filmstrip one knob across its range")
    p.add_argument("knob")
    p.add_argument("--points", type=int, default=12)
    p.add_argument("--values", default=None, help="explicit comma list")
    p.add_argument("--env", type=int, default=0)
    p.add_argument("--pick", default=None, help="'lo:hi' or value to emit")
    p.add_argument("--save-preset", default=None, dest="save_preset")
    common(p, bank=True)
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("grid", help="N envs, one pose, one instant")
    p.add_argument("--knob", default=None)
    common(p)
    p.set_defaults(fn=cmd_grid)

    p = sub.add_parser("stages", help="the full pipeline of one env, one strip")
    p.add_argument("--env", type=int, default=0)
    p.add_argument("--preset", default=None)
    common(p, bank=True)
    p.set_defaults(fn=cmd_stages)

    p = sub.add_parser("prove", help="automated knob checks (exit code!)")
    p.add_argument("knobs", help="comma-separated knob names")
    common(p)
    p.set_defaults(fn=cmd_prove)

    p = sub.add_parser("bank", help="record a raw frame bank")
    p.add_argument("--bank-out", default="datasets/editor_bank", dest="bank_out")
    p.add_argument("--waypoints", default="0,5,10,20,40")
    common(p)
    p.set_defaults(fn=cmd_bank)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
