"""CI gate over the DR editor's prove suite: pass/fail per knob, exit code.

The scriptable twin of ``dr_editor prove`` (same checks, same thresholds),
living beside ``camera_check`` with the same contract: verdict lines on
stdout, a JSON artifact, ``SystemExit(0)`` only when every check passes.
Run it on the GPU box before spending training hours on a DR config::

    python -m deepracer_genesis.validation.dr_check --knobs world_color,brightness
    python -m deepracer_genesis.validation.dr_check --knobs env_map_tint --renderer nyx
"""

from __future__ import annotations

import argparse


def main(argv=None) -> None:
    """Run the prove suite for the requested knobs and exit by verdict.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Raises:
        SystemExit: 0 when every check passes, 1 otherwise.
    """
    from ..tools.dr_editor.cli import cmd_prove

    ap = argparse.ArgumentParser(prog="python -m deepracer_genesis.validation.dr_check")
    ap.add_argument("--knobs", default="world_color,brightness,pixel_noise",
                    help="comma-separated knob names to prove")
    ap.add_argument("--target", default=None)
    ap.add_argument("--track", default="reinvent_base")
    ap.add_argument("--renderer", default="batch",
                    choices=("batch", "nyx", "rasterizer"))
    ap.add_argument("--num-envs", type=int, default=6, dest="num_envs")
    ap.add_argument("--res", default="160x120")
    ap.add_argument("--waypoint", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logs/validation")
    args = ap.parse_args(argv)
    args.knobs = args.knobs  # cmd_prove reads .knobs
    raise SystemExit(cmd_prove(args))


if __name__ == "__main__":
    main()
