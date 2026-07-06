"""CLI for running registered experiments.

    python -m deepracer_genesis.experiment feature_baseline
    python -m deepracer_genesis.experiment SafeTransfer --set budget=10 --seed 2
    python -m deepracer_genesis.experiment cam_baseline --steps 2000000 --video
    python -m deepracer_genesis.experiment --list
    python -m deepracer_genesis.experiment --report

Experiments register on `import experiments` (the authored package in the
repo root / cwd); name lookup is a handle to Python code, never a file path.
"""

from __future__ import annotations

import argparse
import ast
import sys


def _parse_set(pairs: list[str]) -> dict:
    """--set key=value pairs; values parsed as Python literals when possible."""
    out = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        if not _:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        try:
            out[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[key] = raw
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m deepracer_genesis.experiment",
                                     description=__doc__)
    parser.add_argument("name", nargs="?", help="registered experiment name")
    parser.add_argument("--list", action="store_true", help="list registered names")
    parser.add_argument("--report", action="store_true",
                        help="regenerate runs/report.md from stored records")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None,
                        help="override total_env_steps")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="override eval_every_steps")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                        help="extra overrides (Experiment attrs or spec fields)")
    parser.add_argument("--root", default="runs")
    parser.add_argument("--force", action="store_true",
                        help="retrain even on an identity-cache hit")
    parser.add_argument("--video", action="store_true",
                        help="after training, record a spectator rollout video")
    parser.add_argument("--track", default=None,
                        help="with --video: evaluate on this track instead")
    args = parser.parse_args(argv)

    try:
        import experiments  # noqa: F401  (registrations fire)
    except ImportError:
        print("warning: no `experiments` package importable from cwd", file=sys.stderr)

    from .registry import REGISTRY

    if args.list:
        for name in sorted(REGISTRY):
            print(name)
        return 0
    if args.report:
        from .report import build_report
        build_report(args.root)
        print(f"wrote {args.root}/report.md and {args.root}/report.csv")
        return 0
    if not args.name:
        parser.error("experiment name required (or --list / --report)")

    overrides = _parse_set(args.set)
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.steps is not None:
        overrides["total_env_steps"] = args.steps
    if args.eval_every is not None:
        overrides["eval_every_steps"] = args.eval_every

    from .run import run
    record = run(args.name, root=args.root, force=args.force, **overrides)

    if args.video:
        from .visualize import rollout_video
        path = rollout_video(args.name, root=args.root, track=args.track,
                             **overrides)
        print(f"video: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
