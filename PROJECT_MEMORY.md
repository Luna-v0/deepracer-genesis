# PROJECT_MEMORY.md

Mutable working state for deepracer-genesis. Read at loop start, update at loop end and
on any significant decision. Keep concise — prune stale content. Stable standards live
in `CLAUDE.md`; blockers live in `BLOCKERS.md`; decisions live in `docs/decisions/`.

## Goal (restated)
Port the AWS DeepRacer RL environment from ROS/Gazebo to Genesis: ROS-free, GPU-batched,
vision-based, trained with rsl-rl-lib PPO. Original action/observation semantics and the
rsl-rl 5.x VecEnv contract are invariants.

## Architecture summary
- `deepracer_genesis/envs/deepracer_env.py` — batched VecEnv-style environment.
- `deepracer_genesis/envs/track.py` — track registry + GPU waypoint geometry.
- `deepracer_genesis/configs/cfgs.py` — env cfg + rsl-rl 5.x train cfg.
- `deepracer_genesis/train.py` / `eval.py` — training and evaluation entry points.
- `deepracer_genesis/randomization/domain_rand.py` — domain randomization.
- `deepracer_genesis/validation/camera_check.py` — paired onboard/topdown image checks.
- `benchmarks/throughput.py` — throughput sweep -> `results.csv` / `results.md`.
- Three vision renderers: Madrona batch (default, training), Nyx (correct-color eval),
  per-env rasterizer (videos/debug). See `README.md` and `TUTORIAL.md`.

## Active plan
_(tasks -> subtasks with status; nothing recorded yet)_

## Key decisions
_(link each to an ADR in `docs/decisions/`; nothing recorded yet)_

## Constraints
- `uv` only; `../CONVENTIONS.md` is binding for Python style.
- Nyx: OBJ not DAE, unmerged URDF links, denoise/AA off, driver 575+, single-track scenes.
- Long runs are supervised in the background, never awaited in-session (see `CLAUDE.md`).

## Open questions
_(none recorded yet)_

## Changelog
- 2026-09-01 — Installed `CLAUDE.md`, `.claude/playbooks/`, `.claude/context/rl-study.md`,
  and seeded this file, `BLOCKERS.md`, and `docs/decisions/`.
