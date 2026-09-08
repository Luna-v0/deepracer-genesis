# CLAUDE.md — deepracer-genesis (Genesis simulator + batched RL env + analysis)

## North-star goal
Primary objective: a clean, extensible port of the AWS DeepRacer RL environment from
ROS/Gazebo to Genesis — ROS-free, GPU-batched, vision-based, trained with rsl-rl-lib
PPO — where a researcher configures a run, trains it, and analyzes results, and new
tracks / renderers / algorithms are added by implementing a small, well-documented
interface. Reproducibility, testability, and clear documentation take priority over
feature count. The original DeepRacer action/observation semantics and the rsl-rl 5.x
VecEnv contract are invariants — never silently regress them.

The system has coupled components: the **Genesis simulator** (physics + the three
vision renderers: Madrona batch, Nyx, per-env rasterizer), the **batched VecEnv
environment** that wraps it, the **train/eval orchestration**, and **analysis**
(benchmarks, validation, telemetry). Their tightest coupling is the VecEnv/observation
interface — a change there ripples across the env, the renderers, and training. Not
every task is about RL, but EVERY task is a coding-and-testing task held to the
standards in `../CLAUDE.md`. Never regress the project's goal or its public contracts.

## How this file works
- **`../CLAUDE.md` (the workspace manager) always applies** — universal standards, the
  orchestration model, the iteration loop, long-running jobs, and the blocker policy
  live there and are not repeated here. This file adds only what is specific to this
  repo. Where they disagree, this file wins for work inside this repo.
- Before planning, classify the task and read the matching playbook (**Task routing**).
- If the task touches the RL study pipeline, also read `.claude/context/rl-study.md`.
- Mutable working state lives in `../PROJECT_MEMORY.md`, `../BLOCKERS.md` and
  `../PROBLEMS.md` — not here. Keep this file lean and stable.

## Repo specifics
- `../CONVENTIONS.md` is the binding Python style for this repo (there is no local
  copy — do not add one).
- Accepted decisions land in `docs/decisions/` as numbered ADRs (context, decision,
  alternatives, consequences). `../adr-0001-perception.md` is the perception split:
  library code in `deepracer_genesis/perception/`, run drivers in `experiments/`.
- This repo is upstream of `../deepracer-studies`. Public symbols it consumes are a
  contract — see **Cross-repo rules** in `../CLAUDE.md` before changing one.
- For the known simulator crash, see the soak-run checkpoint in
  `.claude/playbooks/debug.md`.

## Task routing
Classify each task and read the matching playbook before planning:
- Building something new → `.claude/playbooks/develop.md`
- Fixing a defect / crash / unexpected behavior → `.claude/playbooks/debug.md`
- Restructuring without behavior change → `.claude/playbooks/refactor.md`
- Investigating / measuring / analyzing → `.claude/playbooks/analyze.md`
- Other recurring types → add a playbook and list it here.

Task type is orthogonal to domain: the same workflow applies whether the code is the
orchestrator, the simulator, or analysis — only `rl-study.md` adds RL specifics. Tasks
that span types (refactor then extend) read each relevant playbook and sequence in the
plan.

## Definition of Done (adds to the workspace baseline)
- For RL-pipeline changes, the smoke pre-flight passes (see `rl-study.md`).
- For simulator stabilization, the soak run clears its health checkpoints (see
  `debug.md`).
- The rsl-rl 5.x VecEnv contract and the DeepRacer action/observation semantics are
  unchanged, or the change is deliberate and recorded in an ADR.
