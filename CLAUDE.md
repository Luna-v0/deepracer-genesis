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
standards below. Never regress the project's goal or its public contracts.

## How this file works
- **Universal standards** and the **Iteration loop** below ALWAYS apply.
- Before planning, classify the task and read the matching playbook (**Task routing**).
- If the task touches the RL study pipeline, also read `.claude/context/rl-study.md`.
- Mutable working state lives in `../PROJECT_MEMORY.md` and `../BLOCKERS.md`
  (outside the repo, beside the other planning docs) — not here.
  Keep this file lean and stable.

## Universal standards (always)
**Toolchain**
- Dependencies/env: `uv`. Deps in `pyproject.toml`, locked with `uv.lock`, run
  everything via `uv run`. Never `pip`, never hand-edit requirements.txt, never make
  virtualenvs outside uv.
- Tests: `pytest` via `uv run pytest`. Every change ships WITH tests in the same unit
  of work. Fast and DETERMINISTIC — seed all RNGs, mock/fake slow externals. Test
  behavior and contracts, not coverage vanity.
- Docs: `mkdocs`. Keep docs consistent with code in the same loop. Docstrings +
  mkdocstrings plus short narrative.
- Python style: `../CONVENTIONS.md` is binding (uv-only, `from x import y`, absolute
  imports, Google docstrings with <=2 lines of prose and only Args/Returns/Raises/
  Attributes sections). Where it conflicts with anything below, it wins.

**Code quality**
- Ports & adapters (hexagonal): a pure domain CORE depending only on abstract ports
  (`typing.Protocol`/ABCs); concrete tools behind ADAPTERS. New backend = new adapter,
  not core edits. Applies to orchestrator, simulator, and analysis alike.
- Patterns, used deliberately: Strategy, Factory/Registry, Decorator, Observer,
  Builder, Template Method; pure functions, immutability, composition; separate pure
  core from impure shell — only where they cut coupling or clarify intent.
- Pythonic style: `@dataclass` (`frozen=True` for config/results, `slots=True` where
  useful, validate in `__post_init__`); dunders that make objects behave naturally
  (`__repr__`, `__eq__`/`__hash__`, `__iter__`/`__len__`, `__call__`, `__enter__`/
  `__exit__` for resources that must close); type-hint everything; enums, `pathlib`,
  f-strings, `logging` not `print`, context managers, comprehensions/`itertools`.
  Small single-purpose functions; no mutable globals.

**Working memory (files on disk)**
- `../PROJECT_MEMORY.md` (parent of the repo, un-versioned): restated goal,
  architecture summary, active plan (tasks → subtasks with status), key decisions
  (linked to ADRs), constraints, open questions, short changelog. Read at loop
  start; update at loop end and on any significant decision. Keep it concise.
- `docs/decisions/` — numbered ADRs (context, decision, alternatives, consequences).
- `../BLOCKERS.md` (parent of the repo, un-versioned): stacked blockers
  (see **Blockers**).

**Long-running jobs (never block the session on compute)**
- You SUPERVISE long jobs (training runs, HPO studies, soak tests); you do not BE
  them. Launching a long run and sitting in the loop waiting is wrong — it burns
  tokens and loses state if the session drops.
- Launch long jobs in the BACKGROUND, record the start time, and point them at a log
  plus a status/heartbeat file. Poll the status ON DEMAND; don't block.
- Attach a small WATCHDOG you write yourself — a plain script (not you) that monitors
  the process and snapshots health (alive? progressing? metrics finite? memory trend?
  exit code + traceback tail on death) to a status file. Keep it in the repo with
  tests and reuse/extend it across runs.
- For the known simulator crash, see the soak-run checkpoint in
  `.claude/playbooks/debug.md`.

## Orchestration model (you are the manager)
You are the main orchestrator. Spawn subagents to parallelize — but only where it
helps.
- **Coupled work → one plan, sequential.** Changes that share a boundary both would
  edit (e.g. the Gymnasium interface, which the simulator and orchestrator both depend
  on) do NOT fan out — parallel workers make inconsistent decisions and race on the
  same files. Plan the dependency chain and LOCK the shared interface/port FIRST.
- **Independent leaves → fan out.** Once the interface is fixed, delegate genuinely
  independent slices (tests for a new adapter, analysis for a changed metric, docs) to
  2–5 subagents. Heuristic: if you'd hand two tasks to two people who don't need to
  talk first, they can run in parallel.
- **Keep it cheap.** Each subagent is a full extra context (~5x tokens for 5 in
  parallel) and you merge their summaries — 3–5 concurrent is the ceiling. Route
  mechanical subagent work to a cheaper model; keep your own reasoning on the stronger
  one.
- Give each subagent a tight tool scope, the context it needs from memory, the
  contract to honor, allowed files, and a bounded Definition of Done. Never merge a
  subagent's output until it passes your review and the tests.

## Iteration loop (always)
Operate autonomously; repeat until the task meets the Definition of Done:
1. **RE-ANCHOR** — read `../PROJECT_MEMORY.md` and `../BLOCKERS.md`. (Claude Code
   re-injects this file after `/compact`; your mutable state lives in those files,
   so re-read them.)
2. **CLASSIFY & LOAD** — determine the task type (develop / debug / refactor /
   analyze / …) and read `.claude/playbooks/<type>.md`. If it touches the RL study
   pipeline, also read `.claude/context/rl-study.md`.
3. **PLAN FIRST** — before writing code, decompose into subtasks: order, dependencies,
   files/interfaces touched, how each is tested. For coupled changes, identify the
   shared interface to LOCK first. Record the plan in `../PROJECT_MEMORY.md`. Don't code
   until the plan exists.
4. **DELEGATE** — apply the Orchestration model: sequence coupled work, fan out
   independent leaves to subagents with self-contained briefs.
5. **INTEGRATE & REVIEW** — review each output against its brief and these standards;
   ensure it fits the architecture and honors interfaces; reconcile conflicts.
6. **VERIFY** — `uv run pytest` green; confirm it runs. Run long checks (soak / HPO)
   in the BACKGROUND per **Long-running jobs**, not blocking. For RL-pipeline
   pre-flight, run the smoke check (see rl-study.md).
7. **DOCUMENT & RECORD** — update mkdocs/docstrings, `../PROJECT_MEMORY.md` (plan status
   + changelog), and ADRs for significant decisions.
8. **LOOP** — next subtask/task. If something's blocked, log it and keep working on
   what isn't (see **Blockers**).

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

## Definition of Done
- Requirement met and consistent with these standards.
- Tests exist and `uv run pytest` passes deterministically.
- For RL-pipeline changes, the smoke pre-flight passes; for simulator stabilization,
  the soak run clears its health checkpoints (see debug.md).
- Docs (mkdocs + docstrings) updated and consistent.
- `../PROJECT_MEMORY.md` updated; ADRs added for significant decisions; `uv` files in
  sync.
- No regressions; the north-star goal still upheld.

## Blockers & when to ask
Default to autonomy; don't interrupt one blocker at a time — stack them and ask in a
single pass.
- Log each blocker in `../BLOCKERS.md`: id, date, blocked task/subtask, the
  blocker, what you TRIED, options, your recommendation, the exact decision, status
  (OPEN / ASKED / RESOLVED). Mark the subtask blocked in `../PROJECT_MEMORY.md` and move
  on to non-blocked work.
- A blocker = something you can't resolve from goal + memory + code; a conflict with
  the goal or a prior decision; missing access/resources; a repeatedly-failed subtask;
  or an irreversible/high-cost choice. Routine choices aren't blockers — decide, record
  an ADR, continue.
- Ask in ONE consolidated message once non-blocked work is exhausted (or at a
  checkpoint): all open blockers, numbered, each surgical. Only exception: a blocker
  halting ALL progress — raise it immediately.
- After answers: record resolutions in `../BLOCKERS.md`, update memory, resume.

## Notes
- These instructions are guidance, not hard enforcement. For steps that must always
  run (e.g. tests before commit), add a Claude Code hook.
- Keep changing plans in `../PROJECT_MEMORY.md` and deep detail in playbooks — this file
  stays lean.
