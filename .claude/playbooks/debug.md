# Playbook: Debug (fix a defect / crash / unexpected behavior)

Use when something is broken, crashing, or behaving unexpectedly. Layers on top of the
universal standards and loop in the workspace `CLAUDE.md` (the repo's parent folder).

## Standard defect
- **Reproduce first.** Write a FAILING test that captures the bug before fixing. That
  test defines "fixed" and becomes the regression guard.
- **Isolate.** Narrow to the smallest failing unit; use logging and fixed seeds to make
  it deterministic. For RL, control the seed and prefer a fake/short env for a fast
  repro.
- **Root cause, not symptom.** State the cause in the commit/ADR if it isn't obvious.
- **Fix minimally.** No opportunistic refactors in the same change — note them for a
  separate refactor task so the fix stays reviewable.
- **Verify.** The new test passes and the full suite stays green.
- **Stuck?** If you can't reproduce or the cause stays unclear after reasonable effort,
  log a blocker with what you tried and your hypotheses, then move on to non-blocked
  work.

## Long / crashing runs (soak run + health checkpoint)
For failures that only appear in long runs — e.g. the simulator crashing around the
~30-minute mark — a short bounded test won't catch it: it may finish just before the
mark (a flaky pass), and you never see whether it survives PAST that point, which is
the whole question. Run a real SOAK run and check health at and beyond the mark.

- **Don't block the session.** Launch the run in the BACKGROUND, record the start time,
  and point it at a log + status file (per Long-running jobs in the workspace
  `CLAUDE.md`). Poll on demand — never sit in the loop waiting for it.
- **Write your own watchdog.** Create a small script (plain Python, not you) that
  monitors the process and snapshots to a status file: is the process alive? seconds
  since the last log/heartbeat line (to catch *alive-but-stuck*)? current memory and
  its trend? are the latest metrics finite? and on death, the exit code + the tail of
  the traceback. Keep it in the repo, give it a couple of tests, and reuse/extend it
  across runs.
- **Check at ~30 min and periodically after.** The simulator is known to crash near
  ~30 min, so the soak must run PAST that. Healthy = process alive AND still
  progressing (heartbeat advanced in the last few minutes) AND metrics finite. If it
  has crashed or stalled by a checkpoint, capture the status and stop babysitting.
- **Classify the failure mode — it usually cracks the bug:**
  - dead process + traceback → unhandled exception / resource exhaustion (file handles,
    zombie subprocesses).
  - alive but heartbeat frozen → deadlock / hang.
  - memory climbing steadily → leak.
  - NaN/inf in metrics → numerical blowup.
- **Turn it into a fix.** Write the failure mode, the exit code/traceback, the memory
  trend, and the step/time of last progress into the run status and `PROJECT_MEMORY.md`.
  Then reproduce it as small and fast as you can (a shorter run, a tighter seed, or a
  targeted test that forces the same condition) and proceed as a standard defect above.
