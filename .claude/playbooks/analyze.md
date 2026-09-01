# Playbook: Analyze (investigate / measure / analyze)

Use for investigation and measurement — RL results, simulator behavior, profiling,
data analysis. Often not "RL code" per se, but still a coding-and-testing task. Layers
on top of the universal standards and loop in `CLAUDE.md`.

- **Frame the question first.** State what you're trying to learn and what output
  answers it (a number, a plot, a table, a decision) before you start.
- **Pure, tested computation.** Write metric/aggregation/transform logic as PURE
  functions and test them on small known inputs, so results are reproducible and
  verifiable rather than one-off.
- **Make it reproducible.** Seed randomness, pin inputs, script the analysis (no
  throwaway manual steps), and save artifacts where they're expected.
- **Keep exploratory code separate from library code.** If an analysis utility turns
  out to be reusable, promote it into the package with tests — that becomes a separate
  develop task.
- **Report honestly.** State findings in your own words with the numbers that support
  them; note assumptions, sample sizes, and uncertainty; don't overclaim.
- **Record outcomes.** Write conclusions and any resulting decisions into
  `PROJECT_MEMORY.md`, and an ADR if the analysis changes direction.
