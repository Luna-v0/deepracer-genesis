# Playbook: Refactor (restructure without changing behavior)

Use when improving structure, readability, or design with no change in behavior. Layers
on top of the universal standards and loop in the workspace `CLAUDE.md` (the repo's
parent folder).

- **Precondition: a green safety net.** You need tests covering the behavior you're
  about to move. If coverage is thin, ADD characterization tests FIRST, then refactor.
- **Behavior must not change.** Public contracts and outputs stay identical; the suite
  stays green throughout. Work in small, reversible steps.
- **This is the place for structure work.** Introduce/clean up patterns and the
  ports/adapters boundary — move concrete deps behind adapters, replace branching with
  Strategy/Registry, extract pure functions out of the impure shell.
- **Don't smuggle in features or fixes.** Spot a bug or a missing feature → log/track it
  separately rather than bundling it here.
- **Keep docs in step.** Update diagrams and docs to match the new structure; record an
  ADR for structural decisions.
- **RL pipeline?** Re-run the smoke pre-flight afterward — same results, no performance
  regression.
