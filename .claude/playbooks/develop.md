# Playbook: Develop (build something new)

Use when adding new functionality — orchestrator features, simulator capabilities,
analysis tooling, or shared utilities. Layers on top of the universal standards and
loop in `CLAUDE.md`.

- **Nail the contract first.** Confirm the requirement and the interface before coding.
  If the shape isn't obvious, define the port/`Protocol` first and put it in the plan;
  implementations follow. If the change is coupled (e.g. it touches the Gymnasium
  interface the simulator and orchestrator share), LOCK that interface before fanning
  any work out.
- **Build at the extension point.** Prefer adding an adapter/strategy over editing core
  branching. A new env/algorithm/analysis backend = implement the relevant port, don't
  special-case it in the core.
- **Test alongside (or before) the code.** Cover the contract and edge cases, not just
  the happy path. Seed any randomness.
- **Keep the surface documented.** Docstrings on public APIs; extend tutorials and the
  extensibility docs whenever you add an extension point.
- **Record decisions.** Write an ADR when you choose a pattern, add a public interface,
  or pick a library.
- **RL pipeline?** Honor `.claude/context/rl-study.md` (patterns + ports) and run the
  smoke pre-flight before calling it done.
