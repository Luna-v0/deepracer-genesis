"""DR / scene editor: inspect, tune, and prove domain-randomization knobs.

Three jobs, one catalog-driven core:

- **inspect** — render what a config looks like (onboard / top-down /
  spectator / pipeline stages), synchronized to one sim instant.
- **tune** — sweep a knob across its range at a fixed pose, pick a range by
  eye, get back pasteable ``>>`` stage code.
- **prove** — assert a knob has a measurable effect on the frames the policy
  consumes and varies on the axis it claims (also CI-able via
  ``python -m deepracer_genesis.validation.dr_check``).

CLI: ``python -m deepracer_genesis.tools.dr_editor <command>``. Import weight:
this package imports genesis lazily (only :mod:`.session` touches it, inside
functions), so the offline tier (pipeline replay, sheets, frame banks, emit)
runs on any machine with torch + PIL.
"""

__all__ = ["knobs", "pipeline", "session", "sheets", "prove", "emit", "frames"]
