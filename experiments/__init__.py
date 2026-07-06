"""Authored experiments — this package IS the config surface (pure Python).

Importing it fires the @experiment / Experiment-subclass registrations, so
`run("cam_baseline")` works after `import experiments`.
"""

from . import camera, feature, safe, template  # noqa: F401  (registrations fire on import)
