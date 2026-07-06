"""run(): the single dispatcher (plan section 1.2).

Accepts a registered name, an @experiment function, an Experiment class or
instance, a Pipeline, or a raw ExperimentSpec. Heavy imports (torch/genesis)
happen only past build_only.
"""

from __future__ import annotations

from dataclasses import replace

from .registry import REGISTRY, Experiment
from .spec import ExperimentSpec, SpecError
from .stages import Pipeline


def build(target, **overrides) -> ExperimentSpec:
    """Resolve any experiment handle into a validated ExperimentSpec."""
    if isinstance(target, str):
        try:
            target = REGISTRY[target]
        except KeyError:
            raise SpecError(
                f"unknown experiment {target!r}; registered: {sorted(REGISTRY)} "
                "(did you import your experiments package?)") from None
    if isinstance(target, type) and issubclass(target, Experiment):
        target = target()
    if isinstance(target, Experiment):
        spec = target.spec()
    elif isinstance(target, Pipeline):
        spec = target.build()
    elif isinstance(target, ExperimentSpec):
        spec = target
    elif callable(target):
        spec = target()
    else:
        raise SpecError(f"cannot build an experiment from {type(target).__name__}")

    if not isinstance(spec, ExperimentSpec):
        raise SpecError(
            f"experiment handle produced {type(spec).__name__}, expected ExperimentSpec "
            "(did the function forget .build()?)")
    if overrides:
        spec = replace(spec, **overrides)
    spec.validate()
    return spec


def run(target, *, root: str = "runs", build_only: bool = False, **overrides):
    """Build the spec and train it. `run('cam_baseline', seed=3)`."""
    spec = build(target, **overrides)
    if build_only:
        return spec
    from .builder import Builder      # heavy imports live behind this line
    from .trainer import Trainer
    return Trainer(Builder(spec), root=root).fit()
