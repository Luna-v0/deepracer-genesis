"""run(): the single dispatcher (plan section 1.2).

Accepts a registered name, an @experiment function, an Experiment class or
instance, a Pipeline, or a raw ExperimentSpec. Heavy imports (torch/genesis)
happen only past build_only.
"""

from __future__ import annotations

from dataclasses import fields, replace

from .registry import REGISTRY, Experiment
from .spec import ExperimentSpec, SpecError
from .stages import Pipeline


def build(target, **overrides) -> ExperimentSpec:
    """Resolve any experiment handle into a validated ExperimentSpec.

    Overrides route by name: keys matching the Experiment class's config
    attributes go to the class (`run("SafeTransfer", budget=10.0)`); the rest
    must be ExperimentSpec fields (`seed`, `variant`, ...).
    """
    if isinstance(target, str):
        try:
            target = REGISTRY[target]
        except KeyError:
            raise SpecError(
                f"unknown experiment {target!r}; registered: {sorted(REGISTRY)} "
                "(did you import your experiments package?)") from None
    if isinstance(target, type) and issubclass(target, Experiment):
        cls_kw = {k: overrides.pop(k) for k in list(overrides) if hasattr(target, k)}
        target = target(**cls_kw)
    if isinstance(target, Experiment):
        for k in list(overrides):
            if hasattr(type(target), k):
                setattr(target, k, overrides.pop(k))
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
        try:
            spec = replace(spec, **overrides)
        except TypeError:
            valid = sorted(f.name for f in fields(ExperimentSpec))
            raise SpecError(
                f"unknown override(s) {sorted(overrides)} for this experiment; "
                f"ExperimentSpec fields are {valid}") from None
    spec.validate()
    return spec


def run(target, *, root: str = "runs", build_only: bool = False,
        force: bool = False, **overrides) -> "EvalRecord | ExperimentSpec":
    """Build the spec and train it. `run('cam_baseline', seed=3)`;
    `force=True` retrains even when the identity cache has a record."""
    spec = build(target, **overrides)
    if build_only:
        return spec
    from .builder import Builder      # heavy imports live behind this line
    from .trainer import Trainer
    return Trainer(Builder(spec), root=root).fit(force=force)
