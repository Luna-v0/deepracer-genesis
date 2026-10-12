"""Single dispatcher resolving any experiment handle into a spec and running it."""

from __future__ import annotations

from dataclasses import fields, replace

from .authoring import Experiment
from .spec import ExperimentSpec, SpecError
from .stages import Pipeline


def build(target, **overrides) -> ExperimentSpec:
    """Resolve an experiment handle into a validated ExperimentSpec.

    Args:
        target: an ``Experiment`` class or instance, a spec-factory function,
            a ``Pipeline``, or an ``ExperimentSpec``. There is no name
            registry — pass the class itself (``run(MyExperiment)``).
        **overrides: route by name — keys matching the Experiment class's
            config attributes go to the class
            (``run(SafeTransfer, budget=10.0)``); the rest must be
            ExperimentSpec fields (``seed``, ``variant``, ...).

    Returns:
        The validated, frozen ExperimentSpec.

    Raises:
        SpecError: unknown target type, unknown override, or invalid config.
    """
    if isinstance(target, str):
        raise SpecError(
            f"experiments are referenced by class, not name (got {target!r}); "
            "pass the Experiment class, e.g. run(MyExperiment). The CLI accepts "
            "a 'module:ClassName' path.")
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


def run(target, *, root: str = "runs", build_only: bool = False, on_eval=None,
        **overrides) -> EvalRecord | ExperimentSpec:
    """Build the spec and train it (always from scratch — no result cache).

    Args:
        target: anything :func:`build` accepts.
        root: runs directory root.
        build_only: return the spec instead of training.
        on_eval: ``fn(frames, metrics)`` fired at every periodic eval —
            raise inside it to stop the run (HPO pruning).
        **overrides: forwarded to :func:`build`.

    Returns:
        The run's EvalRecord (or the ExperimentSpec when ``build_only``).
    """
    spec = build(target, **overrides)
    if build_only:
        return spec
    from .rsl_backend import rsl_supported, run_rsl
    if not rsl_supported(spec):
        raise SpecError(_unsupported_reason(spec))
    return run_rsl(spec, root=root, on_eval=on_eval)


def _unsupported_reason(spec: ExperimentSpec) -> str:
    """Explain why a spec has no backend after the TorchRL removal.

    Args:
        spec: The validated experiment spec.

    Returns:
        A message naming the feature that still needs an rsl-rl implementation.
    """
    if spec.env.emits_cost:
        return ("cost / safe-RL (PPO-Lagrangian) is not yet on the rsl-rl backend "
                "(the TorchRL path was removed); add an rsl-rl Lagrangian algorithm")
    if spec.encoder.kind != "none":
        return "frozen-CNN encoder transfer is not yet on the rsl-rl backend"
    if spec.policy.actions is not None:
        return ("discrete action tables are not yet on the rsl-rl backend "
                "(rsl-rl policies are continuous Gaussian)")
    return "this spec is not supported by the rsl-rl backend"
