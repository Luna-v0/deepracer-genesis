"""Derive variants of a frozen ExperimentSpec by dotted-path field override.

Build variant lists with plain comprehensions over ``override``."""

from __future__ import annotations

from dataclasses import is_dataclass, replace

from .spec import ExperimentSpec, SpecError


def override(spec: ExperimentSpec, path: str, value) -> ExperimentSpec:
    """Replace one field of a frozen spec along a dotted path.

    Coupled cost-budget fields on env and lagrangian stay in sync.

    Args:
        spec: The frozen ExperimentSpec to derive from.
        path: Dotted path to the field, e.g. 'env.num_envs' or
            'algorithm.lagrangian.budget'.
        value: New value for the addressed field.

    Returns:
        A new ExperimentSpec with the field (and any coupled field) replaced.

    Raises:
        SpecError: If a path segment does not exist or cannot be descended
            into.
    """
    out = _override(spec, path, value)
    if isinstance(out, ExperimentSpec):
        if (path == "env.cost_budget" and out.algorithm is not None
                and out.algorithm.requires_cost):
            out = _override(out, "algorithm.lagrangian.budget", value)
        elif (path == "algorithm.lagrangian.budget" and out.env is not None
                and out.env.emits_cost):
            out = _override(out, "env.cost_budget", value)
    return out


def _override(spec: "ExperimentSpec", path: str, value) -> "ExperimentSpec":
    """Rebuild `spec` with the dotted-path field replaced (frozen tree walk)."""
    head, _, rest = path.partition(".")
    if not rest:
        if isinstance(spec, dict):
            out = dict(spec)
            if head not in out:
                raise SpecError(f"unknown override path segment {head!r}")
            out[head] = value
            return out
        return replace(spec, **{head: value})
    child = spec[head] if isinstance(spec, dict) else getattr(spec, head)
    new_child = _override(child, rest, value)
    if isinstance(spec, dict):
        out = dict(spec)
        out[head] = new_child
        return out
    if not is_dataclass(spec):
        raise SpecError(f"cannot descend into {type(spec).__name__} at {head!r}")
    return replace(spec, **{head: new_child})
