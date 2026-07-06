"""Ablation helpers (plan section 6): sweeps, grids and seed fans are plain
Python over frozen specs — comprehensions, not config files."""

from __future__ import annotations

import itertools
from dataclasses import is_dataclass, replace

from .run import build
from .spec import ExperimentSpec, SpecError


def override(spec: ExperimentSpec, path: str, value) -> ExperimentSpec:
    """dataclasses.replace along a dotted path, e.g. 'env.num_envs' or
    'algorithm.lagrangian.budget' (dict leaves are copied, not mutated)."""
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
    new_child = override(child, rest, value)
    if isinstance(spec, dict):
        out = dict(spec)
        out[head] = new_child
        return out
    if not is_dataclass(spec):
        raise SpecError(f"cannot descend into {type(spec).__name__} at {head!r}")
    return replace(spec, **{head: new_child})


def sweep(base, path: str, values, group: str | None = None) -> list[ExperimentSpec]:
    """One axis, many values -> specs auto-tagged into one ablation_group."""
    spec = build(base)
    leaf = path.rsplit(".", 1)[-1]
    group = group or f"sweep_{leaf}"
    out = []
    for v in values:
        s = override(spec, path, v)
        s = replace(s, ablation_group=group, variant=f"{leaf}={v}")
        s.validate()
        out.append(s)
    return out


def seeds(specs, k: int) -> list[ExperimentSpec]:
    """spec x range(k) -> mean +- std material for the reporter."""
    if isinstance(specs, ExperimentSpec):
        specs = [specs]
    return [replace(s, seed=seed) for s in specs for seed in range(k)]


def grid(base, axes: dict[str, list], group: str | None = None) -> list[ExperimentSpec]:
    """Cartesian product over dotted-path axes; invalid combos are dropped
    with a printed reason (e.g. nyx x heterogeneous self-excludes)."""
    spec = build(base)
    group = group or "grid_" + "_".join(p.rsplit(".", 1)[-1] for p in axes)
    paths = list(axes)
    out = []
    for combo in itertools.product(*(axes[p] for p in paths)):
        s = spec
        for p, v in zip(paths, combo):
            s = override(s, p, v)
        variant = ",".join(f"{p.rsplit('.', 1)[-1]}={v}" for p, v in zip(paths, combo))
        s = replace(s, ablation_group=group, variant=variant)
        try:
            s.validate()
        except SpecError as e:
            print(f"[grid] dropped {variant}: {e}")
            continue
        out.append(s)
    return out
