"""Ablation helpers (plan section 6): sweeps, grids and seed fans are plain
Python over frozen specs — comprehensions, not config files."""

from __future__ import annotations

import itertools
from dataclasses import is_dataclass, replace

from .run import build
from .spec import ExperimentSpec, SpecError


def override(spec: ExperimentSpec, path: str, value) -> ExperimentSpec:
    """Replace one field of a frozen spec along a dotted path.

    dataclasses.replace along a dotted path, e.g. 'env.num_envs' or
    'algorithm.lagrangian.budget' (dict leaves are copied, not mutated).
    Coupled fields stay in sync: the cost budget lives on the env AND in the
    inferred lagrangian config — changing either updates both.

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
                and out.algorithm.kind == "ppo_lagrangian"):
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


def sweep(base, path: str, values, group: str | None = None) -> list[ExperimentSpec]:
    """Sweep one axis over many values.

    One axis, many values -> specs auto-tagged into one ablation_group.

    Args:
        base: Any experiment handle accepted by run.build (registered name /
            function / class / spec / pipeline).
        path: Dotted override path (see override()).
        values: Values to place at `path`, one spec per value.
        group: Ablation group name; defaults to 'sweep_<leaf>'.

    Returns:
        List of validated ExperimentSpecs, each tagged with the
        ablation_group and a '<leaf>=<value>' variant.
    """
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
    """Fan specs out across seeds.

    spec x range(k) -> mean +- std material for the reporter.

    Args:
        specs: A single ExperimentSpec or a list of them.
        k: Number of seeds; each spec is replicated with seed = 0..k-1.

    Returns:
        List of len(specs) * k specs.
    """
    if isinstance(specs, ExperimentSpec):
        specs = [specs]
    return [replace(s, seed=seed) for s in specs for seed in range(k)]


def grid(base, axes: dict[str, list], group: str | None = None) -> list[ExperimentSpec]:
    """Cartesian product over dotted-path axes.

    Invalid combinations are dropped with a printed reason (e.g. nyx x
    heterogeneous self-excludes).

    Args:
        base: Any experiment handle accepted by run.build.
        axes: Mapping of dotted override path -> list of values.
        group: Ablation group name; defaults to 'grid_<leaf1>_<leaf2>_...'.

    Returns:
        List of validated specs, one per surviving combination, each tagged
        with the ablation_group and a 'k1=v1,k2=v2' variant.
    """
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
